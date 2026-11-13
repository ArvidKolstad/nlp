import math

import torch
from ax_modules import A1Tokenizer
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""

    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
        intermediate_size=None,
        num_attention_heads=None,
        num_hidden_layers=None,
        rope_theta=None,
        hidden_act="silu",
        max_position_embeddings=None,
        rms_norm_eps=None,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id


class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""

    def __init__(self, config):
        super().__init__()
        assert config.hidden_act == "silu"
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w3(self.silu(self.w1(hidden_states)) * self.w2(hidden_states))


# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""

    def __init__(self, config):
        super().__init__()
        self.rms_norm_eps: float = config.rms_norm_eps
        self.gain = nn.Parameter(torch.ones(config.hiddes_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(
            hidden_states.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps
        )
        return hidden_states / rms * self.gain


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""

    def __init__(self, config):
        super().__init__()
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_normalizer = A2RMSNorm(config)
        self.k_normalizer = A2RMSNorm(config)
        assert config.hidden_size % config.num_attention_heads == 0, (
            "dim must be divisable by number of attention heads "
        )
        self.n_h = config.num_attention_heads
        self.d_h = config.hidden_size // config.num_attention_heads

    def forward(self, hidden_states, rope_rotations) -> torch.Tensor:
        b, m, d = hidden_states.shape

        q = self.W_q(hidden_states)
        k = self.W_k(hidden_states)
        v = self.W_v(hidden_states)

        q = self.q_normalizer(q)
        k = self.k_normalizer(k)

        q = q.view(b, m, self.n_h, self.d_h).transpose(1, 2)
        k = k.view(b, m, self.n_h, self.d_h).transpose(1, 2)
        v = v.view(b, m, self.n_h, self.d_h).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, rope_rotations)

        attention_state = self_attn(q, k, v)
        attention_state = attention_state.transpose(1, 2).reshape(b, m, d)

        return self.W_o(attention_state)


def self_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m, d_h = q.shape[-2], q.shape[-1]
    softmax = nn.Softmax(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)
    temporal_mask = torch.triu(
        torch.ones(m, m, dtype=torch.bool, device=scores.device), diagonal=1
    )
    scores = scores.masked_fill(temporal_mask, float("-inf"))
    return softmax(scores) @ v


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""

    def __init__(self, config):
        super().__init__()
        self.attention_layer = A2Attention(config)
        self.attn_norm = A2RMSNorm(config)
        self.mlp = A2MLP(config)
        self.mlp_norm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations) -> torch.Tensor:

        x = (
            self.attn_norm(self.attention_layer(hidden_states.clone(), rope_rotations))
            + hidden_states
        )

        return self.mlp_norm(self.mlp(x.clone())) + x


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""

    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        # TODO: put all transformer decoder layers in a ModuleList.
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer_layers = nn.ModuleList(
            [A2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.unembedding_layer = nn.Linear(config.hidden_size, config.vocab_size)

        self.norm_layer = A2RMSNorm(config)
        self.pad_token_id = config.pad_token_id

        # This line should be called after you have set up all components.
        self.post_init()

    def forward(self, labels):
        """The forward pass of the Transformer language model.

        Args:
          labels:  The input tensor (2D), consisting of a batch of integer-encoded texts.
        Returns:
          CausalLMOutputWithPast object that stores loss and logits (the output tensor (3D) for all token positions for all vocabulary items.)
        """

        loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        X = labels[..., :-1, :]
        Y = labels[..., 1:, :].reshape(-1)

        rope_rotations = self.rotary_emb(
            X
        )  # pass this to all the transformer decoder layers

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        output = self.embedding_layer(X)
        for layer in self.transformer_layers:
            output = layer(output, rope_rotations)

        output = self.norm_layer(output)
        logits = self.unembedding_layer(output)
        loss = loss_func(logits.reshape(-1, logits.size(-1)), Y)
        # TODO: compute the loss for the model output and Y
        return CausalLMOutputWithPast(loss=loss, logits=logits)


#### RoPE implementation (copied and simplified from HuggingFace). ####


def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert q.shape == k.shape
    assert len(q.shape) == 4
    cos, sin = rope_rotations
    assert q.shape[2] == cos.shape[1]
    assert q.shape[3] == cos.shape[2]
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin


def main():
    print("Hello World")
    tokenizer = A1Tokenizer()


if __name__ == "__main__":
    main()
