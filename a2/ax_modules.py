import pickle
from collections import Counter

import matplotlib.pyplot as plt

#### Hassan Abla, Filip Björnåsen, Simon Eriksson, Arvid Kolstad
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BatchEncoding


def split_into_bytes(text):
    """Split a string into individual UTF-8 byte tokens.
    Args:
            text: The input string to encode.

        Returns:
            A list containing one ``bytes`` object for each byte in the text.
    """
    text_encoded = text.encode("utf-8")
    bytes_list = []

    for byte in text_encoded:
        bytes_list.append(bytes([byte]))

    return bytes_list


def count_pairs(sequences):
    pair_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1
    return pair_counts


def merge_tokens(sequences, merge):
    src = merge
    dst = merge[0] + merge[1]

    for seq in sequences:
        i = 0
        while i < len(seq) - 1:
            if (seq[i], seq[i + 1]) == src:
                seq[i : i + 2] = [dst]
                i += 1
            else:
                i += 1

    return sequences


def build_tokenizer(
    train_file,
    max_voc_size=None,
    model_max_length=None,
    pad_token="<PAD>",
    unk_token="<UNK>",
    bos_token="<BOS>",
    eos_token="<EOS>",
):
    """Build a tokenizer from the given file.

    Args:
         train_file:        The name of the file containing the training texts.
         max_voc_size:      The maximally allowed size of the vocabulary.
         model_max_length:  Truncate texts longer than this length.
         pad_token:         The dummy string corresponding to padding.
         unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
         bos_token:         The dummy string corresponding to the beginning of the text.
         eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, using a byte-level BPE tokenization (explained above)
    # Then return a tokenizer object (implemented below).

    with open(train_file, "r", encoding="utf-8") as f:
        texts = f.readlines()

    bytes_sequences = [split_into_bytes(line.strip()) for line in texts]

    special_tokens = [pad_token, unk_token, bos_token, eos_token]

    vocabulary = {token: idx for idx, token in enumerate(special_tokens)}

    for byte_idx in range(256):
        vocabulary[bytes([byte_idx])] = len(vocabulary)

    merges = []

    if max_voc_size is not None and max_voc_size < len(
        vocabulary
    ):  # ör detta nödvändigt?
        raise ValueError(
            "Vocabulary size must be at least 260 (4 special tokens + 256 bytes)"
        )

    while max_voc_size is not None and len(vocabulary) < max_voc_size:
        pair_counts = count_pairs(bytes_sequences)

        if not pair_counts:
            break

        merge = max(pair_counts, key=pair_counts.get)
        merges.append(merge)

        bytes_sequences = merge_tokens(bytes_sequences, merge)

        new_token = merge[0] + merge[1]
        vocabulary[new_token] = len(vocabulary)

    return A1Tokenizer(
        vocabulary, merges, model_max_length, pad_token, unk_token, bos_token, eos_token
    )


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(
        self,
        vocabulary,
        merges,
        model_max_length,
        pad_token="<PAD>",
        unk_token="<UNK>",
        bos_token="<BOS>",
        eos_token="<EOS>",
    ):
        # TODO: store all values you need in order to implement __call__ below.
        self.vocabulary = vocabulary
        self.inverse_vocabulary = {idx: token for token, idx in vocabulary.items()}
        self.merges = merges

        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.pad_token_id = vocabulary[pad_token]  # Compulsory attribute.
        self.unk_token_id = vocabulary[unk_token]
        self.bos_token_id = vocabulary[bos_token]
        self.eos_token_id = vocabulary[eos_token]

        self.model_max_length = model_max_length  # Needed for truncation.

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.

        Args:
          texts:           The texts to tokenize.
          truncation:      Whether the texts should be truncated to model_max_length.
          padding:         Whether the tokenized texts should be padded on the right side.
          return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

        Returns:
          A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != "pt":
            raise ValueError("Should be pt")

        byte_sequences = [split_into_bytes(text.strip()) for text in texts]
        for merge in self.merges:
            byte_sequences = merge_tokens(byte_sequences, merge)
        encoded_sequences = []
        for seq in byte_sequences:
            encoded_seq = []
            for token in seq:
                if token in self.vocabulary:
                    encoded_seq.append(self.vocabulary[token])
                else:
                    encoded_seq.append(self.vocabulary[self.unk_token])
            encoded_sequences.append(encoded_seq)

        if truncation and self.model_max_length > 2:
            encoded_sequences = [
                seq[: self.model_max_length - 2] for seq in encoded_sequences
            ]

        encoded_sequences = [
            [self.vocabulary[self.bos_token]] + seq + [self.vocabulary[self.eos_token]]
            for seq in encoded_sequences
        ]

        if padding:
            max_length = max(len(seq) for seq in encoded_sequences)
            encoded_sequences = [
                seq + [self.pad_token_id] * (max_length - len(seq))
                for seq in encoded_sequences
            ]
            attention_mask = [
                [1 if token_id != self.pad_token_id else 0 for token_id in seq]
                for seq in encoded_sequences
            ]
        else:
            attention_mask = [[1] * len(seq) for seq in encoded_sequences]

        if return_tensors == "pt":
            encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return BatchEncoding(
            {"input_ids": encoded_sequences, "attention_mask": attention_mask}
        )

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.vocabulary)

    def decode(self, ids):
        """Return a string decoded from given input ids"""
        special_token_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = []
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        for id in ids:
            if id in special_token_ids:
                continue
            if id in self.inverse_vocabulary:
                token = self.inverse_vocabulary[id]
            else:
                token = self.inverse_vocabulary[self.unk_token_id]
            tokens.append(token)
        text_bytes = b"".join(tokens)
        decoded_text = text_bytes.decode("utf-8")
        return decoded_text

    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, "rb") as f:
            return pickle.load(f)


class TrainingArguments:
    def __init__(
        self,
        optim: str = "adamw_torch",
        eval_strategy: str = "epoch",
        use_cpu: bool = False,
        learning_rate: float = 1e-3,
        num_train_epochs: int = 100,
        per_device_train_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        output_dir: str = "./models/new_model",
        collator=None,
    ):
        self.optim = optim
        self.eval_strategy = eval_strategy
        self.use_cpu = use_cpu
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.output_dir = output_dir
        self.collator = collator


class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        train_dataset: dict,
        eval_dataset: dict,
    ):
        """Set up the trainer.

        Args:
          model:          The model to train.
          args:           The training parameters stored in a TrainingArguments object.
          train_dataset:  The dataset containing the training documents.
          eval_dataset:   The dataset containing the validation documents.
          processing_class:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        assert args.optim == "adamw_torch"
        assert args.eval_strategy == "epoch"

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print("Device:", device)
        self.model.to(device)

        best_loss = 1e8
        train_loss: list[float] = []
        train_perplex: list[float] = []

        val_loss: list[float] = []
        val_perplex: list[float] = []
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate
        )

        if self.args.collator:
            train_loader = DataLoader(
                self.train_dataset["input_ids"],
                batch_size=args.per_device_train_batch_size,
                collate_fn=args.collator,
            )
            val_loader = DataLoader(
                self.eval_dataset["input_ids"],
                batch_size=args.per_device_eval_batch_size,
                collate_fn=args.collator,
            )
        else:
            train_loader = DataLoader(
                self.train_dataset["input_ids"],
                batch_size=args.per_device_train_batch_size,
            )
            val_loader = DataLoader(
                self.eval_dataset["input_ids"],
                batch_size=args.per_device_eval_batch_size,
            )
        for epoch in range(args.num_train_epochs):
            print(epoch)
            self.model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                model_outputs = self.model(batch)
                total_train_loss += model_outputs.loss.item()
                model_outputs.loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_train_p = np.exp(avg_train_loss)
            print(f"Mean train loss: {avg_train_loss}, Perplexity {avg_train_p}")

            train_loss.append(avg_train_loss)
            train_perplex.append(avg_train_p)

            total_val_loss = 0.0
            self.model.eval()
            for batch in val_loader:
                batch = batch.to(device)
                total_val_loss += self.model(batch).loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_p = np.exp(avg_val_loss)
            print(f"Mean val loss: {avg_val_loss}, Perplexity {avg_val_p}")
            val_loss.append(avg_val_loss)
            val_perplex.append(avg_val_p)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                self.model.save_pretrained(args.output_dir)
                print(f"Saving to {args.output_dir} with val loss {avg_val_loss}.")

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        axes[0].plot(train_loss, color="blue", label="Training")
        axes[0].plot(val_loss, color="red", label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(train_perplex, color="blue", label="Training")
        axes[1].plot(val_perplex, color="red", label="Validation")
        axes[1].set_title("Perplexity")
        axes[1].set_ylabel("Perplexity")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig("./latest_training.pdf")
        plt.close(fig)


class TextCollator:
    def __init__(self, tokenizer):
        self.token_id: int = tokenizer.pad_token_id

    def __call__(self, samples):
        max_length = max(len(x) for x in samples)
        padded = []
        for sample in samples:
            total_padding = max_length - len(sample)
            padding = torch.full((total_padding,), self.token_id, dtype=torch.long)
            padded.append(torch.cat((sample, padding)))
        return torch.stack(padded)
