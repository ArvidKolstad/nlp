import pickle
from itertools import islice

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

###
### Part 1. Tokenization.
###

# Byte-pair encoding (BPE): the main idea
# ---------------------------------------
# A token is one piece of text (not necessarily a word/subword/"token").
# In this tokenizer, every token is stored as one
# or more bytes. A pair means two tokens that appear next to each other, and a
# merge means joining such a pair to make one longer token.
#
# Training BPE:
#   1. Split every line of the training text into individual UTF-8 bytes.
#   2. Start the vocabulary with the four special tokens and all 256 possible
#      byte values. The vocabulary maps each token to an integer ID.
#   3. Count how often each pair of neighboring tokens occurs.
#   4. Join the most common pair everywhere it occurs. Add the joined token to
#      the vocabulary and save the pair as a merge rule.
#   5. Repeat steps 3 and 4 until the requested vocabulary size is reached
#      (NB: make sure it is provided).

#
# Encoding new text:
#   1. Remove whitespace from the beginning and end.
#   2. Split the text into individual UTF-8 bytes.
#   3. Apply the saved merge rules in the same order in which they were learned.
#   4. Replace the resulting tokens with their vocabulary IDs.
#   5. Add BOS/EOS IDs, and optionally truncate or pad the sequence.
#
# This simple version treats a whole non-empty line as one sequence. Therefore,
# it may learn tokens that contain spaces or parts of more than one word.


def split_into_bytes(text: str) -> list[bytes]:
    """Split a string into individual UTF-8 byte tokens.

    Args:
        text: The input string to encode.

    Returns:
        A list containing one ``bytes`` object for each byte in the text.
    """
    byte_list = [bytes(c, "utf-8") for c in text]
    return byte_list


def build_tokenizer(
    train_file: str,
    max_number_of_merges: int = 0,
    model_max_length: int | None = None,
    pad_token="<PAD>",
    unk_token="<UNK>",
    bos_token="<BOS>",
    eos_token="<EOS>",
):
    """Build a tokenizer from the given file.

    Args:
         train_file:        The name of the file containing the training texts.
         max_number_of_merges:      The maximally allowed size of the total merges in the vocabular.
         model_max_length:  Truncate texts longer than this length.
         pad_token:         The dummy string corresponding to padding.
         unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
         bos_token:         The dummy string corresponding to the beginning of the text.
         eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, using a byte-level BPE tokenization (explained above)
    # Then return a tokenizer object (implemented below).

    vocabular = [bytes([b]) for b in range(256)] + [
        bytes(pad_token, "utf-8"),
        bytes(unk_token, "utf-8"),
        bytes(bos_token, "utf-8"),
        bytes(eos_token, "utf-8"),
    ]
    max_length_voc = len(vocabular) + max_number_of_merges
    encoding: list[list[list[bytes]]] = []

    with open(train_file, encoding="utf-8") as f:
        for line in f:
            encoding.append(
                [
                    [b" "] + [bytes([b]) for b in word.encode("utf-8")]
                    for word in line.split()
                ]
            )

    while len(vocabular) < max_length_voc:
        print(len(vocabular))
        # Find most used token combination
        counter: dict[bytes, int] = {}
        for line in encoding:
            for word in line:
                number_of_bytes = len(word)
                for idx in range(number_of_bytes - 1):
                    if counter.get(word[idx] + word[idx + 1]):
                        counter[word[idx] + word[idx + 1]] += 1
                    else:
                        counter[word[idx] + word[idx + 1]] = 1

        most_common_pair = max(counter, key=lambda x: counter[x])

        # Add this to the vocabular
        vocabular.append(most_common_pair)

        for line in encoding:
            for word in line:
                merge_count = 0
                number_of_bytes = len(word)
                for idx in range(number_of_bytes - 1):
                    if (number_of_bytes - 1) - merge_count <= idx:
                        break

                    if (word[idx] + word[idx + 1]) == most_common_pair:
                        word[idx + 1] = most_common_pair
                        word.pop(idx)
                        merge_count += 1

    voc_map = {token: idx for idx, token in enumerate(vocabular)}
    return A1Tokenizer(model_max_length, voc_map)


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(
        self,
        model_max_length: int | None,
        tokens: dict[bytes, int],
        pad_token: bytes = b"<PAD>",
        unk_token: bytes = b"<UNK>",
        bos_token: bytes = b"<BOS>",
        eos_token: bytes = b"<EOS>",
    ):
        self.pad_token = pad_token  # Compulsory attribute.
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.model_max_length = model_max_length  # Needed for truncation.
        self.vocabular = tokens
        self.inv_vocab = {v: k for k, v in tokens.items()}
        self.n_single_tokens = 260

    def __call__(
        self,
        texts: list[str],
        truncation: bool = False,
        padding: bool = False,
        return_tensors: str | None = None,
    ):
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

        # TODO: Your work here is to split the texts into words and map them to integer values.
        #
        # - If `truncation` is set to True, the length of the encoded sequences should be
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.

        token_list: list[list[bytes]] = [[] for _ in texts]
        id_list: list[list[int]] | torch.Tensor = [[] for _ in texts]
        for no_text, text in enumerate(texts):
            for character in text.strip():
                try:
                    data = character.encode("utf-8")
                except UnicodeEncodeError:
                    token_list[no_text].append(self.unk_token)
                    continue
                token_list[no_text] += [bytes([b]) for b in data]

            for token in islice(self.vocabular.keys(), self.n_single_tokens, None):
                tokens = token_list[no_text]
                new_token_list: list[bytes] = []
                n = len(tokens)
                idx = 0
                while idx < n:
                    if idx < n - 1 and tokens[idx] + tokens[idx + 1] == token:
                        new_token_list.append(token)
                        idx += 2
                    else:
                        new_token_list.append(tokens[idx])
                        idx += 1

                token_list[no_text] = new_token_list

            token_list[no_text] = (
                [self.bos_token] + token_list[no_text] + [self.eos_token]
            )
            id_list[no_text] = [self.vocabular[token] for token in token_list[no_text]]

        if truncation:
            assert self.model_max_length, "model_max must be an integer"
            for no_text, text_vector in enumerate(id_list):
                if len(text_vector) > self.model_max_length:
                    id_list[no_text] = text_vector[: self.model_max_length - 1] + [
                        self.vocabular[self.eos_token]
                    ]

        if padding:
            max_length = max(len(x) for x in id_list)
            padding_idx = self.vocabular[self.pad_token]
            attention_mask = [[] for _ in id_list]
            for no_text, text_vector in enumerate(id_list):
                total_padding = max_length - len(text_vector)
                padding_vector = [padding_idx for i in range(total_padding)]
                attention_mask[no_text] = [1 for _ in text_vector] + [
                    0 for _ in padding_vector
                ]

                id_list[no_text] += padding_vector
        else:
            attention_mask = [[1 for _ in text_vector] for text_vector in id_list]

        if return_tensors:
            try:
                id_list = torch.tensor(id_list)
                attention_mask = torch.tensor(attention_mask)
            except ValueError:
                raise ValueError(
                    "Text generates different length vectors, concider turning on padding"
                )

        return BatchEncoding({"input_ids": id_list, "attention_mask": attention_mask})

    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocabular)

    def decode(self, ids: list[int]) -> str:
        """Return a string decoded from given input ids"""
        byte_encoding = b"".join([self.inv_vocab[id] for id in ids])
        return byte_encoding.decode("utf-8")

    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, "rb") as f:
            return pickle.load(f)


###
### Part 3. Defining the model.
###


class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""

    def __init__(
        self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size: int | None = vocab_size
        self.hidden_size: int | None = hidden_size
        self.embedding_size: int | None = embedding_size


class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""

    config_class = A1RNNModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = ...
        self.rnn = ...
        self.unembedding = ...
        self.post_init()  # for compatibility

    def forward(self, labels):
        """The forward pass of the RNN-based language model.

        Args:
          X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
        Returns:
          CausalLMOutputWithPast object that stores loss and logits (the output tensor (3D) for all token positions for all vocabulary items.)
        """

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        X = ...  # all columns in input_ids except the last one
        Y = ...  # all columns in input_ids except the first one

        assert X.shape == Y.shape

        embedded = ...
        rnn_out, _ = ...
        out = ...

        #  compute the loss for the model output and Y
        return CausalLMOutputWithPast(...)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size:
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.


class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, processing_class):
        """Set up the trainer.

        Args:
          model:          The model to train.
          args:           The training parameters stored in a TrainingArguments object.
          train_dataset:  The dataset containing the training documents.
          eval_dataset:   The dataset containing the validation documents.
          eval_dataset:   The dataset containing the validation documents.
          processing_class:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

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

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(...)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(...)
        val_loader = DataLoader(...)

        # TODO: Your work here is to implement the training loop.
        #
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       labels = apply your tokenizer to B
        #       replace padding tokens with value -100 so loss function will ignore them
        #       put labels onto the GPU (or whatever device you use)
        #       apply the model to labels
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward() # loss is stored as an attribute of model's output
        #       optimizer.step()

        print(f"Saving to {args.output_dir}.")
        self.model.save_pretrained(args.output_dir)


def main():
    TRAIN_FILE = "./train.txt"
    VAL_FILE = "./val.txt"
    batch_size = 32

    # Train Tokenizer
    tokenizer = build_tokenizer("train.txt", max_number_of_merges=10000)
    tokenizer.save("./new_tokenizer_model.pkl")

    tokenizer = A1Tokenizer.from_file("./tokenizer_model.pkl")
    print(tokenizer.vocabular)
    """
    dataset = load_dataset("text", data_files={"train": TRAIN_FILE, "test": VAL_FILE})
    # reduce data
    dataset = dataset["test"].train_test_split(test_size=0.01, train_size=0.01)

    dataset = dataset.filter(lambda x: x["text"].strip() != "")
    dataset = dataset.map(
        lambda batch: dict(
            tokenizer(
                batch["text"], truncation=False, padding=True, return_tensors="pt"
            ),
        ),
        batched=True,
        remove_columns=["text"],
    )

    dl_train = DataLoader(
        dataset["train"],
    )
    dl_val = DataLoader(dataset["test"], batch_size=batch_size)
    """

if __name__ == "__main__":
    main()
