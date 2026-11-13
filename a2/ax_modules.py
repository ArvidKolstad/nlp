import pickle
from collections import Counter

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

        # TODO: Your work here is to split the texts into words and map them to integer values.
        #
        # - If `truncation` is set to True, the length of the encoded sequences should be
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

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
            ]  ### Fråga om BOS EOS ska räknas med i max L, #måste vara större än 2 för att kunna lägga till BOS och EOS

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

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.

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
        # kan ett okänt token_id finnas med ids? vad händer då?
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

        self.pad_token_id = processing_class.pad_token_id

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

    def collate(self, batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        ids = [
            ex["input_ids"] + [self.pad_token_id] * (max_len - len(ex["input_ids"]))
            for ex in batch
        ]
        mask = [
            ex["attention_mask"] + [0] * (max_len - len(ex["attention_mask"]))
            for ex in batch
        ]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print("Device:", device)
        self.model.to(device)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate,
        )
        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collate,
        )

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

        for epoch in range(args.num_train_epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[labels == self.pad_token_id] = -100

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}, Batch index {batch_idx}, Loss: {loss.item():.4f}"
                    )

            avg_train_loss = (
                total_train_loss / len(train_loader)
                if len(train_loader) > 0
                else float("inf")
            )
            perplexity = np.exp(avg_train_loss)

            print(
                f"Epoch {epoch + 1}/{args.num_train_epochs}, Average Training Loss: {avg_train_loss:.4f}, Perplexity: {perplexity:.4f}"
            )

            # Evaluate the model on the validation set after each epoch
            self.model.eval()
            total_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    labels = input_ids.clone()
                    labels[labels == self.processing_class.pad_token_id] = -100

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    total_loss += outputs.loss.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
            val_perplexity = np.exp(avg_loss)
            print(
                f"Epoch {epoch + 1}/{args.num_train_epochs}, Validation Loss: {avg_loss:.4f}, Perplexity: {val_perplexity:.4f}"
            )

        print(f"Saving to {args.output_dir}.")
        self.model.save_pretrained(args.output_dir)
