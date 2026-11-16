import torch
from ax_modules import A1Tokenizer
from train_model import A2RotaryEmbedding, A2Transformer


def choose_token(logits):
    raise NotImplementedError()


def prompt_model(model: A2Transformer, tokenizer: A1Tokenizer, promt: list[str]):
    tokenized_text = tokenizer(promt, return_tensors="pt")

    print(tokenized_text["input_ids"])

    output = model(tokenized_text["input_ids"])
    logits = output.logits()
    tokenizer.decode(logits)
    print(output)


def main():
    tokenizer = A1Tokenizer.from_file("tokenizer.pkl")

    model = A2Transformer.from_pretrained("./models/new_model/")
    model.rotary_emb = A2RotaryEmbedding(model.config)
    model = model.to(torch.device("cpu"))
    promt = ["My name is Arvid"]
    prompt_model(model, tokenizer, promt)


if __name__ == "__main__":
    main()
