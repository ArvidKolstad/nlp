import torch
from ax_modules import A1Tokenizer
from train_model import A2RotaryEmbedding, A2Transformer


class GenerateText:
    def __init__(self, model, tokenizer, temp, k):
        self.model = model
        self.tokenizer = tokenizer
        self.temp = temp
        self.k = k

    def choose_token(self, logits):
        values, idxes = torch.topk(logits, self.k, dim=-1)
        probs = torch.softmax(values / self.temp, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        next_token = torch.gather(idxes, -1, idx)
        return next_token

    def get_next_token(self, promt: list[str]):
        tokenized_text = self.tokenizer(promt, return_tensors="pt")["input_ids"]
        tokenized_text = tokenized_text[:, :-1]
        output = self.model(tokenized_text)
        logits = output.logits
        last_token = logits[..., -1, :]
        print(last_token.shape)
        new_token = self.choose_token(last_token)

        generated_text_encoded = torch.concat((tokenized_text, new_token), dim=-1)
        generated_text = self.tokenizer.decode(generated_text_encoded[0])
        return [generated_text]

    def generate_sentence(self, prompt, max_length: int):
        current_prompt = prompt.copy()
        for i in range(max_length):
            print(current_prompt)
            current_prompt = self.get_next_token(current_prompt)


def main():
    tokenizer = A1Tokenizer.from_file("tokenizer.pkl")

    model = A2Transformer.from_pretrained("./models/new_model/")
    model.rotary_emb = A2RotaryEmbedding(model.config)
    temp = 0.1
    k = 50
    model = model.to(torch.device("cpu"))
    prompt = ["Today is a great "]
    # prompt = ["In natural language processing, a Transformer"]
    # prompt = ["Is Stockholm the capital of Sweden? Answer yes or no. The answer is "]
    # prompt = ["Write a Python program that reverses a list."]

    generator = GenerateText(model, tokenizer, temp, k)
    new_text = generator.get_next_token(prompt)
    print(new_text)
    generator.generate_sentence(prompt, 10)


if __name__ == "__main__":
    main()
