import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "今天天气不错啊"

tokens = tokenizer.encode(text, allowed_special={'<endoftext>'})

print(tokens)

out_str = tokenizer.decode(tokens)

print(out_str)

