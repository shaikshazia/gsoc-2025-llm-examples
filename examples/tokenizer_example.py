# tokenizer_example.py

from transformers import AutoTokenizer

# Load a pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample text input
text = "Hello GSoC! This is a tokenizer test."

# Tokenize the input
tokens = tokenizer(text)

# Print the output tokens
print("Input Text:", text)
print("Tokenized Output:", tokens)
