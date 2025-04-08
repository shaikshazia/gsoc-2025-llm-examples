from transformers import pipeline

# Create a text generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Generate text
prompt = "Hello, I'm a language model"
generated = generator(prompt, max_length=100, num_return_sequences=1)

# Print the result in terminal (optional)
print(generated)

# Save to outputs folder
output_text = generated[0]['generated_text']

with open("outputs/generated_output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("\n✅ Output saved to outputs/generated_output.txt")
