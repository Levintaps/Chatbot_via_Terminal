from transformers import GPT2LMHeadModel, GPT2Tokenizer

def initialize_model_and_tokenizer():
    model_name = "gpt2"  # You can use other GPT-2 variants like gpt2-medium, gpt2-large, etc.
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a response using the model
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    model, tokenizer = initialize_model_and_tokenizer()

    print("ChatGPT: Hello! How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("ChatGPT: Goodbye! Have a great day.")
            break

        response = generate_response(user_input, model, tokenizer)
        print("ChatGPT:", response)

if __name__ == "__main__":
    main()
