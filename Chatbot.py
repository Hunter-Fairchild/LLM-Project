import torch
import Architecture.Generation as Generation

def chatbot(model, tokenizer, device):
    """Basic Chatbot loop"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Hello!")
    print("How may I help you?")

    while True:
        prompt = input()                      # ask user for input
        if prompt == "goodbye": break         # check for exit flag

        chat(model, tokenizer, device, prompt)

    print("Goodbye!")
    
def chat(model, tokenizer, device, start_context):
    """More advanced text generation"""
    n = len(start_context)

    model.eval() # set to evalation mode

    context_size = model.pos_emb.weight.shape[0] # length of input

    encoded = Generation.text_to_token_ids(start_context, tokenizer).to(device) # encode


    with torch.no_grad():            # generate output
        token_ids = Generation.generate(
            model=model,
            idx=encoded,
            max_new_tokens=30,
            context_size=context_size,
            top_k=25,
            temperature=1.4
        )

    decoded_text = Generation.token_ids_to_text(token_ids, tokenizer) # decode tokens

    print("\t"+decoded_text[n:].replace("\n", " ")) # print the output

    model.train() # set to training mode
