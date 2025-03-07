import torch


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """"""
    for _ in range(max_new_tokens):              # loop for number of tokens

        idx_cond = idx[:, -context_size:]        # last (context_size) inputs

        with torch.no_grad():
            logits = model(idx_cond)             # compute scores

        logits = logits[:, -1, :]                # check last score (for next word)

        if top_k is not None:                                        # top-k sampling
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(                                    # apply threshold
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:                                        # temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)    # greedy decoding (no temperature)

        if idx_next == eos_id:                                       # safety (???)
            break

        idx = torch.cat((idx, idx_next), dim=1)   # add the new word to the (growing) list

    return idx                                    # return the list of token id's

def generate_and_print(model, tokenizer, device, start_context):
    """More advanced text generation"""

    model.eval() # set to evalation mode

    context_size = model.pos_emb.weight.shape[0] # length of input

    encoded = text_to_token_ids(start_context, tokenizer).to(device) # encode

    with torch.no_grad():            # generate output
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            top_k=25,
            temperature=1.4
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer) # decode tokens

    print(decoded_text.replace("\n", " ")) # print the output

    model.train() # set to training mode
    
def text_to_token_ids(text, tokenizer):
    """encodes text to vector of token ID's"""
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """decodes vector of token ID's to text"""
    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())
