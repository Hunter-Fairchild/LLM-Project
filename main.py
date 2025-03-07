import torch
import tiktoken
import os
import platform
import Architecture.GPTModel as Model
import Architecture.Embedding as Embedding
import Architecture.Training as Training
from Architecture.Loss_Functions import plot_losses
from Chatbot import chatbot

print("hi 2")


# model parameters
GPT_CONFIG_124M = {
"vocab_size": 50257,
"context_length": 256,
"emb_dim": 768,
"n_heads": 12,
"n_layers": 12,
"drop_rate": 0.1,
"qkv_bias": False
}


# build model
torch.manual_seed(123)
model = Model.GPTModel(GPT_CONFIG_124M)
model.eval()

# build tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# load directory of data
os_version = platform.system()
directory = 'Datasets/Small Dataset'

text_data = ""
for filename in os.listdir(directory):               # iterate over files in that directory
    file_path = os.path.join(directory, filename)    # create filepath

    if os.path.isfile(file_path):                    # checking if it is a file
        print(file_path)

        with open(file_path, "r", encoding="utf-8") as file:    # open file
            text_data += file.read() + "<|endoftext|>"          # read the file

# create training/validation set
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# create dataloaders for each set
torch.manual_seed(123)
train_loader = Embedding.create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = Embedding.create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


# # Loading Code
# device = "cpu"
# checkpoint = torch.load("model_and_optimizer_small.pth", map_location="cpu")
# model = Model.GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train()

# send model to compute device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("device =",device)

# construct optimizer object
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)

# start training (optimization) method
num_epochs = 10
train_losses, val_losses, tokens_seen = Training.train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="The cat sprinted down", tokenizer=tokenizer
)

# produce plot
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


# Basic Chatbot using LLM above
chatbot(model, tokenizer, device)
