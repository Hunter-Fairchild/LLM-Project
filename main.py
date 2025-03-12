import torch
import tiktoken
import os
import platform
import Architecture.GPTModel as Model
import Architecture.Embedding as Embedding
import Architecture.Training as Training
import Training.Dataloaders as Dataloaders
import Training.PartitionTraining as PartitionTraining
from Architecture.Loss_Functions import plot_losses
from Chatbot import chatbot

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

training = PartitionTraining.PartitionTraining(GPT_CONFIG_124M)
training.partition_train(1)
training.save("Testing.pth")


# # Loading Code
# device = "cpu"
# checkpoint = torch.load("model_and_optimizer_small.pth", map_location="cpu")
# model = Model.GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train()


# Basic Chatbot using LLM above
