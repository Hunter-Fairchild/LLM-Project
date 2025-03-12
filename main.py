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
