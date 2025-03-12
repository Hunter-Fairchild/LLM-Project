import torch
import tiktoken
import os
import platform
from typing import Union
import Architecture.GPTModel as Model
import Architecture.Embedding as Embedding
import Architecture.Training as Training
import Training.Dataloaders as Dataloaders
from Architecture.Loss_Functions import plot_losses
from Chatbot import chatbot


class PartitionTraining:
    def __init__(self, config: dict[Union[int, float]], seed: int = 123):
        # build model
        torch.manual_seed(seed)
        
        self.config = config
        self.model = Model.GPTModel(self.config)
        self.model.eval()

        # build tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # load directory of data
        self.os_version = platform.system()

        self.train_loader, self.val_loader = Dataloaders.make_data_loaders(self.config, 'Datasets/Small Dataset', 0.9)
        
        # send model to compute device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("device =", self.device)

        # construct optimizer object
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0004, weight_decay=0.1
        )
        
    def partition_train(self, num_epochs: int):
        # start training (optimization) method
        train_losses, val_losses, tokens_seen = Training.train_model(
            self.model, self.train_loader, self.val_loader, self.optimizer, self.device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context="The cat sprinted down", tokenizer=self.tokenizer
        )
        # produce plots
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    def save(self, file_name: str):
        # save model weights and optimization info
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            },
            file_name
        )
        
    def load(self):
        checkpoint = torch.load("model_and_optimizer_small.pth", map_location="cpu")
        self.model = Model(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=0.1)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.train()
