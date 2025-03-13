import datetime
import json
import os

# import torch
# import tiktoken
# import os
# import platform
# import Architecture.GPTModel as Model
# import Architecture.Embedding as Embedding
# import Architecture.Training as Training
# import Training.Dataloaders as Dataloaders
# import Training.PartitionTraining as PartitionTraining
# from Architecture.Loss_Functions import plot_losses
# from Chatbot import chatbot

class Logging:
    def setup(self):
        train_time = datetime.datetime.now()
        train_time = train_time.replace(second=0, microsecond=0)
        self.date = train_time.strftime("%Y-%m-%d")
        self.time = train_time.strftime("%H:%M")
        
        self.file_name = f"Logs_{self.date}.json"
        
    def write_logs(self, stats):
        self.setup()
        path = f"Training\TrainingLogs\{self.file_name}"
        previous_logs = []
        if self.file_name in os.listdir("Training/TrainingLogs"):
            with open(path, "r") as file:
                previous_logs += json.load(file)
        
        with open(path, "w") as file:
            json.dump(previous_logs + [{
                "time": self.time, 
                "stats": stats
            }], file)

# Logging().write_logs("Hello")
        