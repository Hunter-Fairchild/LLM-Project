import datetime
import json

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
        
        self.file_name = f"Logs_{self.date}"
        
    def write_logs(self, stats):
        self.setup()
        
        with open(f"Training\TrainingLogs\{self.file_name}.json", "a") as file:
            json.dump({
                "time": self.time, 
                "stats": stats
            }, file)
            # file.write(f"{self.time}: {stats} \n")
        
# Logging().write_logs("Hello")
        