import os
from typing import Union
from torch.utils.data import DataLoader
import Architecture.Embedding as Embedding


def make_data_loaders(config: dict[Union[int, float]], directory: str, train_ratio: float) -> tuple[DataLoader]:
    text_data = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            print(file_path)

            with open(file_path, "r", encoding="utf-8") as file:
                text_data += file.read() + "<|endoftext|>"

    # create training/validation set
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = Embedding.create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = Embedding.create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    return train_loader, val_loader


