from torch.utils.data import random_split


def split_dataset(dataset, keep=0.2):
    train_size = int(keep * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    return train_dataset, validation_dataset
