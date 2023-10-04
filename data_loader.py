import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import ast
from embedding import OHAAE

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        self.embedder = OHAAE()
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Assuming the amino acid sequence is in the first column
        amino_acid_sequence = self.dataframe.iloc[idx, 0]
        
        # Embed the amino acid sequence
        embedded_sequence = self.embedder.embbed_sequence(amino_acid_sequence)
        sequence_tensor = torch.Tensor(embedded_sequence)

        
        # Assuming the list of numbers is in the second column
        label = ast.literal_eval(self.dataframe.iloc[idx, 1])  # Use ast.literal_eval to convert string to list
        label = torch.tensor(label, dtype=torch.float32)
        
        return sequence_tensor, label


def collate_fn(batch):
    sequences, labels = zip(*batch)
    return torch.stack(sequences), torch.stack(labels)

def load_data(csv_file, batch_size=32, val_split=0.2):
    dataset = CustomDataset(csv_file)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader
