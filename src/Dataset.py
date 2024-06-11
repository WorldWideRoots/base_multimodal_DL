import torch
import torch.nn as nn
import Dataset

import numpy as np
import pandas as pd


class FullAlignedMultimodalDataset(Dataset): # inherit from torch Dataset
    def __init__(self, tabular_data, smiles_data, target_data):
        assert len(tabular_data) == len(smiles_data), "Tabular data and SMILES data must have the same length."
        self.tabular_data = tabular_data
        self.smiles_data = smiles_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.tabular_data)
    
    def __getitem__(self, idx):
        tabular_data = self.tabular_data[idx]
        smiles_data = self.smiles_data[idx]
        target_data = self.target_data[idx]
        
        return tabular_data, smiles_data, target_data
    
class TabularDataset(Dataset):
    def __init__(self, tabular_data, target_data):
        self.tabular_data = tabular_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.tabular_data)
    
    def __getitem__(self, idx):
        tabular_data = self.tabular_data[idx]
        target_data = self.target_data[idx]
        
        return tabular_data, target_data
    
class SMILESDataset(Dataset):
    def __init__(self, smiles_data, target_data):
        self.smiles_data = smiles_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.smiles_data)
    
    def __getitem__(self, idx):
        smiles_data = self.smiles_data[idx]
        target_data = self.target_data[idx]
        
        return smiles_data, target_data
    

