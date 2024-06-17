import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


class MultimodalFullyAlignedDataset(Dataset): # inherit from torch Dataset
    def __init__(self, modalities:list, data:list):
        self.data = data
        self.modalities = modalities
        self.full_data_dict={}
        for i, modality in enumerate(modalities):
            self.full_data_dict[modality] = data[i]

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        data_dict = {}
        for modality in self.modalities:
            data_dict[modality] = self.full_data_dict[modality][idx,:]
        return data_dict

class FullAlignedTabChemDataset(Dataset): # inherit from torch Dataset
    def __init__(self, tabular_data, smiles_data, target_data, modalities:list):
        assert len(tabular_data) == len(smiles_data), "Tabular data and SMILES data must have the same length."
        self.tabular_data = tabular_data
        self.smiles_data = smiles_data
        self.target_data = target_data
        self.modalities = modalities
        
    def __len__(self):
        return len(self.tabular_data)
    
    def __getitem__(self, idx):
        data_dict = {}
        data_dict[self.modalities[0]] = self.tabular_data[idx,:]
        data_dict[self.modalities[1]] = self.smiles_data[idx]
        target_data = self.target_data[idx]
        
        return data_dict, target_data
    
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
    

