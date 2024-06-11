import torch
import torch.nn as nn 
import torch.nn.functional as F

class ChemWrapper(nn.Module):
    def __init__(self, encoder_out_dim:int, final_out_dim:int, device:str='cpu', norm_type='ln'):
        super().__init__() 
        self.device = device
        self.norm_type = norm_type
        self.encoder_out_dim = encoder_out_dim  
        self.final_out_dim = final_out_dim
        self.norm = nn.LayerNorm if norm_type == 'ln' else nn.BatchNorm1d
        self.projector = nn.Sequential(
            self.norm(encoder_out_dim),
            nn.Linear(encoder_out_dim, final_out_dim)
        )

class ChembertaWrapper(ChemWrapper):
    def __init__(self, tokenizer:nn.Module, chem_encoder:nn.Module, padding:bool=True, truncation:bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.chem_encoder = chem_encoder
        self.padding = padding
        self.truncation = truncation

    def forward(self, inputs):
        inputs = self.tokenizer(inputs, padding=self.padding, truncation=self.truncation, return_tensors='pt').to(self.device)
        outputs = self.chem_encoder(**inputs)
        return self.projector(outputs.last_hidden_state[:, 0, :])                                            


