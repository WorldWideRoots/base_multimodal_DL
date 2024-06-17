import torch
import torch.nn as nn

class MultimodalWrapper(nn.Module):
    def __init__(self, modalities:list, unimodal_encoders:list, unimodal_in_dims:list, unimodal_out_dims:list, desired_unimodal_out_dims:list, 
                  fusion_module:nn.Module, final_proj_head:nn.Module=None, build_unimodal_proj:bool=True, unimodal_proj_norm_type:str='ln', device:str='cpu'):
            super().__init__()
            self.device = device
            self.modalities = modalities
            self.unimodal_in_dims=unimodal_in_dims
            self.unimodal_out_dims=unimodal_out_dims
            self.desired_unimodal_out_dims = desired_unimodal_out_dims
            self.fusion_module = fusion_module.to(device)
            self.final_proj_head = final_proj_head.to(device) if final_proj_head is not None else None
            self.build_unimodal_proj = build_unimodal_proj
            self.unimodal_proj_norm_type = unimodal_proj_norm_type

            encoder_dict = {}
            for i, modality in enumerate(modalities):
                encoder_dict[modality] = unimodal_encoders[i]
            self.unimodal_encoders = nn.ModuleDict(encoder_dict)

            if build_unimodal_proj:
                unimodal_proj_dict = {}
                for i, modality in enumerate(modalities):
                    proj_norm = nn.LayerNorm if unimodal_proj_norm_type == 'ln' else nn.BatchNorm1d
                    unimodal_proj_dict[modality] = nn.Sequential(
                         proj_norm(self.unimodal_out_dims[i]),
                        nn.Linear(self.unimodal_out_dims[i], self.desired_unimodal_out_dims[i])
                    ).to(device)
                self.unimodal_projections = nn.ModuleDict(unimodal_proj_dict)

    def forward(self, data_dict:dict):
        unimodal_outputs = {}
        for modality in self.modalities:
            unimodal_outputs[modality] = self.unimodal_encoders[modality](data_dict[modality].to(self.device))
            if self.build_unimodal_proj:
                unimodal_outputs[modality] = self.unimodal_projections[modality](unimodal_outputs[modality])
        
        fused_output = self.fusion_module(unimodal_outputs)
        if self.final_proj_head is not None:
            fused_output = self.final_proj_head(fused_output)
        return fused_output