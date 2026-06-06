import torch
import torch.nn.functional as F
import torch.nn as nn
import weakref
import os
from utils.myddp import is_main_process, barrier

def generate_couple_mask(idx_range, couple_hidden_size, couple_num_tokens):
    mask = torch.ones((couple_num_tokens, couple_num_tokens), dtype=torch.bool)
    assert idx_range[-1] == couple_num_tokens * couple_hidden_size, "idx_range does not match couple_num_tokens and couple_hidden_size"
    tmp_idx_range = []
    for i in range(len(idx_range)):
        assert idx_range[i] % couple_hidden_size == 0, "idx_range must be divisible by couple_hidden_size"
        tmp_idx_range.append(idx_range[i] // couple_hidden_size)
    for i in range(0, len(idx_range) - 1, 2):
        mask[tmp_idx_range[i]:tmp_idx_range[i+1], tmp_idx_range[i+1]:tmp_idx_range[i+2]] = False
        mask[tmp_idx_range[i+1]:tmp_idx_range[i+2], tmp_idx_range[i]:tmp_idx_range[i+1]] = False
    return mask

class MetanetworkTransformer(nn.Module):
    def __init__(self, cfg, idx_range):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.num_mem_token = cfg.num_mem_token
        self.hidden_size = cfg.hidden_size
        self.mean_pool_size = cfg.metanetwork.transformer_cfg.mean_pool_size
        self.idx_range = idx_range
        self.layer_transformer_first = bool(cfg.metanetwork.transformer_cfg.layer_transformer_first)

        self.layer_pe = nn.Parameter(torch.zeros((self.num_layers, self.hidden_size)), requires_grad=True)
        self.token_pe = nn.Parameter(torch.zeros((self.num_mem_token, self.hidden_size)), requires_grad=True)

        transformer_cfg = cfg.metanetwork.transformer_cfg
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer (**transformer_cfg.encoder_cfg) for _ in range(transformer_cfg.num_layers)])
        
        self.couple_layers = nn.ModuleList([nn.TransformerEncoderLayer (**transformer_cfg.couple_encoder_cfg) for _ in range(transformer_cfg.couple_num_layers)])
        self.couple_hidden_size = cfg.metanetwork.transformer_cfg.couple_encoder_cfg.d_model
        self.couple_num_layers = cfg.metanetwork.transformer_cfg.couple_num_layers
        assert self.hidden_size % self.couple_hidden_size == 0, "hidden_size must be divisible by couple_hidden_size"
        self.couple_num_tokens = self.num_mem_token * self.hidden_size // self.couple_hidden_size
        couple_mask = generate_couple_mask(idx_range, self.couple_hidden_size, self.couple_num_tokens)
        self.register_buffer("couple_mask", couple_mask, persistent=False)
        
        # self.scale = nn.Parameter(torch.ones((1, self.num_layers, self.num_mem_token, 1)), requires_grad=True)
        

    def forward(self, memory_states:torch.Tensor) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        memory_states = memory_states + self.layer_pe.unsqueeze(-2) + self.token_pe # apply PE
        batch_size = memory_states.shape[0]
        for i in range(len(self.transformer_layers)):
            if (i % 2 == 0) == self.layer_transformer_first:
                memory_states = self.transformer_layers[i](memory_states.transpose(1, 2).flatten(0, 1)).unflatten(0, (batch_size, self.num_mem_token)).transpose(1, 2) # exchange information among layers
            else:
                memory_states = self.transformer_layers[i](memory_states.flatten(0, 1)).unflatten(0, (batch_size, self.num_layers)) # exchange information among tokens
        memory_states = torch.mean(memory_states.unflatten(2, (self.mean_pool_size, self.num_mem_token // self.mean_pool_size)), dim=2)  # mean pool, not used.
        
        memory_states = memory_states.view(batch_size * self.num_layers, -1, self.couple_hidden_size)
        for i in range(len(self.couple_layers)):
            memory_states = self.couple_layers[i](memory_states, src_mask = self.couple_mask)
        memory_states = memory_states.view(batch_size, -1)
        return memory_states
        

class MetanetworkLinear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.num_mem_token = cfg.num_mem_token
        self.hidden_size = cfg.hidden_size
        
        self.layer_pe = nn.Parameter(torch.zeros((self.num_layers, self.hidden_size)), requires_grad=True)
        self.token_pe = nn.Parameter(torch.zeros((self.num_mem_token, self.hidden_size)), requires_grad=True)
        
        linear_cfg = cfg.metanetwork.linear_cfg
        self.dim_list = [self.hidden_size] + [linear_cfg.linear_hidden_dim] * (linear_cfg.num_layers - 1) + [self.hidden_size]
        self.linear_layers = nn.ModuleList([nn.Linear(self.dim_list[i], self.dim_list[i+1], bias=linear_cfg.bias) for i in range(linear_cfg.num_layers)])
        

    def forward(self, memory_states:torch.Tensor) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        memory_states = memory_states + self.layer_pe.unsqueeze(-2) + self.token_pe # apply PE
        batch_size = memory_states.shape[0]
        memory_states = memory_states.flatten(0, 1)
        for i in range(len(self.linear_layers)):
            memory_states = F.gelu(self.linear_layers[i](memory_states))
        return memory_states.unflatten(0, (batch_size, self.num_layers)).flatten(1, -1)

class MetanetworkLinearGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.num_mem_token = cfg.num_mem_token
        self.hidden_size = cfg.hidden_size
        
        self.layer_pe = nn.Parameter(torch.zeros((self.num_layers, self.hidden_size)), requires_grad=True)
        self.token_pe = nn.Parameter(torch.zeros((self.num_mem_token, self.hidden_size)), requires_grad=True)
        
        linear_cfg = cfg.metanetwork.linear_cfg
        self.dim_list = [self.hidden_size] + [linear_cfg.linear_hidden_dim] * (linear_cfg.num_layers - 1) + [self.hidden_size * 2]
        self.linear_layers = nn.ModuleList([nn.Linear(self.dim_list[i], self.dim_list[i+1], bias=linear_cfg.bias) for i in range(linear_cfg.num_layers)])
        

    def forward(self, memory_states:torch.Tensor) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        memory_states = memory_states + self.layer_pe.unsqueeze(-2) + self.token_pe # apply PE
        batch_size = memory_states.shape[0]
        memory_states = memory_states.flatten(0, 1)
        for i in range(len(self.linear_layers)):
            memory_states = F.gelu(self.linear_layers[i](memory_states))
        gate, value = memory_states.chunk(2, dim=-1)
        memory_states = torch.sigmoid(gate) * value
        return memory_states.unflatten(0, (batch_size, self.num_layers)).flatten(1, -1)

class Metanetwork(nn.Module):
    def __init__(self, metamodel:nn.Module, cfg, output_dim: int):
        super().__init__()
        self.lora_r = cfg.model.lora_r
        self.output_dim = output_dim
        self.metamodel = metamodel
        self.idx_range, end = self.metamodel.divide_idx(self.lora_r, 0)
        self.idx_range.append(end)
        self.adapter_reg = cfg.optim.adapter_reg if hasattr(cfg, 'optim') else 0.0
        self.method = cfg.metanetwork.method
        self.metamodel.set_generate_func(self.method)
        
        if cfg.metanetwork.type == "transformer":
            self.metanetwork = MetanetworkTransformer(cfg, self.idx_range)
            self.scale = cfg.metanetwork.transformer_cfg.scale
        elif cfg.metanetwork.type == "linear":
            self.metanetwork = MetanetworkLinear(cfg)
            self.scale = cfg.metanetwork.linear_cfg.scale
        elif cfg.metanetwork.type == "lineargate":
            self.metanetwork = MetanetworkLinearGate(cfg)
            self.scale = cfg.metanetwork.linear_gate_cfg.scale
        else:
            raise ValueError(f"Unknown metanetwork type: {cfg.metanetwork.type}")
        
    @property
    def config(self):
        # Prefer live inner config if present; else fall back to cached copy
        return getattr(self.metamodel, "config", None)

    @torch.compile # (mode="max-autotune")
    def forward(self, input_ids, input_attention_mask, evidence_ids, evidence_attention_mask, metalora = None, labels = None, use_metanet = True, use_gradient_checkpoint = False, **kwargs) -> dict:
        '''
        memory_states: (batch_size, num_layer, num_mem_token, hidden_size)
        '''
        if use_metanet:
            assert metalora is not None, "metalora cannot be None when use_metanet is True"
            loradict, plain_output = self.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora, use_gradient_checkpoint=use_gradient_checkpoint, return_plain=True)
            outputs = self.metamodel(input_ids=input_ids, attention_mask=input_attention_mask, loradict=loradict, labels=labels, ignore_mem_token=True, use_gradient_checkpoint=use_gradient_checkpoint, **kwargs)
            outputs.reg_loss = self.adapter_reg * torch.abs(plain_output).sum()
        else:
            outputs = self.metamodel(input_ids=input_ids, attention_mask=input_attention_mask, labels=labels, ignore_mem_token=True, use_gradient_checkpoint=use_gradient_checkpoint, **kwargs)
        return outputs
    
    def generate_lora_dict(self, evidence_ids, evidence_attention_mask, metalora, use_gradient_checkpoint = False, return_plain = False) -> dict:
        outputs = self.metamodel(input_ids=evidence_ids, attention_mask=evidence_attention_mask, loradict=metalora, use_gradient_checkpoint=use_gradient_checkpoint)
        memory_states = outputs.memory_states
        plain_output = self.metanetwork(memory_states)  # (batch_size, output_dim)
        loradict = self.metamodel.generate_lora_dict(self.lora_r, scale=self.scale, plain_tensor=plain_output)
        return loradict if not return_plain else (loradict, plain_output)
    
    
    
