# TODO: Go through the video. 2:04:00

import torch
from torch import nn
import torch.nn.functional as F


# Adds more params in the model giving it more representational power.
# Adds non-linearity, allowing the model to model complex transformations.
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.embed_dim)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states) # [b, num_patches, embed_dim] -> [b, num_patches, intermediate_size]
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states) # [b, num_patches, intermediate_size] -> [b, num_patches, embed_dim]
        return hidden_states # [b, num_patches, embed_dim]
    

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embed_dim // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, hidden_states):
        batch_size, num_patches, _ = hidden_states.shape

        q = self.q_proj(hidden_states) # [b, num_patches, embed_dim] -> [b, num_patches, embed_dim]
        k = self.k_proj(hidden_states) # [b, num_patches, embed_dim] -> [b, num_patches, embed_dim]
        v = self.v_proj(hidden_states) # [b, num_patches, embed_dim] -> [b, num_patches, embed_dim]
    
        q = q.reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # [b, num_heads, num_patches, head_dim]
        k = k.reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # [b, num_heads, num_patches, head_dim]
        v = v.reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # [b, num_heads, num_patches, head_dim]
        
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale # [b, num_heads, num_patches, num_patches]
        
        attn_probs = F.softmax(attn_scores, dim=-1).to(q.dtype) # [b, num_heads, num_patches, num_patches]
        attn_probs = F.dropout(attn_probs, p=self.attn_dropout, training=self.training) # [b, num_heads, num_patches, num_patches]
        
        attn_outputs = torch.matmul(attn_probs, v) # [b, num_heads, num_patches, head_dim]
        attn_outputs = attn_outputs.transpose(1, 2).contiguous() # [b, num_patches, num_heads, head_dim]
        attn_outputs = attn_outputs.reshape(batch_size, num_patches, -1) # [b, num_patches, embed_dim]
        attn_outputs = self.out_proj(attn_outputs) # [b, num_patches, embed_dim]

        return attn_outputs, attn_probs


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.layer_norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        residual = hidden_states # [b, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states) # [b, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states) # [b, num_patches, embed_dim]
        hidden_states += residual # [b, num_patches, embed_dim]
        residual = hidden_states # [b, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states) # [b, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states) # [b, num_patches, embed_dim]
        hidden_states += residual # [b, num_patches, embed_dim]
        return hidden_states # [b, num_patches, embed_dim]


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config)  for _ in range(config.num_hidden_layers)]
        )

    def forward(self, patch_embeds):
        hidden_states = patch_embeds # [b, num_patches, embed_dim]
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states) # [b, num_patches, embed_dim] -> [b, num_patches, embed_dim]
        return hidden_states # [b, num_patches, embed_dim]
    

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size, # for no overlap between patches
            padding="valid" # no padding
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        assert self.num_patches == 256
        # In vanilla Transformers, positional information is encoded using fixed sinusoidal functions.
        # These values are pre-computed and not learnable. The model learns to interpret these encodings 
        # but can't modify them. In ViT, positional embeddings are learnable parameters allowing the 
        # model to discover and learn whichever patterns it finds most useful during training.
        self.position_embedding = nn.Embedding(num_embeddings=self.num_patches, embedding_dim=config.embed_dim)
        self.register_buffer(
            "patch_pos_ids",
            tensor=torch.arange(self.num_patches).expand((1, -1)), # [1, num_patches-1]
            persistent=False # not saved in checkpoints
        )
        
    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values) # [b, c, h, w] -> [b, embed_dim, h//patch_size, w//patch_size]
        patch_embeds = patch_embeds.flatten(start_dim=2) # [b, embed_dim, h//patch_size * w//patch_size]
        patch_embeds = patch_embeds.permute(0, 2, 1) # [b, num_patches, embed_dim]
        patch_embeds = patch_embeds + self.position_embedding(self.patch_pos_ids) # [b, num_patches, embed_dim] + [1, num_patches, embed_dim]
        return patch_embeds # [b, num_patches, embed_dim]
    

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        patch_embeds  = self.embeddings(pixel_values) # [b, c, h, w] -> [b, num_patches, embed_dim]
        hidden_states = self.encoder(patch_embeds) # [b, num_patches, embed_dim] -> [b, num_patches, embed_dim]
        hidden_states = self.post_layernorm(hidden_states) # [b, num_patches, embed_dim]
        return hidden_states # [b, num_patches, embed_dim]
    

class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        return self.vision_model(pixel_values) # [b, c, h, w] -> [b, num_patches, embed_dim]
