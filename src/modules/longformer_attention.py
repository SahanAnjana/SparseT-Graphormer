import torch
from torch import nn
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

class LongformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_window=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_window = attention_window
        self.longformer_attn = LongformerSelfAttention(
            config=None,  # Will be set in forward
            layer_id=0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, attention_mask=None, global_attention_mask=None):
        # Longformer expects batch first: (batch, seq_len, embed_dim)
        # The current code uses (seq_len, batch, embed_dim)
        x = query.permute(1, 0, 2)
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        if global_attention_mask is None:
            global_attention_mask = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        # Create a config on the fly
        from transformers import LongformerConfig
        config = LongformerConfig(
            attention_window=[self.attention_window] * self.num_heads,
            hidden_size=self.embed_dim,
            num_attention_heads=self.num_heads,
        )
        self.longformer_attn.config = config
        attn_output = self.longformer_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )[0]
        attn_output = self.dropout(attn_output)
        # Convert back to (seq_len, batch, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output, None
