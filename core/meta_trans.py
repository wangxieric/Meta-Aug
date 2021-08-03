import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimension of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn_output_weights = F.softmax(score, -1)
        attn_output = torch.bmm(attn_output_weights, value)
        return attn_output, attn_output_weights


class MultiAtrAttention(nn.Module):
    """
    Multi-Atr Attntion is based on Multi-Head Attention, which proposed in "Attention Is All You Need"
    For multi-head attention, instead of performing a single attention function with n-dimensional keys, 
    values, and queries, project the queries, keys and values h times with different, learned linear projections 
    to n dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)
    In Multi-Atr Attention, each head focuses on a single attribute-based sentence representation.
    The value of parameter h relies on the number of attributes.
    Args:
        emb_dim (int): The default demension of the token embedding
        num_atr (int): The number of attributes. (default: 5)
    Inputs: sent_emb, atr_scores
        - **sent_emb** (batch, token_len, emb_dim): the sentence embedding, token_len = 1 
        if sentence level embedding (e.g. [cls]) is used.
        - **atr_scores** (batch, num_atr, token_len)
        - **mask** (-): tensor containing indices to be masked
    Returns: output, attn
        - **output** (batch, token_len, dimensions): tensor containing the attended output features.
        - **attn** (batch, num_atr, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, emb_dim, num_atr):
        super(MultiAtrAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_atr = num_atr
        self.scaled_dot_attn = ScaledDotProductAttention(self.emb_dim)
        self.proj = nn.Linear(self.emb_dim * num_head, self.emb_dim)
        
    def forward(
            self,
            sent_feat: Tensor,
            atr_scores: Tensor,
            mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        
        batch_size = sent_feat.size(0)
        sent_feat = sent_feat.unsqueeze(1).repeat(1, self.num_atr, 1, 1)
        sent_feat = torch.einsum('ijkm,ijk->ijkm', sent_feat, atr_scores)
        
        sent_feat = sent_feat.view(batch_size * self.num_atr, -1, self.emb_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_atr, 1, 1)  # B x N x token_LEN x token_LEN

        # key, value and query are using the same resource, sent_feat, by following the self_attention mechanism 
        attn_output, attn_output_weights = self.scaled_dot_attn(sent_feat, sent_feat, sent_feat, mask)

        attn_output = attn_output.view(self.num_atr, batch_size, -1, self.emb_dim)
        attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_atr * self.emb_dim)  # BxTxND
        
        # shrink the dimension of the output feature
        output = self.proj(attn_output)
        
        return output, attn_output_weights
    

class MetaTrans(nn.Module):
    """
        MetaTrans is a novel Transformer, which models the text data from various perspectives according to 
        available metadata or attributes of the given text.
    """
    def __init__(self):
        super(MetaTrans, self).__init__(emb_dim, num_atr)
        self.emb_dim = emb_dim
        self.num_atr = num_atr
        self.multi_atr_attn = MultiAtrAttention(self.emb_dim, self.num_atr)
        
    def forward(self, sent_feat, atr_scores)
        return self.multi_atr_attn(sent_feat, atr_scores)