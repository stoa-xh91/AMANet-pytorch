import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):
        """
        Args:
            q: Queries [B, L, D]
            k: Keys [B, L, D]
            v: Values [B, L, D]
            scale: scale factor
        Returns:
            context: [B,L,D]
            attetention: [B,L,L]
        """
        # self attention
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        #
        context = torch.bmm(attention, v)
        return [context, attention]

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=32, num_heads=4, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x_k, x_v, x_q):
        """
        :param x_k: [B, L, D]
        :param x_v: [B, L, D]
        :param x_q: [B, L, D]
        :return:
        """
        residual = x_v
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size, num_locs, feat_dims = x_k.size()
        xk_reshaped = x_k.reshape(-1, feat_dims)
        xv_reshaped = x_v.reshape(-1, feat_dims)
        xq_reshaped = x_q.reshape(-1, feat_dims)

        # linear projection
        key = F.leaky_relu(self.linear_k(xk_reshaped),0.02)
        value = F.leaky_relu(self.linear_v(xv_reshaped),0.02)
        query = F.leaky_relu(self.linear_q(xq_reshaped),0.02)

        # split by heads
        key = key.reshape(batch_size, num_locs, num_heads, dim_per_head).transpose(1, 2)
        key = key.reshape(batch_size*num_heads, num_locs, dim_per_head)
        value = value.reshape(batch_size, num_locs, num_heads, dim_per_head).transpose(1, 2)
        value = value.reshape(batch_size * num_heads, num_locs, dim_per_head)
        query = query.reshape(batch_size, num_locs, num_heads, dim_per_head).transpose(1, 2)
        query = query.reshape(batch_size * num_heads, num_locs, dim_per_head)

        # scaled dot product attention
        scale = dim_per_head ** -0.5
        context_outputs = self.dot_product_attention(query, key, value, scale)
        context, attention = context_outputs
        # concat heads
        context = context.reshape(batch_size, num_heads, num_locs, dim_per_head).transpose(1,2)
        context = context.reshape(batch_size * num_locs, num_heads*dim_per_head)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        # output = self.dropout(output)
        output = output.reshape(batch_size, num_locs, -1)
        # add residual and norm layer
        # print(output.size())
        output = F.leaky_relu(output, 0.02)
        output += residual
        return output


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=32, ffn_dim=32, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        :param x: [B, L, D]
        :return:
        """
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):

    def __init__(self, model_dim=32, num_heads=4, ffn_dim=32, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs):

        # self attention
        context_outputs = self.attention(inputs, inputs, inputs)
        context, attention = context_outputs
        # feed forward network
        output = self.feed_forward(context)

        return [output, attention]

class DecoderLayer(nn.Module):

    def __init__(self, model_dim=32, num_heads=4, ffn_dim=32, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
              dec_inputs,
              dir_enc_outputs,
              orth_dir_enc_outputs,
              self_attn_mask=None,
              context_attn_mask=None):
        """
        :param dec_inputs: BxL1xD
        :param dir_enc_outputs: BxL1xD
        :param orth_dir_enc_outputs: BxL2xD
        :return: dec_output, self_attention, context_attention, context_interaction
        """
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs)
        # context attention
        # query is decoder's outputs, key and value are direction encoder's inputs
        context_output, context_attention = self.attention(dir_enc_outputs, dec_output, dec_output)
        # h & v interaction
        # query is orthogonal direction encoder's outputs, key is direction encoder's input and value is context outputs
        dec_output, context_interaction = self.attention(dir_enc_outputs, context_output, orth_dir_enc_outputs)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output