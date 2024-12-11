import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention

class FlatRecModel(SequentialRecModel):
    def __init__(self, args):
        super(FlatRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = FlatRecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)
        return loss


class FlatRecEncoder(nn.Module):
    def __init__(self, args):
        super(FlatRecEncoder, self).__init__()
        self.args = args
        self.frequency_layers = nn.ModuleList(
            [FrequencyLayer(args) for _ in range(args.num_frequency_layers)]
        )

        self.attention_blocks = nn.ModuleList(
            [FlatRecAttentionBlock(args) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]

        # Apply all FrequencyLayer instances in sequence
        for freq_layer in self.frequency_layers:
            hidden_states = freq_layer(hidden_states)
            all_encoder_layers.append(hidden_states)

        # Pass through all attention blocks
        for block in self.attention_blocks:
            hidden_states = block(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers = [hidden_states]  # Return only the final output

        return all_encoder_layers


class FlatRecAttentionBlock(nn.Module):
    def __init__(self, args):
        super(FlatRecAttentionBlock, self).__init__()
        self.layer = FlatRecAttentionLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class FlatRecAttentionLayer(nn.Module):
    def __init__(self, args):
        super(FlatRecAttentionLayer, self).__init__()
        self.attention_layer = MultiHeadAttention(args)

    def forward(self, input_tensor, attention_mask):
        # Perform attention
        hidden_states = self.attention_layer(input_tensor, attention_mask)
        return hidden_states


class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.sqrt_beta = args.sqrt_beta

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)
        x = x * (weight ** self.sqrt_beta)
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor

        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states