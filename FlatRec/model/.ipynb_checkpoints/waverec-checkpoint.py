import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention

class WAVERecModel(SequentialRecModel):
    def __init__(self, args):
        super(WAVERecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
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

class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
            print("all_encoder_layers : ", all_encoder_layers)
        return all_encoder_layers

class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha
        self.wavelet_layer = WaveletTransform(args)  # 웨이블릿 변환 클래스 추가

    def forward(self, input_tensor, attention_mask):
        wavelet_output = self.wavelet_layer(input_tensor) 
        gsp = self.attention_layer(input_tensor, attention_mask)        
        hidden_states = self.alpha * wavelet_output + ( 1 - self.alpha ) * gsp

        return hidden_states

class WaveletTransform(nn.Module):
    def __init__(self, args):
        super(WaveletTransform, self).__init__()

        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.pass_weight = args.pass_weight
        self.filter_type = args.filter_type  # 'haar' 또는 'db2' 선택
        self.sqrt_beta = nn.Parameter(torch.randn(1, args.hidden_size, 1))

        # 필터를 선택적으로 생성
        self.lowpass_filter, self.highpass_filter = self._generate_wavelet_filters(self.filter_type, self.pass_weight)
    
    def _generate_wavelet_filters(self, filter_type, pass_weight):
        """
        필터 유형과 pass_weight를 기반으로 lowpass 및 highpass 필터 생성.
        """
        if filter_type == "haar":
            # Haar 필터 계산
            lowpass = torch.tensor([pass_weight, pass_weight], dtype=torch.float32)
            highpass = torch.tensor([pass_weight, -pass_weight], dtype=torch.float32)
        elif filter_type == "db2":
            # db2 필터 계산
            sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=torch.float32))
            norm_factor = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))  # 정규화 인자
            lowpass = torch.tensor(
                [
                    (1 + sqrt3) / 4 * pass_weight,
                    (3 + sqrt3) / 4 * pass_weight,
                    (3 - sqrt3) / 4 * pass_weight,
                    (1 - sqrt3) / 4 * pass_weight,
                ],
                dtype=torch.float32,
            )
            highpass = torch.tensor(
                [
                    (1 - sqrt3) / 4 * pass_weight,
                    -(3 - sqrt3) / 4 * pass_weight,
                    (3 + sqrt3) / 4 * pass_weight,
                    -(1 + sqrt3) / 4 * pass_weight,
                ],
                dtype=torch.float32,
            )
            # db2는 정규화
            lowpass = lowpass / norm_factor
            highpass = highpass / norm_factor
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return lowpass, highpass

    
    def forward(self, input_tensor):
        # [batch, seq_len, hidden] -> [batch, hidden, seq_len]
        input_tensor = input_tensor.transpose(1, 2)
        # 필터를 hidden_size에 맞게 확장하고 input_tensor의 디바이스로 이동
        lowpass_filter = (self.lowpass_filter.view(1, 1, -1).repeat(input_tensor.size(1), 1, 1).to(input_tensor.device))
        highpass_filter = (self.highpass_filter.view(1, 1, -1).repeat(input_tensor.size(1), 1, 1).to(input_tensor.device))
        
        padding = (lowpass_filter.size(2) - 1) // 2
        
        # 1D Convolution 수행 (behavior 단위로 Wavelet 변환 적용)
        with torch.no_grad():
            lowpass = F.conv1d(input_tensor, lowpass_filter, padding=padding, groups=input_tensor.size(1))
            highpass = F.conv1d(input_tensor, highpass_filter, padding=padding, groups=input_tensor.size(1))
        
        highpass = (self.sqrt_beta ** 2) * highpass
        
        # 필터를 반대로 적용하여 원래 시퀀스를 복원합니다.
        with torch.no_grad():
            lowpass_reconstructed = F.conv_transpose1d(lowpass, lowpass_filter, padding=padding, groups=input_tensor.size(1))
            highpass_reconstructed = F.conv_transpose1d(highpass, highpass_filter, padding=padding, groups=input_tensor.size(1))
        
        # 두 복원된 결과를 합쳐서 원래 입력에 가까운 값을 복원
        wavelet_output = lowpass_reconstructed + highpass_reconstructed
        # 두 결과를 원래 차원으로 복원 후 합산
        # wavelet_output = lowpass.transpose(1, 2) + (self.sqrt_beta ** 2) * highpass.transpose(1, 2)
        # print("wavelet_output shape", wavelet_output.shape)
        # Dropout 및 LayerNorm 적용
        hidden_states = self.out_dropout(wavelet_output.transpose(1, 2))
        hidden_states = self.LayerNorm(hidden_states + input_tensor.transpose(1, 2))  # 원래 차원 맞춰서 복원
        return hidden_states

