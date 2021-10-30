import torch
from torch import nn
import numpy as np
import math


def get_model(input_size, config):
    if config.model_module == "BASE":
        return Model(input_size, config)
    elif config.model_module == "RES":
        return Model_Res(input_size, config)


class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ScaleLayer(nn.Module):
    def __init__(self, config):
        super(ScaleLayer, self).__init__()
        pressure_unique = np.load(config.pressure_unique_path)
        self.min = np.min(pressure_unique)
        self.max = np.max(pressure_unique)
        self.step = pressure_unique[1] - pressure_unique[0]
        self.my_round_func = my_round_func()

    def forward(self, inputs):
        steps = inputs.add(-self.min).divide(self.step)
        int_steps = self.my_round_func.apply(steps)
        rescaled_steps = int_steps.multiply(self.step).add(self.min)
        clipped = torch.clamp(rescaled_steps, self.min, self.max)
        return clipped


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        act = nn.SELU(inplace=False)
        hidden = config.hidden
        self.do_reg = config.do_reg
        
        self.seq_emb = None
        self.pos_encoder = None
        self.transformer_encoder = None
        if config.use_transformer:
            self.seq_emb = nn.Sequential(
                nn.Linear(input_size, config.d_model),
                nn.LayerNorm(config.d_model),
                act,
                nn.Dropout(config.do_transformer),
            )
            self.pos_encoder = PositionalEncoding(d_model=config.d_model, dropout=config.trf_do)
            encoder_layers = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_head,
                                                        dim_feedforward=config.dim_forward,
                                                        dropout=config.do_transformer, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config.num_layers)        
        
        net_fnc = nn.LSTM if config.rnn_model=="LSTM" else nn.GRU
        self.rnns = None
        self.rnn_dos = None
        self.rnn_bns = None
        if config.hidden is not None:
            input_size = input_size if not config.use_transformer else config.d_model
            self.rnns = nn.ModuleList([
                net_fnc(2 * hidden[i-1], hidden[i], batch_first=True, bidirectional=True)
                if i > 0 else net_fnc(input_size, hidden[0], batch_first=True, bidirectional=True)
                for i in range(len(config.hidden))
            ])
            if config.rnn_do > 0:
                self.rnn_dos = nn.ModuleList([nn.Dropout(config.rnn_do) for _ in range(len(config.hidden)-1)])
            self.rnn_bns = nn.ModuleList([nn.BatchNorm1d(80) for _ in range(len(config.hidden))])
        
        self.use_dp = len(config.gpu) > 1
        
        if self.use_ch:
            self.head = nn.Sequential(
                nn.Linear(hidden[-1] * 2, config.fc), nn.BatchNorm1d(80), nn.Dropout(config.ch_do), act,
                nn.Linear(config.fc, config.fc//2), nn.BatchNorm1d(80), nn.Dropout(config.ch_do), act,
            )
        else:
            self.head = nn.Sequential(nn.Linear(2 * hidden[-1], config.fc), act)
            
        if self.do_reg:
            self.fc2 = nn.Linear(config.fc, 1)
            self.scaler = ScaleLayer(config)
        else:
            self.fc2 = nn.Linear(config.fc, 950)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name or "gru" in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
            elif 'fc' in name or "head" in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        if self.seq_emb is not None:
            x = self.seq_emb(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            
        for i in range(len(self.rnns)):
            if self.use_dp:
                self.rnns[i].flatten_parameters()
            x, _ = self.rnns[i](x)
            if self.rnn_bns is not None:
                x = self.rnn_bns[i](x)
            if self.rnn_dos is not None and i < (len(self.rnns)-1):
                x = self.rnn_dos[i](x)
                        
        x = self.head(x)
        x = self.fc2(x)
        if self.do_reg:
            x = self.scaler(x)
        return x

class Model_Res(nn.Module):
    """
    Using Residual Block Technique
    """
    def __init__(self, input_size, config):
        super().__init__()
        act = nn.SELU(inplace=False)
        
        hidden = config.hidden
        hidden_gru = config.hidden_gru
        
        self.do_reg = config.do_reg        
        self.lstms = nn.ModuleList([
            nn.LSTM(2 * hidden[i-1], hidden[i], batch_first=True, bidirectional=True)
            if i > 0 else nn.LSTM(input_size, hidden[0], batch_first=True, bidirectional=True)
            for i in range(len(hidden))
        ])
        
        self.grus = nn.ModuleList([
            nn.LSTM(2 * hidden_gru[i-1], hidden_gru[i], batch_first=True, bidirectional=True)
            if i > 0 else nn.GRU(hidden[1] * 2, hidden_gru[0], batch_first=True, bidirectional=True)
            for i in range(len(hidden_gru))
        ])

        self.use_dp = len(config.gpu) > 1
        
        self.bns = nn.ModuleList([nn.BatchNorm1d(80) for i in range(len(config.hidden))])
        self.bns_gru = nn.ModuleList([nn.BatchNorm1d(80) for i in range(len(config.hidden_gru) - 1)])

        # add batch normalization
        self.fc1 = nn.Linear(2 * hidden[-1] + sum(hidden_gru)*2, config.fc)
        self.act = act
        if self.do_reg:
            self.fc2 = nn.Linear(config.fc, 1)
            self.scaler = ScaleLayer(config)
        else:
            self.fc2 = nn.Linear(config.fc, 950)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name or "gru" in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        xs = []
        zs = []
        z1s = []
        for i in range(len(self.lstms)):
            if self.use_dp:
                self.lstms[i].flatten_parameters()
            x, _ = self.lstms[i](x)
            x = self.bns[i](x)
            xs.append(x)
        for i in range(len(self.grus)):
            if self.use_dp:
                self.grus[i].flatten_parameters()
            z, _ = self.grus[i](xs[i+1]) if i == 0 else self.grus[i](z1s[i-1])
            zs.append(z)
            if i < (len(self.grus) - 1):
                z1 = z * xs[i+2]
                z1 = self.bns_gru[i](z1)
                z1s.append(z1)
        x = torch.cat([xs[-1]] + zs, dim=-1)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.do_reg:
            x = self.scaler(x)
        return x

