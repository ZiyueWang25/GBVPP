import torch
from torch import nn
import numpy as np
import math


def get_model(input_size, config):
    if config.model_module == "BASE":
        return Model(input_size, config)
    elif config.model_module == "CH":
        return Model_CH(input_size, config)
    elif config.model_module == "PulpFiction":
        return Model_PulpFiction(input_size, config)
    elif config.model_module == "transformer":
        return Model_transformer(input_size, config)



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


class Model(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        act = nn.SELU(inplace=False)
        hidden = config.hidden
        use_bi = config.bidirectional
        self.do_reg = config.do_reg
        self.lstms = nn.ModuleList([
            nn.LSTM((1+use_bi) * hidden[i-1], hidden[i], batch_first=True, bidirectional=use_bi)
            if i > 0 else nn.LSTM(input_size, hidden[0], batch_first=True, bidirectional=use_bi)
            for i in range(len(config.hidden))
        ])
        self.use_dp = len(config.gpu) > 1
        self.use_bn_after_lstm = config.use_bn_after_lstm
        if self.use_bn_after_lstm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(80) for i in range(len(config.hidden))])

        # add batch normalization
        self.fc1 = nn.Linear(2 * hidden[-1], config.fc)
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
            if 'lstm' in name:
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
        for i in range(len(self.lstms)):
            if self.use_dp:
                self.lstms[i].flatten_parameters()
            x, _ = self.lstms[i](x)
            if self.use_bn_after_lstm:
                x = self.bns[i](x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.do_reg:
            x = self.scaler(x)
        return x


class Model_CH(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        act = nn.SELU(inplace=False)
        hidden = config.hidden
        use_bi = config.bidirectional
        self.do_reg = config.do_reg
        self.lstms = nn.ModuleList([
            nn.LSTM((1+use_bi) * hidden[i-1], hidden[i], batch_first=True, bidirectional=use_bi)
            if i > 0 else nn.LSTM(input_size, hidden[0], batch_first=True, bidirectional=use_bi)
            for i in range(len(config.hidden))
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden[-1] * 2, config.nh), nn.BatchNorm1d(80), nn.Dropout(config.do_prob), act,
            nn.Linear(config.nh, config.nh), nn.BatchNorm1d(80), nn.Dropout(config.do_prob), act,
        )
        if self.do_reg:
            self.final_head = nn.Sequential(
                nn.Linear(config.nh, 1),
                ScaleLayer(config)
            )
        else:
            self.final_head = nn.Linear(config.nh, 950)

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'head.0' in name or "head.4" in name or "head.8" in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        for i in range(len(self.lstms)):
            self.lstms[i].flatten_parameters()
            x, _ = self.lstms[i](x)
        x = self.final_head(self.head(x))
        return x


class Model_PulpFiction(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        act = nn.SELU(inplace=False)
        hidden = config.hidden
        hidden_gru = config.hidden_gru
        use_bi = config.bidirectional
        self.do_reg = config.do_reg
        self.lstms = nn.ModuleList([
            nn.LSTM((1+use_bi) * hidden[i-1], hidden[i], batch_first=True, bidirectional=use_bi)
            if i > 0 else nn.LSTM(input_size, hidden[0], batch_first=True, bidirectional=use_bi)
            for i in range(len(hidden))
        ])
        self.grus = nn.ModuleList([
            nn.LSTM((1+use_bi) * hidden_gru[i-1], hidden_gru[i], batch_first=True, bidirectional=use_bi)
            if i > 0 else nn.GRU(hidden[1] * 2, hidden_gru[0], batch_first=True, bidirectional=use_bi)
            for i in range(len(hidden_gru))
        ])

        self.use_dp = len(config.gpu) > 1
        self.use_bn_after_lstm = config.use_bn_after_lstm
        if self.use_bn_after_lstm:
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
            if self.use_bn_after_lstm:
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


class Model_transformer(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        act = nn.SELU(inplace=False)
        hidden = config.hidden
        use_bi = config.bidirectional
        self.do_reg = config.do_reg
        self.seq_emb = nn.Sequential(
            nn.Linear(input_size, config.d_model),
            nn.LayerNorm(config.d_model),
            act,
            config.do_transformer,
        )
        self.pos_encoder = PositionalEncoding(d_model=config.d_model, dropout=config.do_transformer)
        encoder_layers = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_head,
                                                    dim_feedforward=config.dim_forward,
                                                    dropout=config.do_transformer, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config.num_layers)

        self.lstms = nn.ModuleList([
            nn.LSTM((1+use_bi) * hidden[i-1], hidden[i], batch_first=True, bidirectional=use_bi)
            if i > 0 else nn.LSTM(config.d_models, hidden[0], batch_first=True, bidirectional=use_bi)
            for i in range(len(config.hidden))
        ])
        self.use_dp = len(config.gpu) > 1
        self.use_bn_after_lstm = config.use_bn_after_lstm
        if self.use_bn_after_lstm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(80) for i in range(len(config.hidden))])

        # add batch normalization
        self.fc1 = nn.Linear(2 * hidden[-1], config.fc)
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
            if 'lstm' in name:
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
        x = self.seq_emb(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        for i in range(len(self.lstms)):
            if self.use_dp:
                self.lstms[i].flatten_parameters()
            x, _ = self.lstms[i](x)
            if self.use_bn_after_lstm:
                x = self.bns[i](x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.do_reg:
            x = self.scaler(x)
        return x