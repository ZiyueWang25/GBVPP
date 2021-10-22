import torch
from torch import nn
import numpy as np

def get_model(input_size, config):
    if config.model_module == "BASE":
        return Model(input_size, config)
    elif config.model_module == "CH":
        return Model_CH(input_size, config)


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
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        for i in range(len(self.lstms)):
            self.lstms[i].flatten_parameters()
            x, _ = self.lstms[i](x)
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