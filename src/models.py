from torch import nn


def get_model(input_size, config):
    if config.model_module == "BASE":
        return Model(input_size, config)
    elif config.model_module == "CH":
        return Model_CH(input_size, config)


class Model(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        act = nn.SELU(inplace=False)
        hidden = config.hidden
        use_bi = config.bidirectional
        self.lstms = nn.ModuleList([
            nn.LSTM((1+use_bi) * hidden[i-1], hidden[i], batch_first=True, bidirectional=use_bi)
            if i > 0 else nn.LSTM(input_size, hidden[0], batch_first=True, bidirectional=use_bi)
            for i in range(len(config.hidden))
        ])
        self.fc1 = nn.Linear(2 * hidden[-1], 50)
        self.act = act
        self.fc2 = nn.Linear(50, 1)
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
        return x


class Model_CH(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        act = nn.SELU(inplace=False)
        hidden = config.hidden
        use_bi = config.bidirectional
        self.lstms = nn.ModuleList([
            nn.LSTM((1+use_bi) * hidden[i-1], hidden[i], batch_first=True, bidirectional=use_bi)
            if i > 0 else nn.LSTM(input_size, hidden[0], batch_first=True, bidirectional=use_bi)
            for i in range(len(config.hidden))
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden[-1] * 2, config.nh), nn.BatchNorm1d(config.nh), nn.Dropout(config.do_prob), act,
            nn.Linear(config.nh, config.nh), nn.BatchNorm1d(config.nh), nn.Dropout(config.do_prob), act,
            nn.Linear(config.nh, 1),
        )
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
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        x = self.head(x)
        x = x.reshape(-1, 80, 1)
        return x

## TODO: transformer based model