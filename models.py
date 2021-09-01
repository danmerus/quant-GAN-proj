import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    layer to get rid of excess padding
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal convolution block based on https://arxiv.org/pdf/1803.01271.pdf
    with PReLU as activation
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding=0, dropout=0.05):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        if padding > 0:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.prelu1, self.dropout1,
                                     self.conv2, self.chomp2, self.prelu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.prelu1, self.dropout1,
                                     self.conv2, self.prelu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.prelu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.75)
        self.conv2.weight.data.normal_(0, 0.75)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.75)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out, self.prelu(out + res)


class Generator(nn.Module):
    '''
    Replicated from Quant GANs paper: https://arxiv.org/pdf/1907.06673.pdf
    skip connections used
    '''

    def __init__(self):
        super(Generator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(3, 80, kernel_size=1, dilation=1, padding=0),
                                  *[TemporalBlock(80, 80, kernel_size=2, dilation=2 ** i, padding=2 ** i) for i in range(6)]])
        self.one = nn.Conv1d(80, 1, kernel_size=1, dilation=1)

    def forward(self, x):
        skips = []
        for layer in self.tcn:
            skip, x = layer(x)
            skips.append(skip)
        x = self.one(x + sum(skips))
        return x


class Discriminator(nn.Module):
    """
    Replicated from Quant GANs paper: https://arxiv.org/pdf/1907.06673.pdf
    skip connections used
    """

    def __init__(self, seq_len, conv_dropout=0.05):
        super(Discriminator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(1, 80, kernel_size=1, dilation=1, padding=0),
                                  *[TemporalBlock(80, 80, kernel_size=2, dilation=2 ** i, padding=2 ** i) for i in range(6)]])
        self.one = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for layer in self.tcn:
            skip, x = layer(x)
            skips.append(skip)
        x = self.one(x + sum(skips))
        return self.to_prob(x).squeeze()





