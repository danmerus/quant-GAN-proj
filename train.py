import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models
import preprocess

def cuda(xs):
    """
    helper function to put obj on cuda
    """
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]

def cmdline_args():
    p = argparse.ArgumentParser()
    p.add_argument("data_path",
                   help="path to time-series in csv")
    p.add_argument("--column", default='Price',
                   help="column with time-series")
    p.add_argument("--replace_commas", default='True', type=bool,
                   help="commas present in dataset")
    p.add_argument("--reverse", default='True', type=bool,
                   help="datset in reversed order")
    p.add_argument("--epochs", default=100, type=int,
                   help="number of train epochs")
    p.add_argument("--lr", default=0.00004, type=float,
                   help="learning rate in AdamW optimizer")
    p.add_argument("--ts_len", default=127, type=int,
                   help="length of training samples")
    p.add_argument("--bs", default=2, type=int,
                   help="batch size")

    return p.parse_args()


if __name__ == '__main__':
    params = cmdline_args()
    print(f'Input params: {params},\ndata processing...')
    data = preprocess.prepare_dataset(params.data_path, params.column,
                                      replace_commas=params.replace_commas, reverse=params.reverse)
    data_cropped = data['t_rolling'][126:]
    data_loader = DataLoader(preprocess.TimeseriesLoader(data_cropped, params.ts_len),
                             batch_size=params.bs, shuffle=True, drop_last=True)
    D = models.Discriminator(params.ts_len)
    G = models.Generator()
    bce = nn.BCELoss()

    cuda([D, G, bce])

    d_optimizer = torch.optim.AdamW(D.parameters(), lr=params.lr, betas=(0.9, 0.999), weight_decay=0.03)
    g_optimizer = torch.optim.AdamW(G.parameters(), lr=params.lr, betas=(0.9, 0.999), weight_decay=0.03)
    for epoch in range(params.epochs):
        D_ls = []
        G_ls = []
        for i, ts in enumerate(data_loader):
            # set train
            G.train()
            ts = Variable(ts).float()
            bs = ts.size(0)
            z = Variable(torch.randn(bs, 3, params.ts_len))
            r_lbl = Variable(torch.ones(bs))
            f_lbl = Variable(torch.zeros(bs))
            ts, z, r_lbl, f_lbl = cuda([ts, z, r_lbl, f_lbl])

            f_ts = G(z)

            # train D
            r_logit = D(ts)
            f_logit = D(f_ts.detach())

            d_r_loss = bce(r_logit, r_lbl)
            d_f_loss = bce(f_logit, f_lbl)
            d_loss = d_r_loss + d_f_loss
            D_ls.append(d_loss.item())

            D.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_value_(D.parameters(), 1)
            d_optimizer.step()

            # train G
            f_logit = D(f_ts)
            g_loss = bce(f_logit, r_lbl)
            G_ls.append(g_loss.item())

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_value_(G.parameters(), 1)
            g_optimizer.step()

        print(f'Epoch {epoch}, Discriminator loss:{np.mean(D_ls)}, Generator loss:{np.mean(G_ls)}')
    torch.save(G, 'QuantGenerator.pt')
    torch.save(D, 'QuantDiscriminator.pt')
    print('Training complete, models saved!')

