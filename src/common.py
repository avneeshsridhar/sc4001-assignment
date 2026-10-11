import copy
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import prepare_data

CSV = "hdb_price_prediction.csv"
OUT = "outputs"


def load(final=False):
    cfg = Config()
    if final:
        cfg.train_years = (2017, 2018, 2019, 2020, 2021)
    return prepare_data(CSV, cfg)


def outdir(name):
    path = os.path.join(OUT, name)
    os.makedirs(path, exist_ok=True)
    return path


class Net(nn.Module):
    def __init__(self, cards, mean, std, emb=8, hidden=64, use_cat=True, act="relu", lam=None):
        super().__init__()
        self.use_cat = use_cat
        self.lam = lam
        self.mean = mean
        self.std = std
        n = 6
        if use_cat:
            self.embs = nn.ModuleList(nn.Embedding(c, emb) for c in cards)
            n += emb * len(cards)
        self.mlp = nn.Sequential(
            nn.Linear(n, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU() if act == "relu" else nn.Sigmoid(),
            nn.Linear(hidden, 1),
        )
        if lam is not None:
            self.wide = nn.Linear(n, 1)

    def forward(self, cat, cont):
        x = cont
        if self.use_cat:
            x = torch.cat([e(cat[:, i]) for i, e in enumerate(self.embs)] + [cont], 1)
        out = self.mlp(x)
        if self.lam is not None:
            out = out + self.lam * self.wide(x)
        return out.squeeze(1)

    def price(self, cat, cont):
        return self(cat, cont) * self.std + self.mean


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, ds):
    model.eval()
    cat, cont, y = ds.tensors
    with torch.no_grad():
        p = model.price(cat, cont)
    rmse = (p - y).square().mean().sqrt().item()
    r2 = 1 - (p - y).square().sum().item() / (y - y.mean()).square().sum().item()
    return rmse, r2


def train(model, tr, va=None, epochs=20, stop=True, cb=None):
    cfg = Config()
    loader = DataLoader(tr, cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    hist = {"train": [], "val": []}
    best, best_w, best_ep, wait = float("inf"), None, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        for cat, cont, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(cat, cont), (y - model.mean) / model.std)
            loss.backward()
            opt.step()

        hist["train"].append(evaluate(model, tr)[0])
        if va is not None:
            hist["val"].append(evaluate(model, va)[0])
        if cb:
            cb(hist)

        if not stop:
            continue
        v = hist["val"][-1]
        if v < best - cfg.early_stopping_threshold:
            best, best_w, best_ep, wait = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait == cfg.early_stopping_patience:
                break

    if stop:
        model.load_state_dict(best_w)
    return hist, best_ep
