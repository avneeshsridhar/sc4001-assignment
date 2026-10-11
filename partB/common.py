import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

NPZ = "chestxray_binary_64.npz"
OUT = "outputs"
SEED = 42
dev = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def outdir(name):
    path = os.path.join(OUT, name)
    os.makedirs(path, exist_ok=True)
    return path


def load():
    d = np.load(NPZ)
    tr = d["train_images"] / 255.0
    mean, std = tr.mean(), tr.std()
    data = {}
    for k in ["train", "val", "test"]:
        x = (d[f"{k}_images"] / 255.0 - mean) / std
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(dev)
        y = torch.tensor(d[f"{k}_labels"], dtype=torch.float32).to(dev)
        data[k] = (x, y)
    return data


def conv(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv(1, 16, 5), nn.MaxPool2d(2),
            conv(16, 32, 5), nn.MaxPool2d(2),
            conv(32, 64, 3), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(self.features(x))


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv(1, 8), conv(8, 8), nn.MaxPool2d(2),
            conv(8, 16), conv(16, 16), nn.MaxPool2d(2),
            conv(16, 32), conv(32, 32), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        return self.fc(self.features(x))


class Block(nn.Module):
    def __init__(self, cin, cout, s):
        super().__init__()
        self.c1 = conv(cin, cout, 3, s)
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.short = nn.Identity()
        if s != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, s, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.c2(self.c1(x)) + self.short(x))


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv(1, 16), nn.MaxPool2d(2))
        self.b1 = Block(16, 16, 1)
        self.b2 = Block(16, 32, 2)
        self.b3 = Block(32, 64, 2)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(self.b3(self.b2(self.b1(self.stem(x)))))


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    out = model(x).squeeze(1)
    loss = nn.functional.binary_cross_entropy_with_logits(out, y).item()
    p = (out > 0).float()
    tp = ((p == 1) & (y == 1)).sum().item()
    tn = ((p == 0) & (y == 0)).sum().item()
    fp = ((p == 1) & (y == 0)).sum().item()
    fn = ((p == 0) & (y == 1)).sum().item()
    return {
        "loss": loss,
        "acc": (tp + tn) / len(y),
        "error": 1 - (tp + tn) / len(y),
        "sens": tp / (tp + fn) if tp + fn else 0.0,
        "spec": tn / (tn + fp) if tn + fp else 0.0,
    }


def train(model, data, lr, bs, epochs=20, patience=4, delta=1e-4):
    xtr, ytr = data["train"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best, best_w, best_ep, wait = float("inf"), copy.deepcopy(model.state_dict()), 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(ytr), generator=g).to(dev)
        for i in range(0, len(ytr), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(xtr[b]).squeeze(1), ytr[b])
            loss.backward()
            opt.step()

        tr = evaluate(model, *data["train"])
        va = evaluate(model, *data["val"])
        hist["train_loss"].append(tr["loss"])
        hist["train_acc"].append(tr["acc"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        print(f"  epoch {ep:02d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | val loss {va['loss']:.4f} acc {va['acc']:.4f}")

        if va["loss"] < best - delta:
            best, best_w, best_ep, wait = va["loss"], copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait == patience:
                break

    model.load_state_dict(best_w)
    return hist, best_ep


def sweep_plots(hists, label, out, prefix):
    for key, name in [("val_loss", "validation loss"), ("val_acc", "validation accuracy")]:
        for k, h in hists.items():
            plt.plot(range(1, len(h[key]) + 1), h[key], "o-", label=f"{label}={k}")
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.title(f"{prefix}: {name}")
        plt.legend()
        plt.savefig(f"{out}/{key}.png", dpi=150)
        plt.close()


def curve_plot(h, out, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ep = range(1, len(h["train_loss"]) + 1)
    ax[0].plot(ep, h["train_loss"], "o-", label="train")
    ax[0].plot(ep, h["val_loss"], "o-", label="validation")
    ax[0].set_ylabel("loss")
    ax[1].plot(ep, h["train_acc"], "o-", label="train")
    ax[1].plot(ep, h["val_acc"], "o-", label="validation")
    ax[1].set_ylabel("accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{out}/learning_curves.png", dpi=150)
    plt.close(fig)


def save_json(obj, path):
    json.dump(obj, open(path, "w"), indent=2)
