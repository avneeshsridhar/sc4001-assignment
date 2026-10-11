import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
from captum.attr import FeatureAblation

from common import Net, load, outdir, train
from config import Config

SEED = 42


def importance(m, data):
    m.eval()
    cat, cont, _ = data["test"].tensors
    tr_cat = data["train"].tensors[0]
    base_cat = tr_cat.mode(0).values.unsqueeze(0)
    base_cont = torch.zeros(1, cont.shape[1])
    mask_cat = torch.arange(4).unsqueeze(0)
    mask_cont = torch.arange(4, 10).unsqueeze(0)

    fa = FeatureAblation(m.price)
    with torch.no_grad():
        a_cat, a_cont = fa.attribute((cat, cont), baselines=(base_cat, base_cont), feature_mask=(mask_cat, mask_cont))
    return torch.cat([a_cat, a_cont], 1).abs().mean(0)


def main():
    out = outdir("a4")
    cfg = Config()
    names = list(cfg.categorical_features) + list(cfg.continuous_features)
    best = json.load(open("outputs/a1/best.json"))
    lam = json.load(open("outputs/a3/best.json"))["lam"]
    data = load(final=True)

    models = {"baseline": None, f"wide_deep (lam={lam})": lam}
    scores = {}
    for name, l in models.items():
        torch.manual_seed(SEED)
        m = Net(data["cardinalities"], data["target_mean"], data["target_std"], best["emb"], best["hidden"], lam=l)
        train(m, data["train"], epochs=best["epochs"], stop=False)
        scores[name] = importance(m, data).numpy()

    df = pd.DataFrame(scores, index=names)
    df.to_csv(f"{out}/importance.csv")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, name in zip(ax, scores):
        s = df[name].sort_values(ascending=False)
        a.bar(s.index, s.values)
        a.set_title(name)
        a.set_ylabel("mean |attribution| (SGD)")
        a.tick_params(axis="x", rotation=60)
        print(f"{name} top 3: {list(s.index[:3])}")
        print(s.round(0))
    fig.suptitle("A4 FeatureAblation importance on 2022 test set")
    fig.tight_layout()
    fig.savefig(f"{out}/importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
