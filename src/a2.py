import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import Net, count, evaluate, load, outdir, train

SEEDS = [0, 1, 2, 3, 4]


def main():
    out = outdir("a2")
    best = json.load(open("outputs/a1/best.json"))
    data = load(final=True)

    variants = {
        "baseline": {},
        "no_categorical": {"use_cat": False},
        "sigmoid": {"act": "sigmoid"},
    }

    rows = []
    curves = {}
    for name, kw in variants.items():
        curves[name] = []
        for s in SEEDS:
            torch.manual_seed(s)
            m = Net(data["cardinalities"], data["target_mean"], data["target_std"], best["emb"], best["hidden"], **kw)
            hist, _ = train(m, data["train"], epochs=best["epochs"], stop=False)
            rmse, r2 = evaluate(m, data["test"])
            rows.append({"model": name, "seed": s, "params": count(m), "test_rmse": rmse, "test_r2": r2})
            curves[name].append(hist["train"])
            print(f"{name} seed {s}: RMSE {rmse:,.0f}  R2 {r2:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/runs.csv", index=False)
    summary = df.groupby("model", sort=False).agg(
        params=("params", "first"),
        rmse_mean=("test_rmse", "mean"), rmse_std=("test_rmse", "std"),
        r2_mean=("test_r2", "mean"), r2_std=("test_r2", "std"),
    )
    summary.to_csv(f"{out}/summary.csv")
    print(summary.round(4))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    names = list(summary.index)
    ax[0].bar(names, summary["rmse_mean"], yerr=summary["rmse_std"], capsize=5)
    ax[0].set_ylabel("test RMSE (SGD)")
    ax[1].bar(names, summary["r2_mean"], yerr=summary["r2_std"], capsize=5)
    ax[1].set_ylabel("test R2")
    for a in ax:
        a.tick_params(axis="x", rotation=15)
    fig.suptitle("A2 ablations (mean ± s.d. over 5 seeds)")
    fig.tight_layout()
    fig.savefig(f"{out}/ablation_bars.png", dpi=150)
    plt.close(fig)

    ep = range(1, best["epochs"] + 1)
    for name, c in curves.items():
        c = np.array(c)
        plt.plot(ep, c.mean(0), "o-", label=name)
        plt.fill_between(ep, c.mean(0) - c.std(0), c.mean(0) + c.std(0), alpha=0.2)
    plt.xlabel("epoch")
    plt.ylabel("train RMSE (SGD)")
    plt.title("A2 training curves (mean ± s.d. over 5 seeds)")
    plt.legend()
    plt.savefig(f"{out}/train_curves.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
