import json

import matplotlib.pyplot as plt
import pandas as pd
import torch

from common import Net, count, evaluate, load, outdir, train

SEED = 42
SEEDS = [0, 1, 2, 3, 4]
LAMS = [0, 0.25, 0.5, 1, 2]


def main():
    out = outdir("a3")
    best = json.load(open("outputs/a1/best.json"))

    data = load()
    rows = []
    for lam in LAMS:
        torch.manual_seed(SEED)
        m = Net(data["cardinalities"], data["target_mean"], data["target_std"], best["emb"], best["hidden"], lam=lam)
        hist, ep = train(m, data["train"], data["validation"])
        rows.append({"lam": lam, "best_val_rmse": min(hist["val"]), "best_epoch": ep})
        plt.plot(range(1, len(hist["val"]) + 1), hist["val"], "o-", label=f"λ={lam}")
        print(f"lam {lam}: best val RMSE {min(hist['val']):,.0f} at epoch {ep}")
    plt.xlabel("epoch")
    plt.ylabel("validation RMSE (SGD)")
    plt.title("A3 λ search")
    plt.legend()
    plt.savefig(f"{out}/lambda_val_curves.png", dpi=150)
    plt.close()

    search = pd.DataFrame(rows)
    search.to_csv(f"{out}/lambda_search.csv", index=False)
    lam = float(search.loc[search["best_val_rmse"].idxmin(), "lam"])
    print(f"\nbest lambda: {lam}")

    data = load(final=True)
    rows = []
    for s in SEEDS:
        torch.manual_seed(s)
        m = Net(data["cardinalities"], data["target_mean"], data["target_std"], best["emb"], best["hidden"], lam=lam)
        train(m, data["train"], epochs=best["epochs"], stop=False)
        rmse, r2 = evaluate(m, data["test"])
        rows.append({"model": f"wide_deep_lam{lam}", "seed": s, "params": count(m), "test_rmse": rmse, "test_r2": r2})
        print(f"seed {s}: RMSE {rmse:,.0f}  R2 {r2:.4f}")

    base = pd.read_csv("outputs/a2/runs.csv")
    df = pd.concat([base[base["model"] == "baseline"], pd.DataFrame(rows)])
    df.to_csv(f"{out}/runs.csv", index=False)
    summary = df.groupby("model", sort=False).agg(
        params=("params", "first"),
        rmse_mean=("test_rmse", "mean"), rmse_std=("test_rmse", "std"),
        r2_mean=("test_r2", "mean"), r2_std=("test_r2", "std"),
    )
    summary.to_csv(f"{out}/summary.csv")
    print(summary.round(4))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    names = list(summary.index)
    ax[0].bar(names, summary["rmse_mean"], yerr=summary["rmse_std"], capsize=5)
    ax[0].set_ylabel("test RMSE (SGD)")
    ax[1].bar(names, summary["r2_mean"], yerr=summary["r2_std"], capsize=5)
    ax[1].set_ylabel("test R2")
    for name in names:
        d = df[df["model"] == name]
        ax[2].plot(d["seed"], d["test_rmse"], "o-", label=name)
    ax[2].set_xlabel("seed")
    ax[2].set_ylabel("test RMSE (SGD)")
    ax[2].legend()
    fig.suptitle("A3 wide-and-deep vs baseline (5 seeds)")
    fig.tight_layout()
    fig.savefig(f"{out}/compare.png", dpi=150)
    plt.close(fig)

    json.dump({"lam": lam}, open(f"{out}/best.json", "w"))


if __name__ == "__main__":
    main()
