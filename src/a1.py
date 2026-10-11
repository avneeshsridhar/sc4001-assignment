import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ray import tune

from common import Net, count, evaluate, load, outdir, train

SEED = 42


def trial(cfg, data):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    m = Net(data["cardinalities"], data["target_mean"], data["target_std"], cfg["emb"], cfg["hidden"])
    train(m, data["train"], data["validation"], cb=lambda h: tune.report({
        "train_rmse": h["train"][-1],
        "val_rmse": h["val"][-1],
        "best_val": min(h["val"]),
    }))


def main():
    out = outdir("a1")
    data = load()

    tuner = tune.Tuner(
        tune.with_parameters(trial, data=data),
        param_space={
            "hidden": tune.grid_search([16, 32, 64, 128, 256]),
            "emb": tune.grid_search([2, 4, 8, 12, 16]),
        },
        tune_config=tune.TuneConfig(metric="best_val", mode="min"),
        run_config=tune.RunConfig(storage_path=os.path.abspath("outputs/ray"), name="a1", verbose=1),
    )
    res = tuner.fit()

    rows = []
    for r in res:
        df = r.metrics_dataframe
        i = df["val_rmse"].idxmin()
        rows.append({
            "hidden": r.config["hidden"],
            "emb": r.config["emb"],
            "best_val_rmse": df["val_rmse"][i],
            "best_epoch": int(df["training_iteration"][i]),
        })
    grid = pd.DataFrame(rows).sort_values("best_val_rmse")
    grid.to_csv(f"{out}/grid.csv", index=False)
    print(grid.pivot(index="emb", columns="hidden", values="best_val_rmse").round(0))

    top = grid.iloc[0]
    hidden, emb, epochs = int(top["hidden"]), int(top["emb"]), int(top["best_epoch"])
    print(f"\nbest: hidden={hidden} emb={emb} best_epoch={epochs} val_rmse={top['best_val_rmse']:,.0f}")

    df = res.get_best_result().metrics_dataframe
    plt.plot(df["training_iteration"], df["train_rmse"], "o-", label="train")
    plt.plot(df["training_iteration"], df["val_rmse"], "o-", label="validation")
    plt.axvline(epochs, color="gray", ls="--", label="best epoch")
    plt.xlabel("epoch")
    plt.ylabel("RMSE (SGD)")
    plt.title(f"A1 selected model (hidden={hidden}, emb={emb})")
    plt.legend()
    plt.savefig(f"{out}/train_val_curve.png", dpi=150)
    plt.close()

    data = load(final=True)
    torch.manual_seed(SEED)
    m = Net(data["cardinalities"], data["target_mean"], data["target_std"], emb, hidden)
    hist, _ = train(m, data["train"], data["test"], epochs=epochs, stop=False)
    rmse, r2 = evaluate(m, data["test"])
    print(f"params: {count(m)}")
    print(f"test RMSE: {rmse:,.0f}  test R2: {r2:.4f}")

    ep = range(1, epochs + 1)
    plt.plot(ep, hist["train"], "o-", label="train (2017-2021)")
    plt.plot(ep, hist["val"], "o-", label="test (2022)")
    plt.xlabel("epoch")
    plt.ylabel("RMSE (SGD)")
    plt.title("A1 retrained on train+val")
    plt.legend()
    plt.savefig(f"{out}/train_test_curve.png", dpi=150)
    plt.close()

    json.dump({
        "hidden": hidden, "emb": emb, "epochs": epochs,
        "val_rmse": float(top["best_val_rmse"]), "params": count(m),
        "test_rmse": rmse, "test_r2": r2,
    }, open(f"{out}/best.json", "w"), indent=2)


if __name__ == "__main__":
    main()
