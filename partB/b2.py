import json

import pandas as pd
import torch

from common import SEED, VGG, curve_plot, dev, evaluate, load, outdir, save_json, sweep_plots, train

BATCHES = [16, 32, 64, 128, 256]


def main():
    out = outdir("b2")
    data = load()
    lr = json.load(open("outputs/b1/best.json"))["lr"]

    rows, hists, models = [], {}, {}
    for bs in BATCHES:
        print(f"batch size {bs}")
        torch.manual_seed(SEED)
        m = VGG().to(dev)
        h, ep = train(m, data, lr, bs)
        va = evaluate(m, *data["val"])
        rows.append({"batch_size": bs, "best_epoch": ep, "val_loss": va["loss"], "val_error": va["error"],
                     "val_sens": va["sens"], "val_spec": va["spec"]})
        hists[bs] = h
        models[bs] = m

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/bs_table.csv", index=False)
    print(df.round(4))
    sweep_plots(hists, "bs", out, f"B2 VGG (lr={lr})")

    bs = int(df.sort_values(["val_error", "val_loss"]).iloc[0]["batch_size"])
    m = models[bs]
    curve_plot(hists[bs], out, f"B2 VGG (lr={lr}, bs={bs})")
    te = evaluate(m, *data["test"])
    print(f"\nbest batch size: {bs}")
    print(f"test acc {te['acc']:.4f}  sens {te['sens']:.4f}  spec {te['spec']:.4f}")

    torch.save(m.state_dict(), f"{out}/model.pt")
    save_json({"lr": lr, "bs": bs, "test": te}, f"{out}/best.json")


if __name__ == "__main__":
    main()
