import pandas as pd
import torch

from common import SEED, AlexNet, curve_plot, dev, evaluate, load, outdir, save_json, sweep_plots, train

LRS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
BS = 128


def main():
    out = outdir("b1")
    data = load()

    rows, hists, models = [], {}, {}
    for lr in LRS:
        print(f"lr {lr}")
        torch.manual_seed(SEED)
        m = AlexNet().to(dev)
        h, ep = train(m, data, lr, BS)
        va = evaluate(m, *data["val"])
        rows.append({"lr": lr, "best_epoch": ep, "val_loss": va["loss"], "val_error": va["error"],
                     "val_sens": va["sens"], "val_spec": va["spec"]})
        hists[lr] = h
        models[lr] = m

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/lr_table.csv", index=False)
    print(df.round(4))
    sweep_plots(hists, "lr", out, "B1 AlexNet")

    lr = float(df.sort_values(["val_error", "val_loss"]).iloc[0]["lr"])
    m = models[lr]
    curve_plot(hists[lr], out, f"B1 AlexNet (lr={lr})")
    te = evaluate(m, *data["test"])
    print(f"\nbest lr: {lr}")
    print(f"test acc {te['acc']:.4f}  sens {te['sens']:.4f}  spec {te['spec']:.4f}")

    torch.save(m.state_dict(), f"{out}/model.pt")
    save_json({"lr": lr, "bs": BS, "test": te}, f"{out}/best.json")


if __name__ == "__main__":
    main()
