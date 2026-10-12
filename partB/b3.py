import json

import pandas as pd
import torch
from torch import nn

from common import SEED, AlexNet, ResNet, VGG, count, curve_plot, dev, evaluate, load, outdir, save_json, train


def flops(m):
    tot = [0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            d1, f = mod.in_channels, mod.kernel_size[0]
            d2, h2, w2 = out.shape[1:]
            tot[0] += 2 * d1 * f * f * d2 * h2 * w2
        elif isinstance(mod, nn.Linear):
            tot[0] += 2 * mod.in_features * mod.out_features

    for mod in m.modules():
        mod.register_forward_hook(hook)
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 1, 64, 64))
    return tot[0]


def main():
    out = outdir("b3")
    data = load()
    lr = json.load(open("outputs/b1/best.json"))["lr"]
    bs = json.load(open("outputs/b2/best.json"))["bs"]

    print(f"lr {lr}  batch size {bs}")
    torch.manual_seed(SEED)
    m = ResNet().to(dev)
    h, ep = train(m, data, lr, bs)
    curve_plot(h, out, f"B3 ResNet (lr={lr}, bs={bs})")
    va = evaluate(m, *data["val"])
    te = evaluate(m, *data["test"])
    print(f"best epoch {ep}  val error {va['error']:.4f}")
    print(f"test acc {te['acc']:.4f}  sens {te['sens']:.4f}  spec {te['spec']:.4f}")
    torch.save(m.state_dict(), f"{out}/model.pt")
    save_json({"lr": lr, "bs": bs, "best_epoch": ep, "val": va, "test": te}, f"{out}/best.json")

    results = {
        "B1 AlexNet": (AlexNet, json.load(open("outputs/b1/best.json"))["test"]),
        "B2 VGG": (VGG, json.load(open("outputs/b2/best.json"))["test"]),
        "B3 ResNet": (ResNet, te),
    }
    rows = []
    for name, (cls, t) in results.items():
        rows.append({
            "model": name, "accuracy": t["acc"], "sensitivity": t["sens"], "specificity": t["spec"],
            "error": t["error"], "params": count(cls()), "MFLOPs": flops(cls()) / 1e6,
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/compare.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
