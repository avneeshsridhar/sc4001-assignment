import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import LayerAttribution, LayerGradCam

from common import NPZ, SEED, AlexNet, ResNet, VGG, dev, load, outdir


def gradcam(m, layer, x):
    m.eval()
    gc = LayerGradCam(m, layer)
    a = gc.attribute(x, target=0, relu_attributions=True)
    a = LayerAttribution.interpolate(a, (64, 64), interpolate_mode="bilinear").squeeze(1)
    lo = a.amin((1, 2), keepdim=True)
    hi = a.amax((1, 2), keepdim=True)
    a = (a - lo) / (hi - lo + 1e-8)
    with torch.no_grad():
        p = torch.sigmoid(m(x)).squeeze(1)
    return a.detach().cpu().numpy(), p.cpu().numpy()


def main():
    out = outdir("b4")
    data = load()
    x, y = data["test"]
    raw = np.load(NPZ)["test_images"]

    rng = np.random.default_rng(SEED)
    idx = rng.choice(np.where(y.cpu().numpy() == 1)[0], 5, replace=False)
    print(f"test indices: {idx.tolist()}")
    xs = x[idx]

    models = {}
    for name, cls, path in [("B1 AlexNet", AlexNet, "b1"), ("B2 VGG", VGG, "b2"), ("B3 ResNet", ResNet, "b3")]:
        m = cls().to(dev)
        m.load_state_dict(torch.load(f"outputs/{path}/model.pt", map_location=dev))
        models[name] = m
    layers = {
        "B1 AlexNet": models["B1 AlexNet"].features[4],
        "B2 VGG": models["B2 VGG"].features[7],
        "B3 ResNet": models["B3 ResNet"].b3,
    }

    maps = {name: gradcam(m, layers[name], xs) for name, m in models.items()}

    fig, ax = plt.subplots(5, 4, figsize=(10, 13))
    for r, i in enumerate(idx):
        ax[r, 0].imshow(raw[i], cmap="gray")
        ax[r, 0].set_title(f"test #{i}", fontsize=9)
        for c, name in enumerate(models, 1):
            a, p = maps[name]
            ax[r, c].imshow(raw[i], cmap="gray")
            ax[r, c].imshow(a[r], cmap="jet", alpha=0.4)
            ax[r, c].set_title(f"{name}\np(pneumonia)={p[r]:.2f}", fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("B4 Grad-CAM on pneumonia X-rays")
    fig.tight_layout()
    fig.savefig(f"{out}/gradcam.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
