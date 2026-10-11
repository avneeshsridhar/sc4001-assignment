# Part B: chest X-ray classification

## Environment
- Windows 11, Python 3.10.9, NVIDIA RTX 4050 Laptop GPU (CUDA 11.8); also runs on CPU
- torch 2.7.1, numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.7, captum 0.9.0, thop 0.1.1

```bash
pip install torch numpy pandas matplotlib captum thop
```

## Data
Put `chestxray_binary_64.npz` in the folder you run the commands from (the parent of `partB/`).

## Seeds
All runs use seed 42 (model init, batch shuffling, and picking the 5 Grad-CAM images). cuDNN is set to deterministic.

## Run (in order)
```bash
python partB/b1.py
python partB/b2.py
python partB/b3.py
python partB/b4.py
```

Outputs go to `outputs/b1` ... `outputs/b4`. B2 reads the best learning rate from B1, B3 reads both, and B4 loads the three saved models.

```text
partB/common.py   data loading, AlexNet / VGG / ResNet models, training loop, metrics, plots
partB/b1.py       AlexNet learning-rate sweep (batch 128)
partB/b2.py       VGG batch-size sweep (best lr from B1)
partB/b3.py       ResNet with best lr and batch size, comparison table (params, FLOPs)
partB/b4.py       Grad-CAM overlays on 5 random pneumonia test X-rays
```
