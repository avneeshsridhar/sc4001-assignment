# HDB price baseline

## Run it

Install the exact environment recorded in `uv.lock`:

```bash
uv sync
```

Run the baseline with:

```bash
uv run python src/baseline.py
```

Run the command from the repository root. The input CSV and output directory are constants at the top of `src/baseline.py`, while the data split, features, model size, and training hyperparameters are defined in `src/config.py`.

## Code map

```text
src/config.py       run configuration
src/data.py         split, fit preprocessing, tensor conversion
src/model.py        categorical embeddings and MLP
src/baseline.py     single-run training, RMSE evaluation, and model export
```

## Part A solutions

### Environment
- Windows 11, Python 3.10.9, CPU training
- torch 2.7.1, numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.7, ray[tune] 2.59.0, captum 0.9.0

```bash
pip install torch numpy pandas matplotlib "ray[tune]" captum
```

### Seeds
- A1 grid search and retraining: 42
- A2 / A3 five-seed runs: 0, 1, 2, 3, 4
- A3 lambda search and A4: 42

### Run (from repository root, in order)
```bash
python src/a1.py
python src/a2.py
python src/a3.py
python src/a4.py
```

Each script writes tables and plots to `outputs/a1` ... `outputs/a4`. Later scripts read the chosen settings from `outputs/a1/best.json` and `outputs/a3/best.json`.

```text
src/common.py   shared model (baseline / ablations / wide-and-deep), training loop, metrics
src/a1.py       Ray Tune grid search (hidden width x embedding dim), retrain on 2017-2021, test on 2022
src/a2.py       ablations: no categorical inputs, sigmoid activation (5 seeds)
src/a3.py       wide-and-deep lambda search and 5-seed comparison
src/a4.py       Captum FeatureAblation importance in SGD
```
