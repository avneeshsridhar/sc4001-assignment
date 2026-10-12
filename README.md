# Part A

## Environment
Windows 11, Python 3.10.9, CPU

torch 2.7.1, numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.7, ray 2.59.0, captum 0.9.0

```
pip install torch numpy pandas matplotlib "ray[tune]" captum
```

`hdb_price_prediction.csv` goes in the main folder.

## Seeds
- A1: 42
- A2, A3: 10, 13, 42, 33, 104
- A4: 42

## Run
```
python src/a1.py
python src/a2.py
python src/a3.py
python src/a4.py
```

Outputs are saved in `outputs/`.
