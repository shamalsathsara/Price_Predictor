Retraining with a Pipeline (recommended)

This repo now includes `model/train_pipeline.py` which trains and saves a full sklearn `Pipeline` that includes preprocessing (ColumnTransformer + OneHotEncoder) followed by a `RandomForestRegressor`.

Why this is better
- The Pipeline stores both preprocessing and model together, ensuring the same encoding/column order at inference as during training.

Quick steps
1. Prepare CSV: columns `ram,weight,company,typename,opsys,cpuname,gpuname,price`. Optional `touchscreen,ips` as 0/1.
2. Install deps:

```bash
pip install -r requirements.txt
pip install pandas
```

3. Train:

```bash
python model/train_pipeline.py --data data/laptops.csv --out model/predictor.pickle
```

4. Restart the Flask app:

```bash
python model/app.py
```

Notes
- `OneHotEncoder(handle_unknown='ignore')` is used so unseen categories won't break inference; they will be treated as zero-valued columns.
- If your dataset uses different category labels, they will be encoded accordingly during training; ensure production inputs match training labels when possible.
