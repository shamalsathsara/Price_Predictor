# Developer Manual — Price Predicter ML (Website)

This document helps developers inspect, run, debug, retrain, and extend the project.
Keep it concise and reference code in `model/`, `templates/`, and `static/`.

## Project Overview
- Purpose: A small Flask web app that predicts laptop prices using a saved sklearn `Pipeline` (`model/predictor.pickle`).
- Web UI: `templates/index.html` and `static/style.css`.
- App entry: `model/app.py` — loads `predictor.pickle`, handles form submissions, converts EUR→LKR.
- Training utility: `model/train_pipeline.py` — builds and saves the full preprocessing+estimator pipeline.

## Repository Layout (important files)
- `model/app.py` — Flask application and inference wrapper.
- `model/train_pipeline.py` — training & pipeline building script.
- `model/predictor.pickle` — saved sklearn `Pipeline` (preprocessing + estimator).
- `templates/index.html` — HTML form and result display.
- `static/style.css` — CSS for UI.
- `data/` — expected place for training CSV (if present).
- `DEVELOPER.md` — this file.

## Development Setup
1. Create and activate Python virtual environment (recommended):
```bash
python -m venv env
# Windows
env\Scripts\activate
# Unix
source env/bin/activate
```
2. Install dependencies (project may not contain `requirements.txt` up-to-date; at minimum install):
```bash
pip install flask pandas scikit-learn joblib numpy scipy
```
3. Verify Python path used by VS Code or terminal points to the activated venv.

## Running the App Locally
1. Start the app:
```bash
python -m model.app
```
2. Open browser: `http://127.0.0.1:5000`
3. Fill the form and submit. The UI shows both EUR prediction and converted LKR value (uses exchangerate.host with fallback rate).

## How Inference Works
- `model/app.py` loads `model/predictor.pickle` at runtime (module-relative path).
- `predictor.pickle` is a sklearn `Pipeline` with named steps `pre` (ColumnTransformer) and `model` (estimator).
- The app converts form inputs into a dict with keys:
  `['ram','weight','touchscreen','ips','company','typename','opsys','cpuname','gpuname']`
  and the pipeline expects these column names for transformation.
- Currency: the model predicts prices in the currency used during training (your dataset values). The app assumes EUR and converts to LKR using an external API with a fallback constant.

## Retraining the Model
- Use `model/train_pipeline.py` to retrain and overwrite `model/predictor.pickle`.
- Example:
```bash
python -m model.train_pipeline --data data/laptops.csv --out model/predictor.pickle
```
- The script expects a CSV with at least the columns:
  `ram,weight,company,typename,opsys,cpuname,gpuname,price` (touchscreen, ips optional).
- The script normalizes common column formats (e.g., `8GB` → `8`, `1.37kg` → `1.37`) and canonicalizes CPU/GPU/OS labels.
- After training the pipeline, the saved `predictor.pickle` will contain preprocessing and the regressor; the app loads and uses it without extra code changes.

## Testing & Evaluation
- Quick evaluation: you can run a small Python snippet to load `predictor.pickle` and call `.predict()` on sample rows (I can run that for you).
- To compute RMSE on a CSV split, run the `train_pipeline` script which prints validation RMSE.

## Troubleshooting
- Missing dependencies: If app returns errors about `sklearn` or `pandas`, install them in the venv.
- FileNotFoundError: ensure `model/predictor.pickle` exists in `model/` (the app loads it relative to the `model` folder).
- Feature mismatch errors: make sure input dict keys match the pipeline's expected column names.
- Currency issues: If external API fails or is blocked, the app falls back to a default rate inside `model/app.py` — update `get_eur_to_lkr_rate()` if you want a different default.
- Sklearn/unpickle version mismatch: if the environment sklearn version differs from the one used to save the pipeline, unpickling may fail. In that case, retrain on the current environment or pin sklearn versions.

## Logging & Debugging
- App uses `app.logger` for errors — check console output where Flask runs.
- `model/train_pipeline.py` uses `logging` to report training progress and RMSE.

## Maintenance notes
- Keep `model/predictor.pickle` under version control only if you want a snapshot in the repository; otherwise add it to `.gitignore` and provide instructions to retrain.
- If you share the repo, include a `requirements.txt` or `pyproject.toml` that pins versions to ensure reproducible unpickling.

## Suggested Enhancements
- Add unit tests for `prediction()` and for `train_pipeline` data-cleaning helpers.
- Add a small script to batch-evaluate the pipeline against a CSV and output per-row diffs.
- Add a small health-check endpoint that returns model load status and current EUR→LKR rate.
- Replace exchangerate.host with a paid/robust service if high availability is required.

## Contact / Authors
- Add maintainer contact information here if the team wants it.
- Shamal Sathsara 
   - shamalsathsara4@gmail.com
   - 077 158 1916

---
*End of developer manual.*
