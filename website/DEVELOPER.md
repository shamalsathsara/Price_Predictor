#  Developer Manual — Price Predictor ML

Welcome to the Developer Manual for the **Laptop Price Predictor** project. This guide provides all the necessary information to run, debug, retrain, and extend the application.

##  Project Overview
- **Purpose:** A clean, Flask-based web application that predicts laptop prices using a pre-trained scikit-learn `Pipeline` (`model/predictor.pickle`).
- **Web UI:** Modern, responsive light theme built with `templates/index.html` and `static/style.css`. The UI handles user input and dynamically animates the predicted price.
- **Backend Entry (`model/app.py`):** Loads the ML model, handles form submissions, validates inputs, and performs background currency conversion (EUR → LKR). The UI exclusively presents the final LKR price to the user.
- **Training Utility (`model/train_pipeline.py`):** A robust script to clean raw CSV data, build a `ColumnTransformer` pipeline, train a `RandomForestRegressor`, and save the complete artifact.

##  Repository Layout
- `model/app.py` — The core Flask application and inference wrapper.
- `model/train_pipeline.py` — The data cleaning and model training script.
- `model/predictor.pickle` — The serialized scikit-learn pipeline (preprocessing + model).
- `templates/index.html` — The frontend HTML interface.
- `static/style.css` — The modern UI styling.
- `data/` — The expected directory for your training dataset (`laptops.csv`).
- `DEVELOPER.md` — This documentation file.

##  Development Setup
1. **Create and activate a Python virtual environment** (Highly Recommended):
   ```bash
   python -m venv env
   
   # Windows
   env\Scripts\activate
   
   # Unix/macOS
   source env/bin/activate
   ```
2. **Install dependencies**:
   Ensure you have the required packages installed. At a minimum, you will need:
   ```bash
   pip install flask pandas scikit-learn numpy
   ```
3. **Verify Python Path**: Ensure your IDE or terminal points to the activated `env`.

##  Running the App Locally
1. **Start the Flask server**:
   ```bash
   python -m model.app
   ```
2. **Access the Web UI**: Open your browser and navigate to `http://127.0.0.1:5000`.
3. **Usage**: Fill out the laptop specifications and submit. The app will validate the inputs, run the ML pipeline, perform a live currency conversion to LKR, and display the final price with a sleek UI animation.

##  How Inference Works
- The application automatically loads `model/predictor.pickle` on startup to optimize response times.
- The form inputs are canonicalized via robust helper functions to match the exact format the model expects:
  `['ram', 'weight', 'touchscreen', 'ips', 'company', 'typename', 'opsys', 'cpuname', 'gpuname']`
- **Currency Conversion:** The model natively predicts prices in EUR (based on the training data). `app.py` fetches the live exchange rate from an external API (falling back to a static constant if offline) and calculates the final LKR price to display.

##  Retraining the Model
When your dataset updates, you can easily regenerate the model artifact:

1. Place your updated CSV in the `data/` folder.
2. Run the training script:
   ```bash
   python -m model.train_pipeline --data data/laptops.csv --out model/predictor.pickle
   ```
3. **What happens?** The script will automatically normalize common dirty data formats (e.g., parsing `8GB` to `8`, canonicalizing CPU labels), split the data, evaluate the model's RMSE, and save the new pipeline ready for immediate use by the Flask app.

##  Troubleshooting & Debugging
- **Missing Dependencies:** If the app crashes on startup, ensure `scikit-learn` and `pandas` are installed in your active virtual environment.
- **Model Not Found:** Ensure `predictor.pickle` is located inside the `model/` directory.
- **Form Submission Errors:** If inputs fail validation, `app.logger` will output the exact reason to the console, and a user-friendly error message will display on the UI.
- **Currency API Failure:** If the exchange rate API is down, the app safely defaults to a hardcoded rate inside `app.py`. Update `get_eur_to_lkr_rate()` if you need to adjust this default.
- **Version Mismatches:** If you encounter `UnpicklingError`s, it usually means your local `scikit-learn` version is significantly different from the one used to train the model. You can either align your environment versions or simply retrain the model locally.

##  Maintenance & Next Steps
- **Code Quality:** The backend has been thoroughly refactored with modular helper functions, strict `snake_case` variable naming, and extensive human-readable comments. 
- **Future Enhancements:** 
  - Add unit tests for the data canonicalization logic.
  - Implement a caching layer for the exchange rate API to prevent hitting rate limits.

##  Contact / Author
- **Shamal Sathsara**
  -  shamalsathsara4@gmail.com
  -  077 158 1916
