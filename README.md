#  Laptop Price Predictor ML

An end-to-end Machine Learning web application that predicts laptop prices based on hardware specifications.

##  Overview

The **Laptop Price Predictor** bridges the gap between Machine Learning and Full-Stack Web Development. By leveraging a custom-trained **Random Forest** model, this application takes user inputs (like RAM, CPU, GPU, brand, and weight) and instantly generates an accurate price prediction.

The prediction is displayed in a sleek, modern UI with a live currency conversion backend that dynamically converts the base model predictions into localized pricing (LKR).

##  Features

- **Robust ML Pipeline**: Built with `scikit-learn`, the model utilizes a `ColumnTransformer` to handle dirty data and one-hot encoding before passing it to a highly accurate `RandomForestRegressor`.
- **Flask Backend**: A lightweight, modular REST backend that serves the model, validates form data, and integrates with external exchange rate APIs.
- **Modern "Glassmorphism" UI**: A beautiful, responsive frontend built with vanilla HTML/CSS featuring a vibrant animated background, clean micro-animations, and dynamic count-up effects.
- **Clean Architecture**: The codebase strictly follows PEP-8 standards, modular design patterns, and includes thorough developer documentation.

##  Tech Stack

- **Machine Learning**: Python, `scikit-learn`, Pandas, NumPy
- **Backend**: Flask (Python), REST APIs
- **Frontend**: HTML5, CSS3 (Glassmorphism design), Vanilla JavaScript

##  Repository Structure

```text
├── model/                     # Jupyter notebooks & raw data exploration
├── website/                   # Full-stack web application
│   ├── model/                 # Backend ML scripts & saved pipeline
│   │   ├── app.py             # Main Flask application
│   │   ├── train_pipeline.py  # Data cleaning and model training script
│   │   └── predictor.pickle   # Serialized ML Pipeline
│   ├── static/                # CSS and frontend assets
│   ├── templates/             # HTML templates
│   └── DEVELOPER.md           # Extensive backend/developer documentation
└── README.md                  # Project overview
```

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/ShamalSathsara/Price-Predicter-ML.git
cd Price-Predicter-ML/website
```

### 2. Set up a virtual environment
```bash
python -m venv env
# Windows
env\Scripts\activate
# macOS/Linux
source env/bin/activate
```

### 3. Install dependencies
```bash
pip install flask pandas scikit-learn numpy
```

### 4. Run the application
```bash
python -m model.app
```
Open your browser and navigate to `http://127.0.0.1:5000` to start predicting!

##  Developer Guide
If you want to debug, retrain the model with new data, or extend the application, please check out the comprehensive [Developer Manual](website/DEVELOPER.md).

##  Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

##  Contact
**Shamal Sathsara**
- Email: [EMAIL_ADDRESS]
- LinkedIn: [www.linkedin.com/in/shamal-sathsara-93a245267]
- GitHub: [https://github.com/ShamalSathsara]