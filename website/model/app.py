"""
Web front-end for the laptop price predictor.

This small Flask app loads a saved sklearn Pipeline (preprocessing + model)
from predictor.pickle and exposes a single page where users can submit
specs and get a predicted price.
"""

import json
import os
import pickle
import urllib.request
from urllib.error import URLError, HTTPError

import pandas as pd
from flask import Flask, render_template, request

# Set up our Flask application, pointing to the correct template and static folders
app = Flask(__name__, template_folder="../templates", static_folder="../static")

def load_model_pipeline():
    """
    Helper function to load the saved sklearn pipeline from disk.
    """
    filename = os.path.join(os.path.dirname(__file__), 'predictor.pickle')
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        app.logger.error("Model file not found: %s", filename)
        return None
    except pickle.UnpicklingError as err:
        app.logger.error("Failed to unpickle model: %s", err)
        return None


# Load the model once when the app starts so we don't reload it on every request
PREDICTOR_MODEL = load_model_pipeline()


def prediction(input_row):
    """
    Generate a price prediction using the loaded model.
    """
    if PREDICTOR_MODEL is None:
        return {'ok': False, 'error': "The prediction model could not be loaded."}

    try:
        # Convert the dictionary to a pandas DataFrame for the pipeline
        dataframe = pd.DataFrame([input_row])
        pred = PREDICTOR_MODEL.predict(dataframe)
        return {'ok': True, 'value': float(pred[0])}
    except ValueError as err:
        return {'ok': False, 'error': f"Prediction error (Value): {err}"}
    except TypeError as err:
        return {'ok': False, 'error': f"Prediction error (Type): {err}"}


def get_eur_to_lkr_rate(timeout=5):
    """
    Fetch current EUR to LKR exchange rate from exchangerate.host.
    Falls back to a safe default if the network fetch fails.
    """
    default_rate = 415.0  # Fallback rate if the API goes down
    url = 'https://api.exchangerate.host/latest?base=EUR&symbols=LKR'
    
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
            rate = data.get('rates', {}).get('LKR')
            if rate:
                return float(rate)
            
            app.logger.warning('EUR to LKR rate missing in response; using default')
            return default_rate
    except (URLError, HTTPError, ValueError, json.JSONDecodeError) as err:
        app.logger.warning('Failed to fetch EUR to LKR rate (%s); using default %.2f', 
                           err, default_rate)
        return default_rate


def canonicalize_form_data(form_data):
    """
    Extract and clean data from the submitted form, mapping it to the 
    format expected by the machine learning model.
    """
    # Extract raw string values from the form
    company = form_data.get('company')
    typename = form_data.get('typename')
    opsys = form_data.get('opsys')
    cpu = form_data.get('cpuname')
    gpu = form_data.get('gpuname')
    
    # Checkboxes are omitted if unchecked; we convert them to 1 or 0
    touchscreen_flag = 1 if form_data.get('touchscreen') else 0
    ips_flag = 1 if form_data.get('ips') else 0

    # Ensure company name matches our training labels (case-insensitive)
    company_list = ['Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Acer', 'Toshiba', 
                    'Samsung', 'Razer']
    company_canon = company
    if company:
        for item in company_list:
            if item.lower() == company.lower():
                company_canon = item
                break

    # Map UI dropdown values to the canonical training labels
    typename_map = {
        '2in1convertible': '2 in 1 Convertible', 'gaming': 'Gaming', 'netbook': 'Net Book',
        'notebook': 'Notebook', 'ultrabook': 'Ultrabook', 'workstation': 'Workstation'
    }
    typename_canon = typename_map.get(typename, typename)

    opsys_map = {'windows': 'Windows', 'mac': 'Mac', 'linux': 'Linux', 'other': 'Other'}
    opsys_canon = opsys_map.get(opsys, opsys)

    cpuname_map = {
        'intelcorei3': 'Intel Core i3', 'intelcorei5': 'Intel Core i5', 
        'intelcorei7': 'Intel Core i7', 'amd': 'AMD Processor', 'other': 'Other'
    }
    cpu_canon = cpuname_map.get(cpu, cpu)

    gpuname_map = {'intel': 'intel', 'amd': 'amd', 'nvidia': 'NVIDIA GeForce'}
    gpu_canon = gpuname_map.get(gpu, gpu)

    return {
        'touchscreen': touchscreen_flag,
        'ips': ips_flag,
        'company': company_canon,
        'typename': typename_canon,
        'opsys': opsys_canon,
        'cpuname': cpu_canon,
        'gpuname': gpu_canon,
    }


def validate_numeric_inputs(form_data):
    """
    Validate that RAM and Weight are present and properly formatted.
    Raises ValueError if anything is missing or invalid.
    """
    ram_str = form_data.get('ram')
    weight_str = form_data.get('weight')

    if ram_str is None or str(ram_str).strip() == '':
        raise ValueError('RAM value is required')
    
    if weight_str is None or str(weight_str).strip() == '':
        raise ValueError('Weight value is required')

    return int(ram_str), float(weight_str)


def get_css_version():
    """
    Compute a CSS version based on file modification time to bust browser caches.
    """
    try:
        css_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'style.css')
        return int(os.path.getmtime(css_path))
    except OSError:
        return 0


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Render the main page and handle form submissions.
    """
    css_version = get_css_version()
    rate = get_eur_to_lkr_rate()

    if request.method == 'POST':
        # 1. Validate numeric inputs first
        try:
            ram_val, weight_val = validate_numeric_inputs(request.form)
        except ValueError as err:
            app.logger.error('Invalid input: %s', err)
            return render_template('index.html', pred_value=0, pred_value_lkr=0, 
                                   error=str(err), css_version=css_version, eur_to_lkr_rate=rate)

        # 2. Extract and format categorical data
        input_row = canonicalize_form_data(request.form)
        input_row['ram'] = ram_val
        input_row['weight'] = weight_val

        # 3. Predict!
        result = prediction(input_row)
        if result.get('ok'):
            pred_value_eur = float(result.get('value'))
            pred_value_lkr = pred_value_eur * rate
            
            return render_template('index.html', pred_value=pred_value_eur, 
                                   pred_value_lkr=pred_value_lkr, eur_to_lkr_rate=rate, 
                                   css_version=css_version)
        
        # If prediction fails, show the error
        error_msg = result.get('error')
        app.logger.error(error_msg)
        return render_template('index.html', pred_value=0, pred_value_lkr=0, 
                               error=error_msg, css_version=css_version, eur_to_lkr_rate=rate)

    # For a simple GET request, just show the empty form
    return render_template('index.html', pred_value=0, pred_value_lkr=0, 
                           eur_to_lkr_rate=rate, css_version=css_version)


if __name__ == '__main__':
    app.run(debug=True)