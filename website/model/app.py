"""
Web front-end for the laptop price predictor.

This small Flask app loads a saved sklearn `Pipeline` (preprocessing + model)
from `predictor.pickle` and exposes a single page where users can submit
specs and get a predicted price. The app expects the pipeline to accept a
pandas DataFrame row with columns:
    ['ram','weight','touchscreen','ips','company','typename','opsys','cpuname','gpuname']

Notes for maintainers:
- Keep inference code minimal and pass raw form values as a single dict
    to the pipeline — the `train_pipeline` defines canonical column names.
- Errors while loading or predicting are returned to the template and
    logged via `app.logger` for easier debugging.
"""

from flask import Flask, render_template  # type: ignore[import]
from flask import request  # type: ignore[import]
import pickle
import os
import json
import urllib.request
from urllib.error import URLError, HTTPError

# Application instance: templates and static folders are relative to the
# repository layout where `model/` contains this file and sibling `../templates`.
app = Flask(__name__, template_folder="../templates", static_folder="../static")

def prediction(input_row):
    """Load the saved pipeline and return a prediction for `input_row`.

    Parameters
    - input_row: dict or list
        If dict: keys should match the training columns (see module docstring).
        If list: treated as a raw feature vector and passed directly to `predict`.

    Returns
    - dict: {'ok': bool, 'value': float} on success or {'ok': False, 'error': str}
    """

    # Quick dependency checks with helpful messages to the caller.
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return {
            'ok': False,
            'error': "Missing dependency: scikit-learn not installed. Run `pip install scikit-learn`."
        }

    # Resolve the stored pipeline file next to this module.
    filename = os.path.join(os.path.dirname(__file__), 'predictor.pickle')
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return {'ok': False, 'error': f"Model file not found: {filename}"}
    except Exception as e:
        return {'ok': False, 'error': f"Failed to load model: {e}"}

    # The saved pipeline expects pandas DataFrame rows — ensure pandas is present.
    try:
        import pandas as pd
    except Exception:
        return {'ok': False, 'error': 'Missing dependency: pandas required. Run `pip install pandas`.'}

    # If caller provided a dict, convert to single-row DataFrame for the pipeline.
    if isinstance(input_row, dict):
        df = pd.DataFrame([input_row])
        try:
            pred = model.predict(df)
            return {'ok': True, 'value': float(pred[0])}
        except Exception as e:
            return {'ok': False, 'error': f"Prediction error: {e}"}

    # Otherwise, assume the caller provided a raw feature vector and try directly.
    try:
        pred = model.predict([input_row])
        return {'ok': True, 'value': float(pred[0])}
    except Exception as e:
        return {'ok': False, 'error': f"Prediction error: {e}"}


def get_eur_to_lkr_rate(timeout=5):
    """Fetch current EUR->LKR rate from exchangerate.host with a safe fallback.

    Returns a float exchange rate (LKR per 1 EUR). If network fetch fails,
    returns a sensible default and logs the issue via `app.logger`.
    """
    DEFAULT_RATE = 415.0  # fallback rate; update if stale
    url = 'https://api.exchangerate.host/latest?base=EUR&symbols=LKR'
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
            rate = data.get('rates', {}).get('LKR')
            if rate:
                return float(rate)
            else:
                app.logger.warning('EUR->LKR rate missing in response; using default')
                return DEFAULT_RATE
    except (URLError, HTTPError, ValueError, json.JSONDecodeError) as e:
        app.logger.warning('Failed to fetch EUR->LKR rate (%s); using default %.2f', e, DEFAULT_RATE)
        return DEFAULT_RATE



@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the index page and handle form submissions.

    The HTML form posts raw string values; this handler canonicalizes
    those values into the same format used during training and then
    delegates to `prediction()`.
    """

    # compute CSS version based on static file mtime to bust browser cache
    try:
        css_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'style.css')
        css_version = int(os.path.getmtime(css_path))
    except Exception:
        css_version = 0

    if request.method == 'POST':
        ram = request.form.get('ram')
        weight = request.form.get('weight')
        company = request.form.get('company')
        typename = request.form.get('typename')
        opsys = request.form.get('opsys')
        cpu = request.form.get('cpuname')
        gpu = request.form.get('gpuname')
        touchscreen = request.form.get('touchscreen')
        ips = request.form.get('ips')

        # checkboxes are omitted from form when unchecked; convert to 0/1
        touchscreen_flag = 1 if touchscreen else 0
        ips_flag = 1 if ips else 0

        # Build a dict matching training pipeline input columns:
        # ['ram','weight','touchscreen','ips','company','typename','opsys','cpuname','gpuname']
        company_list = ['Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Acer', 'Toshiba', 'Samsung', 'Razer']
        typename_list = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation']

        # canonicalize company (case-insensitive match to training labels)
        company_canon = None
        if company:
            for item in company_list:
                if item.lower() == company.lower():
                    company_canon = item
                    break
        if company_canon is None:
            company_canon = company

        # map template form values to canonical labels where applicable
        typename_map = {
            '2in1convertible': '2 in 1 Convertible', 'gaming': 'Gaming', 'netbook': 'Net Book',
            'notebook': 'Notebook', 'ultrabook': 'Ultrabook', 'workstation': 'Workstation'
        }
        typename_canon = typename_map.get(typename, typename)

        opsys_map = {'windows': 'Windows', 'mac': 'Mac', 'linux': 'Linux', 'other': 'Other'}
        opsys_canon = opsys_map.get(opsys, opsys)

        cpuname_map = {
            'intelcorei3': 'Intel Core i3', 'intelcorei5': 'Intel Core i5', 'intelcorei7': 'Intel Core i7',
            'amd': 'AMD Processor', 'other': 'Other'
        }
        cpu_canon = cpuname_map.get(cpu, cpu)

        gpuname_map = {'intel': 'intel', 'amd': 'amd', 'nvidia': 'NVIDIA GeForce'}
        gpu_canon = gpuname_map.get(gpu, gpu)

        # Validate and convert numeric inputs. Return a friendly error if parsing fails.
        try:
            if ram is None or str(ram).strip() == '':
                # Log diagnostics to help find out why RAM was missing from the form
                try:
                    form_dict = dict(request.form)
                    values_dict = dict(request.values)
                    raw_body = request.get_data(as_text=True)[:1000]
                except Exception:
                    form_dict = '<unavailable>'
                    values_dict = '<unavailable>'
                    raw_body = '<unavailable>'
                app.logger.error('Missing RAM in submitted form. form=%s values=%s body_start=%s client=%s',
                                 form_dict, values_dict, raw_body, request.remote_addr)
                raise ValueError('RAM value is required')
            ram_val = int(ram)
        except (TypeError, ValueError) as e:
            error = f'Invalid RAM value: {e}'
            app.logger.error(error)
            return render_template('index.html', pred_value=0, error=error, css_version=css_version)

        try:
            if weight is None or str(weight).strip() == '':
                raise ValueError('Weight value is required')
            weight_val = float(weight)
        except (TypeError, ValueError) as e:
            error = f'Invalid weight value: {e}'
            app.logger.error(error)
            return render_template('index.html', pred_value=0, error=error)

        input_row = {
            'ram': ram_val,
            'weight': weight_val,
            'touchscreen': touchscreen_flag,
            'ips': ips_flag,
            'company': company_canon,
            'typename': typename_canon,
            'opsys': opsys_canon,
            'cpuname': cpu_canon,
            'gpuname': gpu_canon,
        }

        result = prediction(input_row)
        if result.get('ok'):
            pred_value_eur = float(result.get('value'))
            # Convert to LKR using live rate with fallback
            rate = get_eur_to_lkr_rate()
            pred_value_lkr = pred_value_eur * rate
            # Keep three decimal places for display
            return render_template('index.html', pred_value=pred_value_eur, pred_value_lkr=pred_value_lkr, eur_to_lkr_rate=rate, css_version=css_version)
        else:
            error = result.get('error')
            # log and show error to user
            app.logger.error(error)
            return render_template('index.html', pred_value=0, error=error, css_version=css_version)


        
    
    # On GET, show empty values and include the current approximate rate
    rate = get_eur_to_lkr_rate()
    return render_template('index.html', pred_value=0, pred_value_lkr=0, eur_to_lkr_rate=rate, css_version=css_version)

if __name__ == '__main__':
    app.run(debug=True)