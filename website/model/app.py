from flask import Flask, render_template  # type: ignore[import]
from flask import request  # type: ignore[import]
import pickle
import os
app = Flask(__name__, template_folder="../templates", static_folder="../static")

def prediction(input_row):
    # ensure scikit-learn is available before unpickling
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return {
            'ok': False,
            'error': "Missing dependency: scikit-learn not installed. Run `pip install scikit-learn` in your environment."
        }

    # resolve predictor.pickle relative to this module's directory
    filename = os.path.join(os.path.dirname(__file__), 'predictor.pickle')
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return {'ok': False, 'error': f"Model file not found: {filename}"}
    except Exception as e:
        return {'ok': False, 'error': f"Failed to load model: {e}"}

    # If the loaded object is a Pipeline that accepts raw DataFrame input, use it directly.
    try:
        import pandas as pd
    except Exception:
        return {'ok': False, 'error': 'Missing dependency: pandas required for preprocessing. Run `pip install pandas`.'}

    # Accept either dict-like input_row or pre-built feature list
    if isinstance(input_row, dict):
        df = pd.DataFrame([input_row])
        try:
            pred = model.predict(df)
            return {'ok': True, 'value': float(pred[0])}
        except Exception as e:
            return {'ok': False, 'error': f"Prediction error: {e}"}
    else:
        # fallback: if the caller passed a raw feature list, try to predict directly
        try:
            pred = model.predict([input_row])
            return {'ok': True, 'value': float(pred[0])}
        except Exception as e:
            return {'ok': False, 'error': f"Prediction error: {e}"}



@app.route('/', methods=['GET', 'POST'])
def index():

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

        input_row = {
            'ram': int(ram),
            'weight': float(weight),
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
            pred_value = result.get('value')
            return render_template('index.html', pred_value=pred_value)
        else:
            error = result.get('error')
            # log and show error to user
            print('Prediction error:', error)
            return render_template('index.html', pred_value=0, error=error)


        
    
    return render_template('index.html', pred_value=0)

if __name__ == '__main__':
    app.run(debug=True)