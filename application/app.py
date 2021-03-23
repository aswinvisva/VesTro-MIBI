from flask import Flask, render_template, request, redirect, url_for

from config.config_settings import Config
from mibipy.mibi_pipeline import MIBIPipeline

app = Flask(__name__)


# home route
@app.route("/")
def home():
    return render_template('index.html')


@app.route('/analyze_landing', methods=['GET'])
def analyze_landing():
    return render_template('analysis.html')


@app.route('/api/v1/analyze', methods=['POST'])
def analyze():
    conf = Config()

    data_dir = request.form['data_dir']
    mask_dir = request.form['mask_dir']
    output_dir = request.form['output_dir']

    if len(request.form['interval']) > 0:
        interval = float(request.form['interval'])
    else:
        interval = None

    if len(request.form['inward_distance']) > 0:
        inward_distance = float(request.form['inward_distance'])
    else:
        inward_distance = None

    if len(request.form['outward_distance']) > 0:
        outward_distance = float(request.form['outward_distance'])
    else:
        outward_distance = None

    if data_dir is not None and len(data_dir) > 0:
        conf.data_dir = data_dir

    if mask_dir is not None and len(mask_dir) > 0:
        conf.masks_dir = mask_dir

    if output_dir is not None and len(output_dir) > 0:
        conf.visualization_results_dir = output_dir

    if interval is not None:
        conf.distance_interval = interval

    if inward_distance is not None:
        conf.max_inward_expansion = int(inward_distance / conf.distance_interval)

    if outward_distance is not None:
        conf.max_expansions = int(outward_distance / conf.distance_interval)

    try:
        pipe = MIBIPipeline(conf)
        pipe.load_preprocess_data()
        pipe.generate_visualizations()
    except FileNotFoundError:
        print("Could not find file!")

    return redirect(url_for('app.functionB', visualizer=pipe))


if __name__ == '__main__':
    app.run()
