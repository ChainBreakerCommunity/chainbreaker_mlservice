from flask import Flask, jsonify, send_file, request, url_for, render_template
from flask_cors import CORS
from utils.env import get_config
from routes.model import model
conf = get_config()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.register_blueprint(model, url_prefix = "/model")

@app.route('/')
def root():
    return render_template('index.html')

@app.route("/status", methods = ["GET"])
def status():
    return jsonify({'status' : 200})

@app.route("/get_logging", methods=["POST"])
def get_logging():
    data = request.values
    return send_file("./logger/{route}.log".format(route = data["route"])), 200

if __name__ == "__main__":
    debug = (conf["DEBUG"] == "TRUE")
    app.run(host = "0.0.0.0", port = conf["PORT"], debug = debug) #bool(conf_debug["DEBUG"]))
