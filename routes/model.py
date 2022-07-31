from concurrent.futures import thread
from flask import Blueprint, request, jsonify, render_template
from model.train import train_model
from utils.inference import batch_classification
from utils.env import get_config
import threading
config = get_config()

model = Blueprint("data", __name__)

@model.route('/', methods = ["GET"])
def index():
    return jsonify({"message": "DATA ROUTE"})

@model.route("/eda", methods = ["GET"])
def eda():
    """Return Exploratory Data Analysis page.
    """
    return render_template("eda.html")

@model.route("/retrain_model", methods = ["GET"])
def retrain_model():
    """Retrain model (AUTOML)
    """
    train_model()
    return render_template("automl.html")

@model.route("/classify_unlabel_ads", methods = ["GET"])
def classify_unlabel_ads():
    thread = threading.Thread(target = batch_classification, args=("en", ))
    thread.start()
    #batch_classification(language = "en")
    return jsonify({"message": "ok"})
