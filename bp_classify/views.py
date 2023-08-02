from flask import Blueprint, request, jsonify
from predict.predict import get_prediction

bp_classify = Blueprint('bp_classify', __name__)

@bp_classify.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    result = get_prediction(data)
    return jsonify(result)
