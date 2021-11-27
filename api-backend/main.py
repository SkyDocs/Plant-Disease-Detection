from flask import Flask, jsonify, request
from flask_cors import CORS

from model.dl_predict import predict_disease, neuralNet

import base64

app = Flask("Plant Disease Detection")

CORS(app)

@app.route('/', methods=['POST'])

def predict():
	data_ret = request.get_json()
	image = data_ret["image"]
	image = base64.b64decode(image)
	#predict
	model = neuralNet()
	result = predict_disease(model, image)

	response = {
		result
	}

	response = jsonify(result)
	return response


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=8080)