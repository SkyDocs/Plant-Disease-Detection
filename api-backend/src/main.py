from flask import Flask, jsonify, request
from flask_cors import CORS

from model.dl_predict import predict_disease, neuralNet

import base64

app = Flask("Plant Disease Detection")
app.config['JSON_SORT_KEYS'] = False

CORS(app)

@app.route('/', methods=['GET'])
def main():
	return "Plant Disease Detection by Team SkyDocs, https://github.com/SkyDocs/Plant-Disease-Detection"

@app.route('/', methods=['POST'])

def predict():
	data_ret = request.get_json()
	image = data_ret["image"]
	image = base64.b64decode(image)
	#predict
	model = neuralNet()
	result = predict_disease(model, image)

	plant = result.split("___")[0]
	plant_disease = " ".join((result.split("___")[1]).split("_"))
	remedy = " ".join((result,split("___")[2]).split("_"))

	response = {
		"Plant" : plant,
		"Disease" : plant_disease,
		"remedy" : remedy
	}

	response = jsonify(response)
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080)