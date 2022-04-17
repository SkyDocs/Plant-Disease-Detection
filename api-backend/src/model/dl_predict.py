import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import io
import json
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_names = ['Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___healthy',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Late_blight',
 'Potato___healthy',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Septoria_leaf_spot',
 'Tomato___healthy']

model = load_model('model/model.h5')

def get_remedy(plant_disease):
	print("get_remedy is trigerred")
	f = open('model/disease.json')
	data = json.load(f)

	for i in data:
		print(i)
		if(plant_disease == i):
			return(data[i])

	f.close()
	return


def predict_disease(image):
	img = Image.open(io.BytesIO(image))
	img = img.convert('RGB')
	img = img.resize((256, 256), Image.NEAREST)

	# img = image.load_img(image, target_size=(256, 256))
	#x = image.img_to_array(img)
	x = np.asarray(img)
	x = np.expand_dims(x, axis=0)
	pred = model.predict(x)

	# print("prediction", pred)
	
	d = pred.flatten()
	j = d.max()
	for index,item in enumerate(d):
		if item == j:
			disease = class_names[index]

	confidence = round(100 * j, 3)
	print(disease, confidence)

	if "healthy" not in disease:
		try:
			remedy = get_remedy(disease)
		except:
			remedy = "Sorry the remedy is not available"
	else:
		remedy = "NAN, the plant is healthy"
	
	return disease, remedy 

