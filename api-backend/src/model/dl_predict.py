import io
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

model = load_model('model/model.h5')

def get_remedy(plant_disease):
	with open("model/disease.json", 'r') as f:
		remedies = json.load(f)

	for key in remedies:
		if key == plant_disease:
			return(remedies[key])


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

	# confidence = round(100 * j, 3)
	# print(disease, confidence)

	if "healthy" not in disease:
		try:
			remedy = get_remedy(disease)
		except:
			remedy = "Sorry the remedy is not available"
	else:
		remedy = "NAN, the plant is healthy"
	
	return disease, remedy 

