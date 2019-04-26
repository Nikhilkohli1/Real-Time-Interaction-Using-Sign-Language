# USAGE
# Start the server:
# 	python predict.py
# Submit a request via cURL:
# 	curl -X POST -F image=@gesture.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python predict_image.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model, model_from_json
from PIL import Image
import numpy as np
import flask
import io

MODEL_WEIGHTS_PATH_H5="model_BW.h5"
MODEL_PATH_JSON="model_BW.json"
gesture_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','OK GOOGLE','P','Q','R','S','STOP','T','U','V','W','X','Y','Z']
print("len:",len(gesture_list))
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
def load_new_model():
    # load json and create model
    json_file = open(MODEL_PATH_JSON, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global model
#     model = None
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(MODEL_WEIGHTS_PATH_H5)
    print("Loaded model from disk")
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model._make_predict_function()
#     return loaded_model

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()

			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			# 
			
			probs = model.predict(image)
			preds = gesture_list[list(probs[0]).index(max(list(probs[0])))]
			data["predictions"] = []
			r = {"label": preds}
			data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_new_model()
	app.run()