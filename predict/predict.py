from io import BytesIO
from PIL import Image
import numpy as np
import requests, base64,os

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the pre-trained model
model = ResNet50(weights='imagenet')

def get_prediction(data):
    if 'url' in data:
        url = data['url']

        # Download the image
        response = requests.get(url)
        img_path = 'downloaded_image.jpg'
        with open(img_path, 'wb') as f:
            f.write(response.content)

        # Load the image file, resizing it to 224x224 pixels (required by this model)
        img = image.load_img(img_path, target_size=(224, 224))

        # Cleanup
        os.remove(img_path)
    elif 'base64' in data:
        base64_str = data['base64']
        base64_img_bytes = base64.b64decode(base64_str)
        image_data = BytesIO(base64_img_bytes)

        # Load the image data, resizing it to 224x224 pixels (required by this model)
        img = Image.open(image_data)
        img = img.resize((224, 224))

    # Convert the image to a numpy array
    x = image.img_to_array(img)

    # Add a forth dimension since Keras expects a list of images
    x = np.expand_dims(x, axis=0)

    # Preprocess the input image
    x = preprocess_input(x)

    # Make a prediction
    predictions = model.predict(x)

    # Convert the probabilities to class labels
    predicted_classes = decode_predictions(predictions, top=3)

    # Select the prediction with the highest probability
    best_prediction = max(predicted_classes[0], key=lambda x: x[2])

    result = {
        "Prediction": best_prediction[1],
        "Probability": round(float(best_prediction[2]), 2)
    }

    return result
