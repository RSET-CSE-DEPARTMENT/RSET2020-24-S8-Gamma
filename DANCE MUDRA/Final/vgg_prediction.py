import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Constants
num_classes = 25  # Assuming you have 50 classes
img_size = (300, 300)
batch_size = 32
save_dir = r'C:\Users\menon\Desktop\RSET STUFFS\PROJECT\S7-BHARATHANATYAM\Final\model'

def predict_with_loaded_model(uploaded_file, model_path):
    # Load the saved model
    loaded_model = load_model(model_path)
    #print(loaded_model.summary())
    
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=img_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    
    # Perform inference
    prediction = loaded_model.predict(img)
    
    return prediction.tolist()

# Example usage:
image_file = r'C:\Users\menon\Desktop\RSET STUFFS\PROJECT\S7-BHARATHANATYAM\Bharatanatyam-Mudra-Dataset-master\Testing\Anjali\Anjali_2.jpg'  # Replace '/path/to/your/image.jpg' with the actual path to your image file
model_path = r'C:\Users\menon\Desktop\RSET STUFFS\PROJECT\S7-BHARATHANATYAM\Final\model\subset_model_0.h5'    # Replace '/path/to/your/model.h5' with the actual path to your h5 file

prediction = predict_with_loaded_model(image_file, model_path)
print(prediction)
