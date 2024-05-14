from flask import Flask, render_template,request,redirect,url_for,jsonify,send_from_directory, session
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import csv
import numpy as py
import os
# import correction
app = Flask(__name__)

# Initialize YOLO model
yolo = YOLO('best.pt')
# yolo.info()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploadimages')
def upload_images():
    return render_template('uploadimages.html')

@app.route('/viewmudra')
def view_mudra():
    return render_template('viewmudra.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/result_details')
def predict_result():
    mudra_name = request.args.get('mudra')
    return render_template('result_details.html', mudra_name=mudra_name)


@app.route('/mudra_details')
def mudra_details():
    mudra_name = request.args.get('mudra')
    return render_template('mudra_details.html', mudra_name=mudra_name)

@app.route('/submit', methods=['POST'])
def submit_feedback():
    # Get the form data from the request
    name = request.form['name']
    email = request.form['email']
    rating = request.form['rating']
    message = request.form['message']

    # Create a dictionary to hold the feedback data
    feedback_data = {
        'Name': name,
        'Email': email,
        'Rating': rating,
        'Message': message
    }

    # Define the path to the CSV file
    csv_file_path = 'feedback_data.csv'

    # Write the feedback data to the CSV file
    with open(csv_file_path, mode='a', newline='') as csv_file:
        fieldnames = ['Name', 'Email', 'Rating', 'Message']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Check if the CSV file is empty and write the header if needed
        if csv_file.tell() == 0:
            writer.writeheader()

        # Write the feedback data to the CSV file
        writer.writerow(feedback_data)

    return redirect(url_for('success'))


@app.route('/success')
def success():
    return render_template('success.html')

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Load the uploaded image
    image = Image.open(file)
    
    # Perform object detection using YOLO
    results = yolo.predict(image, save_txt=True)
    names=yolo.names

    # with open("/content/ABC.txt", 'w') as file:
    #  for prediction in results.xyxy[0]:
    #     file.write(f"{prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")
    # cname=yolo(image)

    # Save annotated images
    annotated_image_files = []
    for idx, result in enumerate(results):
        im_array = result.plot()  # Plot annotated image
        #predictedMudra=result.names()
        annotated_image = Image.fromarray(im_array[..., ::-1])  # Convert to PIL image
        
        # Save annotated image to BytesIO object
        image_buffer = BytesIO()
        annotated_image.save(image_buffer, format='JPEG')
        
        # Convert BytesIO object to base64 string
        annotated_image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        
        # Save base64 string to list
        annotated_image_files.append(annotated_image_base64)
        
        # Save annotated image to file (optional)
        annotated_image.save(f'static/annotated_image_{idx}.jpg')
    
    for r in results:
        for c in r.boxes.cls:
            predicted_name=names[int(c)]
    # Prepare response data
    response_data = {
    'predicted_name': predicted_name,
    'annotated_images': annotated_image_files
}
    
    return jsonify(response_data)

# @app.route('/get_labels')
# def get_labels():
#     print("Accessing /get_labels route")
#     root_dir = os.path.dirname(os.getcwd())  # Change this to your directory path
#     file_path = os.path.join(root_dir, 'runs', 'detect', 'predict5', 'labels.txt')
#     print("File path:", file_path)  # Debugging output
#     return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path))

# 

@app.route('/realTime')
def realtime():
    # name = correction.ObjectDetection(capture_index=0)
    try:
        model=YOLO('best_v8.pt')
        real_result=model.predict(source=0,show=True)
        return jsonify({"status": "success"})
    except Exception as e:
        # If an error occurs during prediction
        return jsonify({"status": "failure", "error": str(e)})

# @app.route('/realTime', methods=['GET', 'POST'])
# def realtime():
#     if request.method == 'GET':
#         # Check if the camera is running in the session
#         if 'camera_running' in session and session['camera_running']:
#             session['camera_running'] = False
#             return jsonify({"status": "Camera stopped"})
#         else:
#             return jsonify({"status": "Camera already stopped"})

#     elif request.method == 'POST':
#         try:
#             if 'camera_running' not in session or not session['camera_running']:
#                 # Start the camera only if it's not already running
#                 session['camera_running'] = True
#                 model = YOLO('best_v8.pt')
#                 real_result = model.predict(source=0, show=True)
#                 return jsonify({"status": "Camera started"})
#             else:
#                 return jsonify({"status": "Camera already running"})
#         except Exception as e:
#             # If an error occurs during prediction or camera start
#             return jsonify({"status": "failure", "error": str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
