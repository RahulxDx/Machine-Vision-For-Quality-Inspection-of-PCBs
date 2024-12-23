from flask import Flask, request, send_file, render_template_string
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("best.pt")  # Ensure the .pt file is accessible

OUTPUT_FOLDER = "static"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCB Fault Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #282a36; /* New background color */
            color: #e5e5e7;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .main-container {
            text-align: center;
            background: #2c2c2e;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.8);
            width: 90%;
            max-width: 600px;
        }
        .main-container h1 {
            font-size: 2.5rem;
            margin-bottom: 20px;
            color: #ffd700;
        }
        .file-upload {
            background-color: #3c3c3e;
            border: 2px dashed #545456;
            padding: 30px;
            border-radius: 10px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .file-upload input {
            opacity: 0;
            position: absolute;
            z-index: -1;
        }
        .file-upload:hover {
            background-color: #444446;
        }
        .submit-btn {
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(90deg, #ff8c00, #ff4500);
            color: white;
            cursor: pointer;
            transition: 0.3s;
        }
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0px 5px 15px rgba(255, 140, 0, 0.5);
        }
        .result-section {
            margin-top: 30px;
        }
        .result-section img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        }
    </style>
</head>
<body>
    <div class="main-container">
        <h1>PCB Fault Detection</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <div class="file-upload">
                <label for="fileInput">
                    <i class="fas fa-cloud-upload-alt fa-3x"></i>
                    <p>Click to upload an image</p>
                </label>
                <input id="fileInput" type="file" name="image" accept="image/*" required>
            </div>
            <button type="submit" class="submit-btn">Run Detection</button>
        </form>
        {% if output_image %}
        <div class="result-section">
            <h2>Detection Results:</h2>
            <img src="{{ output_image }}" alt="Detected Objects">
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded!", 400

    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image file!", 400

    # Run YOLOv8 inference
    results = model.predict(source=img, imgsz=416)

    # Check detection results
    detection_status = "No Objects Detected"
    if len(results[0].boxes) > 0:
        detection_status = "Objects Detected"

    # Annotate the image
    annotated_image = results[0].plot()
    output_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
    cv2.imwrite(output_path, annotated_image)

    return render_template_string(
        HTML_TEMPLATE,
        output_image=f"/{output_path}",
        detection_status=detection_status,
    )

if __name__ == "__main__":
    app.run(debug=True)
