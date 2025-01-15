import torch
from torchvision import models
from torchvision import transforms
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from typing import Tuple, Dict
import random
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
import gradio as gr
from flask import Flask, render_template, Response
import cv2

device ='cuda' if torch.cuda.is_available() else 'cpu'

# Define data transformations for EfficientNet-B0
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet-B0 default input size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load datasets
train_dataset = ImageFolder('output_dataset/train', transform=train_transforms)
val_dataset = ImageFolder('output_dataset/train', transform=val_transforms)

# train_dataset.to(device)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names=train_dataset.classes
# class_names

num_classes=len(class_names)
# num_classes
# Create title, description and article strings
title = "E-Waste 🍕🥩🍣"
description = "An EfficientNetB0 feature extractor computer vision model to classify images of E-Waste material."
article = "Made by Alok"


# Display some sample images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = val_transforms(img).unsqueeze(0)
    # img.to(device)
    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(model(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


# Step 1: Define the model architecture
num_classes = 10  # Replace with the number of classes in your dataset
model = efficientnet_b0(pretrained=False)  # Same architecture as used during training
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Step 2: Load the state dictionary
state_dict = torch.load("model_weights/efficientnet_b0_e_waste_ep_15_10_classes.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Step 3: Set the model to evaluation mode (for inference)
model.eval()

print("Model loaded successfully!")
# Put EffNetB2 on CPU
model.to("cpu")
# next(iter(model.parameters())).device
test_data_paths = list(Path("").glob("*/*.jpeg"))
# Create a list of example inputs to our Gradio demo
example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=0)]

#################################------Gradio VERSION----------##################################
# Create the Gradio demo
# demo = gr.Interface(fn=predict, # mapping function from input to output
#                     inputs=gr.Image(type="pil"), # what are the inputs?
#                     outputs=[gr.Label(num_top_classes=10, label="Predictions"), # what are the outputs?
#                              gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
#                     examples=example_list,
#                     title=title,
#                     description=description,
#                     article=article)

# Launch the demo!
# demo.launch(debug=True, # print errors locally?
#             share=True) # generate a publically shareable URL?

#################################------FLASK VERSION----------##################################

# app = Flask(__name__)

# Load your model
# model = torch.load("model_name.pth")
# model.eval()

# def gen_frames():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             # Preprocess frame
#             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             tensor_img = ToTensor()(img).unsqueeze(0)
#             prediction = model(tensor_img)
#             class_name = prediction.argmax(1).item()

#             # Display the class name on the frame
#             cv2.putText(frame, str(class_name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Encode and yield the frame
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)

#################################------OPENCV VERSION----------##################################
# cap = cv2.VideoCapture("Screen Recording 2025-01-14 024542.mp4")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_img = ToTensor()(img).unsqueeze(0)
    prediction = model(tensor_img)
    class_name = prediction.argmax(1).item()
    # pred_labels_and_probs = {class_names[i]: float(prediction.argmax(1).item()) for i in range(len(class_names))}
    # Display class name
    cv2.putText(frame, str(class_names[class_name]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()