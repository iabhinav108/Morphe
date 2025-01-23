import os
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import io
import numpy as np
from flask import Flask, request, render_template, jsonify
from torch import nn
from torchvision import models

app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Coordinate Attention block
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.relu(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = identity * a_h * a_w
        return out

# Define the custom DenseNet121 with Coordinate Attention
class CustomDenseNet121(nn.Module):
    def __init__(self):
        super(CustomDenseNet121, self).__init__()
        densenet = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        self.features = nn.Sequential(
            densenet.features.conv0,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
            densenet.features.denseblock1,
            densenet.features.transition1,
            CoordinateAttention(in_channels=128, out_channels=128),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 16)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

# Define the main DeepFake detection model
class DeepFakeDetectionModel(nn.Module):
    def __init__(self):
        super(DeepFakeDetectionModel, self).__init__()
        self.branch1 = CustomDenseNet121()
        self.branch2 = CustomDenseNet121()
        self.branch2.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.branch3 = CustomDenseNet121()
        self.fc_final = nn.Linear(48, 2)

    def forward(self, x):
        x_edge = self.apply_canny(x)
        x_texture = self.extract_texture(x)

        out1 = self.branch1(x)
        out2 = self.branch2(x_edge)
        out3 = self.branch3(x_texture)

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fc_final(out)

        return out

    def apply_canny(self, x):
        x = x.permute(0, 2, 3, 1).cpu().numpy()
        x_edge = []
        for img in x:
            edge = cv2.Canny((img * 255).astype('uint8'), 100, 200)
            x_edge.append(edge)
        x_edge = torch.tensor(np.array(x_edge))
        x_edge = x_edge.unsqueeze(1).float().to(device)
        return x_edge

    def extract_texture(self, x):
        x = x.permute(0, 2, 3, 1).cpu().numpy()
        x_texture = []
        for img in x:
            texture = cv2.Laplacian((img * 255).astype('uint8'), cv2.CV_64F)
            x_texture.append(texture)
        tex_array = np.array(x_texture)
        tex_array = np.transpose(tex_array, (0, 3, 1, 2))
        x_texture = torch.tensor(tex_array).float().to(device)
        return x_texture

model = DeepFakeDetectionModel()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
# model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('video')
        if file:
            temp_filename = 'temp_video.mp4'
            file.save(temp_filename)

            prediction = process_video(temp_filename)
            os.remove(temp_filename)
            return f'Prediction: {prediction}'

        return 'No file received'
    return render_template('index.html')

def process_full_video(filename):
    video_capture = cv2.VideoCapture(filename)

    prediction = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor_image = preprocess_image(pil_image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(tensor_image)
            probability = torch.sigmoid(output)
            prediction = 'Real' if 0 <= probability.item() <= 0.7 else 'Deep Fake'
            break

    video_capture.release()
    return prediction

# def preprocess_image(pil_image):
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     return preprocess(pil_image)

def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    
    if not success:
        raise ValueError("Failed to read video frame")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_image = preprocess(pil_image).unsqueeze(0)

    model.to(device)
    tensor_image = tensor_image.to(device)

    with torch.no_grad():
        output = model(tensor_image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()


if __name__ == '__main__':
    app.run(debug=True)