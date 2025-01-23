import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Coordinate Attention block
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
        visualize_layers("CoordinateAttention", x)  # Visualize input
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
        
        visualize_layers("CoordinateAttention", out)  # Visualize output
        
        return out

def visualize_layers(name, tensor, title=None):
    """Visualize the feature maps from a tensor output at different stages."""
    tensor = tensor.detach().cpu().numpy()
    
    if len(tensor.shape) == 4:  # For Conv layer outputs (BxCxHxW)
        fig, axes = plt.subplots(1, min(4, tensor.shape[1]), figsize=(15, 15))
        if title is not None:
            plt.suptitle(title)
        for i in range(min(4, tensor.shape[1])):
            axes[i].imshow(tensor[0, i, :, :], cmap='viridis')
            axes[i].set_title(f"{name} Layer {i+1}")
            axes[i].axis('off')
    elif len(tensor.shape) == 2:  # For FC layer outputs (BxN)
        plt.figure(figsize=(10, 4))
        plt.plot(tensor[0])
        plt.title(f"{name} - Output of FC Layer")
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation')
    else:
        print(f"Cannot visualize tensor of shape: {tensor.shape}")
    
    plt.show()

# Custom DenseNet121 with Coordinate Attention
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
        visualize_layers("CustomDenseNet121 Input", x)  # Visualize input
        # Iterate through features to visualize each layer
        for layer in self.features:
            x = layer(x)
            visualize_layers("CustomDenseNet121", x, f'Output after {layer.__class__.__name__}')
        
        x = self.fc(x)
        visualize_layers("CustomDenseNet121 Final", x, "Final Output after FC Layer")
        return x

# Main DeepFake detection model
class DeepFakeDetectionModel(nn.Module):
    def __init__(self):
        super(DeepFakeDetectionModel, self).__init__()
        self.branch1 = CustomDenseNet121()
        self.branch2 = CustomDenseNet121()
        self.branch2.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.branch3 = CustomDenseNet121()
        self.fc_final = nn.Linear(48, 2)

    def forward(self, x):
        visualize_layers("DeepFakeDetectionModel Input", x)  # Visualize input
        x_edge = self.apply_canny(x).to(device)
        x_texture = self.extract_texture(x).to(device)

        out1 = self.branch1(x)
        visualize_layers("DeepFakeDetectionModel Branch 1", out1, "Output after Branch 1")

        out2 = self.branch2(x_edge)
        visualize_layers("DeepFakeDetectionModel Branch 2", out2, "Output after Branch 2")

        out3 = self.branch3(x_texture)
        visualize_layers("DeepFakeDetectionModel Branch 3", out3, "Output after Branch 3")

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fc_final(out)
        visualize_layers("DeepFakeDetectionModel Final", out, "Final Output after Combining Branches")

        return out

    def apply_canny(self, x):
        x = x.permute(0, 2, 3, 1).cpu().numpy()  # BxCxHxW -> BxHxWxC
        x_edge = []
        for img in x:
            edge = cv2.Canny((img * 255).astype('uint8'), 100, 200)
            x_edge.append(edge)
        
        x_edge = torch.tensor(np.array(x_edge))
        x_edge = x_edge.unsqueeze(1).float()
        return x_edge

    def extract_texture(self, x):
        x = x.permute(0, 2, 3, 1).cpu().numpy()
        x_texture = []
        for img in x:
            texture = cv2.Laplacian((img * 255).astype('uint8'), cv2.CV_64F)
            x_texture.append(texture)
        
        tex_array = np.array(x_texture)
        tex_array = np.transpose(tex_array, (0, 3, 1, 2))
        x_texture = torch.tensor(tex_array).float()
        return x_texture

if __name__ == '__main__':
    root_dir = 'final_dataset'
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    img_path = os.path.join(root_dir, os.listdir(root_dir)[0])
    image = Image.open(img_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0).to(device)

    model = DeepFakeDetectionModel().to(device)

    output = model(image)
    print(f'Final Output: {output}')
