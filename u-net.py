import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()

        # Encoder (ResNet)
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Decoder
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(64, num_classes, kernel_size=1)
        ])

    def forward(self, x):
        # Encoder
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        # Decoder
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(self.decoder) - 1:  # Skip the last layer
                x = torch.cat((x, features[-i - 2]), dim=1)  # -2 because of the last pooling layer
        return x



def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)


def remove_background(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
    mask = torch.sigmoid(output[0, 0]).numpy()
    foreground = np.array(Image.open(image_path).convert('RGBA'))
    background = np.zeros_like(foreground)
    background[:, :, 3] = (mask > 0.5) * 255
    foreground[:, :, 3] = 255 - background[:, :, 3]
    return Image.fromarray(foreground), Image.fromarray(background)


def display_images(foreground, background):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(foreground)
    axs[0].axis('off')
    axs[0].set_title('Foreground')
    axs[1].imshow(background)
    axs[1].axis('off')
    axs[1].set_title('Background')
    plt.show()


if __name__ == "__main__":
    # Load the UNet model
    model = UNet(num_classes=1)

    # Load pre-trained weights if available
    # model.load_state_dict(torch.load('unet_weights.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Provide the path to your input image
    image_path = "/home/jafar/sceneGen/Scene-Generator/test_1/results_3/image.png"

    # Remove background
    foreground, background = remove_background(image_path, model)

    # Display foreground and background
    display_images(foreground, background)
