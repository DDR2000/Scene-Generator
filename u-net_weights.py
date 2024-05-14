import torch
import torch.nn as nn
from torchvision.models import segmentation

# Load pre-trained U-Net model
model = segmentation.deeplabv3_resnet50(pretrained=True)

# Remove the classifier layer since we don't need it for background extraction
model.classifier = nn.Identity()

# Save the model weights
torch.save(model.state_dict(), 'unet_background.pth')
