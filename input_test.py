import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd import Variable

from PIL import Image

alexnet = models.alexnet(pretrained=True) # loading pretrained model
alexnet.eval()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

imsize = 224
#loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
loader = transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image#assumes that you're using GPU

image = image_loader("data/puppy.jpg")

a = torch.exp(alexnet(image).data)
a = a/torch.sum(a)
a = a.numpy()
print a.argsort()[0][995:1000]
