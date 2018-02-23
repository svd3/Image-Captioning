from PIL import Image

import torch
import torch.utils.data as data
from pycocotools.coco import COCO
from vocab import Vocab
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])

def image_loader(img_name, transform=None):
    """load image, returns tensor"""
    image = Image.open(img_name)
    image = transform(image).float()
    image = torch.Tensor(image)
    #image = image.unsqueeze(0)
    return image#assumes that you're using GPU

print "here"
coco = COCO()
