import numpy as np
import torch
import torch.nn as nn
import torchvision
import json, pickle
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from collections import OrderedDict
from torch.autograd import Variable
import cPickle

with open("data/annotations/captions_val2014.json", 'r') as f:
    coco = json.load(f)
images = coco['images']

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                     std = [ 0.229, 0.224, 0.225 ]),
                                ])

def image_loader(img_name, transform=None):
    """load image, returns tensor"""
    image = Image.open(img_name)
    image = transform(image).float()
    image = torch.Tensor(image)
    if image.size(0) == 1:
        image = torch.cat((image, image, image), 0)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image

def getdata(i):
    filename = "./data/val2014/" + str(images[i]['file_name'])
    image = image_loader(filename, transform)
    idx = images[i]['id']
    return idx, image

if __name__ == '__main__':
    alexnet = models.alexnet(pretrained=True) # loading pretrained model
    feature_extractor =  alexnet.features
    features = {}

    feature_extractor.eval()
    for i in range(len(images)):
        if i%1000 == 0: print i
        idx, image = getdata(i)
        image_features = feature_extractor(image)
        image_features = image_features.view(image_features.size(0), -1)
        features[idx] = image_features.data
    with open("data/image_features.pkl", 'wb') as f:
        pickle.dump(features, f)
    print "Saved to file " + "data/image_features.pkl"
