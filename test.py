import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
alexnet = models.alexnet(pretrained=True) # loading pretrained model
print alexnet
image_embedding_size = 256
features = alexnet.features
for param in features.parameters():
    param.requires_grad = False

linear = nn.Linear(9216, image_embedding_size)
linear.weight.data.normal_(0.0, 0.02)
linear.bias.data.fill_(0)
#features.add_module("fc1", nn.Linear(9216, image_embedding_size))
feature_proj = nn.Sequential(nn.Linear(9216, image_embedding_size),
                                  nn.BatchNorm1d(image_embedding_size),
                                  )

def init_weights(m):
    classname = m.__class__.__name__
    print classname
    if classname.find('Conv') != -1:
        pass #m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Embed') != -1:
        m.weight.data.uniform_(-0.2, 0.2)
#initialize weights and biases
feature_proj.apply(init_weights)

embedding = nn.Embedding(1024, 256)
init_weights(embedding)
