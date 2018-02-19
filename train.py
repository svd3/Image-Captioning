import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from data_loader import get_loader

from caption_model import CNN_FeatureExtractor, RNN_CaptionGen

im_size = 224
batch_size = 64
embedding_dim = 128
vocab_size
# define image transform
transform = transforms.Compose([transforms.RandomResizedCrop(im_size),
                                transforms.RandomHorizontalFlip()
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                      std = [ 0.229, 0.224, 0.225 ]),
                                ])

coco = CocoDataset(root=root, json=json, vocab=vocab, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=num_workers,
                                          collate_fn=collate_fn)
# look into this data loading part

feature_extractor = CNN_FeatureExtractor(embedding_dim)
caption_generator = RNN_CaptionGen(embedding_dim, hidden_dim=256, vocab_size)


loss = nn.NLLLoss()
model_parameters = list(caption_generator.parameters()) + list(feature_extractor.parameters())

optimizer = torch.optim.Adam(model_parameters, lr=1e-3)


 for epoch in range(args.num_epochs):
