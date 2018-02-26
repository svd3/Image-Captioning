import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

from vocab import Vocab
import cPickle as pickle
import json
from data_loader import CocoData, get_loader
from caption_model import ImageFeatures, CaptionGen

from train import train, save_checkpoint, load_checkpoint
from data_loader import CocoData

image_size = 224
batch_size = 64
embedding_dim = 256
root = "../Coco data/val2014"
cocofile = "../Coco data/annotations/captions_val2014.json"

# load vocab_file
with open("data/vocab_file.pkl", 'rb') as f:
    vocab = pickle.load(f)

with open(cocofile, 'r') as f:
    coco = json.load(f)
anns = coco['annotations']

vocab_size = len(vocab)
# define image transform
transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                      std = [ 0.229, 0.224, 0.225 ]),
                                ])

data_loader = get_loader(root=root, cocofile=cocofile, vocab=vocab, transform=transform, batch_size=64)

# Model
img_features = ImageFeatures(embedding_dim)
generator = CaptionGen(embedding_dim=embedding_dim, hidden_dim=256, vocab_size=vocab_size)

loss = nn.NLLLoss()
model_parameters = list(generator.parameters()) + [param for param in img_features.parameters() if param.requires_grad]

optimizer = torch.optim.Adam(model_parameters, lr=1e-3)

load_checkpoint('checkpoint.pth.tar')

idx = 20
print anns[idx]['caption']
coco = CocoData(root=root, cocofile=cocofile, vocab=vocab, transform=transform)
image, caption = coco[idx]

image = Variable(image)
caption = Variable(caption)

img_features.eval()
features = img_features(image)

pred_caption = generator(features, caption)

_, words_idx = torch.max(pred_caption, 2, keepdim=True)

print words_idx.size()
ind = words_idx.data.numpy()
ind = ind.reshape(-1,)
for i in ind:
    print vocab.get_word(i)
