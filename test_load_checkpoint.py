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

def load_checkpoint(resumefile):
    if resumefile:
        if os.path.isfile(resumefile):
            print("=> loading checkpoint '{}'".format(resumefile))
            checkpoint = torch.load(resumefile)
            start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            img_features.load_state_dict(checkpoint['encoder_state_dict'])
            generator.load_state_dict(checkpoint['decoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resumefile, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resumefile))

image_size = 224
batch_size = 16
embedding_dim = 128
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

data_loader = get_loader(root=root, cocofile=cocofile, vocab=vocab, transform=transform, batch_size=16)

# Model
img_features = ImageFeatures(embedding_dim)
generator = CaptionGen(embedding_dim=embedding_dim, hidden_dim=64, vocab_size=vocab_size, batch_size=1)
generator.hidden = generator.init_hidden()

loss = nn.NLLLoss()
model_parameters = list(generator.parameters()) + [param for param in img_features.parameters() if param.requires_grad]

optimizer = torch.optim.Adam(model_parameters, lr=1e-3)

load_checkpoint('../Coco data/model_best.pth.tar')

idx = 0
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

print vocab.get_word(0), vocab.get_word(1), vocab.get_word(2), vocab.get_word(3) 
