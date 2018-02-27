import numpy as np
import os, shutil
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

image_size = 224
batch_size = 16
embedding_dim = 128
root = "../Coco data/val2014"
cocofile = "../Coco data/annotations/captions_val2014.json"
transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                      std = [ 0.229, 0.224, 0.225 ]),
                                ])

def train(data_loader, batch_size=16, epochs=50, load=True):
    # load vocab_file
    with open("data/vocab_file.pkl", 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    # define image transform

    data_loader = get_loader(root=root, cocofile=cocofile, vocab=vocab, transform=transform, batch_size=batch_size)

    # Model
    img_features = ImageFeatures(embedding_dim)
    #generator = CaptionGen(embedding_dim=embedding_dim, hidden_dim=64, vocab_size=vocab_size, batch_size=batch_size)
    generator = CaptionGen(embedding_dim, 64, vocab_size, batch_size)
    loss = nn.NLLLoss()
    model_parameters = list(generator.parameters()) + [param for param in img_features.parameters() if param.requires_grad]

    optimizer = torch.optim.Adam(model_parameters, lr=1e-4)

    if load:
        load_checkpoint('../Coco data/checkpoint.pth.tar')

    prev_best = torch.Tensor([1000.])
    is_best = False
    for epoch in range(epochs):
        for i, (image, caption_in, caption_out) in enumerate(data_loader):
            print i
            img_features.zero_grad()
            generator.zero_grad()
            generator.hidden = generator.init_hidden()
            image = Variable(image)

            optimizer.zero_grad()
            
            features = img_features(image)
            pred_caption = generator(features, caption_in)
            pred = pred_caption.permute(dims=(0,2,1)) # N, C, T ; C = number of words / vocab_size
            # this is needed for nll loss in 2d otherwise do input.view(N*T,C) and target.view(N*T,)
            error = loss(pred, caption_out)
            error.backward()
            optimizer.step()

            if i % 100 == 0:
                print "\n"
                print "Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f" %(epoch, epochs, i, len(data_loader),error.data[0], np.exp(error.data[0]))
                if error.data[0] < prev_best:
                    is_best = True
                    prev_best = error.data[0]
                else:
                    is_best = False
                save_checkpoint({'encoder_state_dict': img_features.state_dict(),
                                 'decoder_state_dict': generator.state_dict(),
                                 }, is_best=is_best)
        print "Epoch done."
    print "Training Done."

def save_checkpoint(state, is_best, filename='../Coco data/checkpoint.pth.tar'):
    print "Writing..."
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../Coco data/model_best.pth.tar')
    print "Checkpoint created."

def load_checkpoint(resumefile):
    if os.path.isfile(resumefile):
        print("=> loading checkpoint '{}'".format(resumefile))
        checkpoint = torch.load(resumefile)
        img_features.load_state_dict(checkpoint['encoder_state_dict'])
        generator.load_state_dict(checkpoint['decoder_state_dict'])
        print("=> loaded checkpoint '{}'".format(resumefile))
    else:
        print("=> no checkpoint found at '{}'".format(resumefile))

if __name__ == '__main__':
    train(data_loader, 5)
