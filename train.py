import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

from data_loader import data_loader
from caption_model import ImageFeatures, CaptionGen

image_size = 224
batch_size = 64
embedding_dim = 256

# load vocab_file
with open("data/vocab_file.pkl", 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
# define image transform
transform = transforms.Compose([transforms.Resize((224,224))
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

img_features = ImageFeatures(embedding_dim)
generator = CaptionGen(embedding_dim, hidden_dim=256, vocab_size)
print generator.parameters()

loss = nn.NLLLoss()
model_parameters = list(generator.parameters()) + list(img_features.parameters())
optimizer = torch.optim.Adam(model_parameters, lr=1e-3)

def train(data_loader, epochs=500):
    for epoch in range(epochs):
        for i, (image, caption_in, caption_out) in enumerate(data_loader):
            optimizer.zero_grad()
            features = img_features(image)
            pred_caption = generator(features, caption_in)
            pred = pred_caption.permute(dims=(0,2,1)) # N, C, T ; C = number of words / vocab_size
            # this is needed for nll loss in 2d otherwise do input.view(N*T,C) and target.view(N*T,)
            error = loss(pred_caption, caption_in)
            error.backward()
            optimizer.step()

            if i % 100 == 0:
                print "Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f"
                      %(epoch, epochs, i, len(data_loader),
                        loss.data[0], np.exp(loss.data[0]))

        save_checkpoint({'epoch': epoch + 1,
                         'encoder_state_dict': img_features.state_dict(),
                         'decoder_state_dict': generator.state_dict(),
                         'optimizer' : optimizer.state_dict(),
                         }, is_best=False)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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
