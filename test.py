import numpy as np
import torch
import torch.nn as nn
import torchvision
import json, re
import cPickle as pickle
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd import Variable
from vocab import Vocab
from caption_model import ImageFeatures, CaptionGen

with open("data/vocab_file.pkl", 'rb') as f:
    vocab = pickle.load(f)

with open("data/annotations/captions_val2014.json", 'r') as f:
    coco = json.load(f)
anns = coco['annotations']

def getdata(i):
    #i = 0
    caption = anns[i]['caption']
    tokens = re.findall(r"[\w']+", str(caption).lower())
    print tokens
    image_id = str(anns[i]['image_id'])
    if len(image_id) < 6:
        for i in range(6-len(image_id)):
            image_id = "0" + image_id
    filename = "../Coco data/val2014/COCO_val2014_000000" + image_id + ".jpg"
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    caption = torch.LongTensor(caption)
    caption = Variable(caption, requires_grad=False)
    caption = caption.unsqueeze(0)
    print "captionsize: ", caption.size()
    image = image_loader(filename, transform)
    return caption, image

def image_loader(img_name, transform=None):
    """load image, returns tensor"""
    image = Image.open(img_name)
    image = transform(image).float()
    image = torch.Tensor(image)
    image = Variable(image, requires_grad=False)
    print image.size()
    image = image.unsqueeze(0)
    print image.size()
    return image


transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                     std = [ 0.229, 0.224, 0.225 ]),
                                ])
pilTrans = transforms.ToPILImage()


vocab_size = len(vocab)
# model definition
image_features = ImageFeatures(embedding_dim=256)

word_embedding = nn.Embedding(vocab_size, 256)
rnn = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True)
word_decoding = nn.Sequential(nn.Linear(64, vocab_size),
                              nn.LogSoftmax(dim=2),
                              )
for i, ele in enumerate(anns):
    if ele['image_id'] == 9420:
        idx = i
caption, image = getdata(idx)


embedded_words = word_embedding(caption)
print "embedded_words.size: ", embedded_words.size()

image_features.eval()
with torch.no_grad():
    embedded_image = image_features(image)
embedded_image = embedded_image.unsqueeze(1)
print "embedded_image.size: ", embedded_image.size()

x = torch.cat((embedded_image, embedded_words), 1)
print "after concat image, words: ", x.size()

out, hidden_n = rnn(x)

print "out: ", out.size()
N,T,D = out.size()
print "hidden: ", hidden_n[0].size()

#out = out.view(-1, D)
ans = word_decoding(out)
print "final: ", ans.size()
#print torch.sum(torch.exp(ans), 2, keepdim=True)
m, mi = torch.max(ans, 2, keepdim=True)
mi = mi.view(N,T)
ind = mi.data.numpy()
print ind
for i in ind[0]:
    #print i
    print vocab.get_word(i)
#act_img = Image.open(filename)
#act_img.show()
#pilImg = pilTrans(image)
#pilImg.show()
