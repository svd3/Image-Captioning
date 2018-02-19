import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.nn import Parameter
import torch.autograd as autograd
from torch.autograd import Variable


class CNN_FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim):
        super(self.__class__, self).__init__()
        alexnet = models.alexnet(pretrained=True) # loading pretrained model
        self.features = alexnet.features
        for param in self.features.parameters():
            param.requires_grad = False
        self.feature_proj = nn.Sequential(nn.Linear(9216, 4096),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm1d(4096),
                                          nn.Linear(4096, embedding_dim),
                                          )
        #initialize weights and biases
        self.feature_proj.apply(self.init_weights)

    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            pass #m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.feature_proj(x)
        return x

class RNN_CaptionGen(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, celltype='lstm', num_layers=1):
        super(self.__class__, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        if celltype == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.word_decoding = nn.Sequential(nn.Linear(hidden_dim, vocab_size),
                                           nn.LogSoftmax(),
                                           )
        init_weights(self.word_embedding)
        init_weights(self.word_decoding)

    def forward(self, features, words, seq_len):
        embedded_words = self.word_embedding(words) # dim: N, T, D
        #print embedded_words.size()
        # features dim: N, D
        # rnn input dim: T, N, D    rnn output dim: N, T, H (batch_first=True)
        """
        In Pytorch we can either run RNNs stepwise in iterative loop or we can
        pass the entire sequence and it does that for us more conveniently.
        So we'll create an input sequence like (image_embedding, word1_em, word2_em, ... wordn_em)
        and pass that
        """
        embedded_words = self.word_embedding(words) # dim: T, N, D
        embedded_image = features.unsqueeze(0) # added time-seq dim
        # other way of doing this would be:
        # N, D = features.size()
        # embedded_image = features.view(1, N, D)
        x = torch.cat((embedded_image, embedded_words), 0) # dim: (T+1), N, D
        out, hidden_n = self.rnn(x)
        #not sure what exactly should we return
        return self.word_decoding(out[0]) # first predicted word in sequence (probs)

    def init_weights(m):
        classname = m.__class__.__name__
        #print classname
        if classname.find('Conv') != -1:
            pass #m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.05)
            m.bias.data.fill_(0)
        elif classname.find('Embed') != -1:
            m.weight.data.uniform_(-0.2, 0.2)

    def gen_caption(self, features):
        inputs = features.unsqueeze(0)
        sentence = []
        for i in range():
             out, hidden = self.lstm(inputs, hidden)
             prob_words = self.word_decoding(out)
             # sort and sample from top 10 indices

             pred_word = sample()
             sentence.append(pred_word)
             inputs = self.word_embedding(pred_word)
             inputs = inputs.unsqueeze(0)
        sentence = torch.cat(sentence, 0)
        return sentence

class Trainer(object):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, celltype='lstm', num_layers=1):
        self.feature_extractor = CNN_FeatureExtractor(embedding_dim, hidden_dim,
                                                 vocab_size, celltype, num_layers)
        self.caption_generator = RNN_CaptionGen(embedding_dim)

        self.parameters = list(caption_generator.parameters()) +
                           list(feature_extractor.parameters())

        optimizer = torch.optim.Adam(self.parameters)

    def train(data_loader, epochs=500):
        for epoch in range(epochs):
            for iters, (image, caption, lengths) in enumerate(data_loader):
                optimizer.zero_grad()

                images = Variable(images, volatile=True)
                caption = Variable(caption, volatile=True)
                target = caption
                feature_extractor.zero_grad()
                caption_generator.zero_grad()
                features = feature_extractor(images)
                pred_words = caption_generator(features, caption)
                loss = torch.nn.functional.nll_loss(pred_words, target)
                loss.backward()
                optimizer.step()
            print loss
