import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO

class Vocab(object):
    def __init__(self,):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word(self, idx):
        assert idx < len(Vocab)
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

datafile = 'path.json'

def construct(datafile):
    coco = COCO(datafile)
    #ids = coco.anns.keys()
    words = []
    for i, key in enumerate(coco.anns.keys()):
        caption = str(coco.anns[key]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        #counter.update(tokens)
        words.append(tokens)

    vocab = Vocab()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def save_vocab(filename):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)
    print "Saved file."
