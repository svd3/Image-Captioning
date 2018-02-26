#import nltk
import json
import cPickle as pickle
import argparse, re
from collections import Counter
#from pycocotools.coco import COCO

class Vocab(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = int(self.idx)
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word(self, idx):
        #assert idx < len(self.idx2word)
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def make_vocab(filename, threshold=1000):
    #coco = COCO(datafile)
    ##ids = coco.anns.keys()
    #for i, key in enumerate(coco.anns.keys()):
        #caption = str(coco.anns[key]['caption'])
    with open(filename, 'r') as f:
        coco = json.load(f)
    anns = coco['annotations']
    words = []
    counter = Counter()
    for i in range(len(anns)):
        caption = str(anns[i]['caption'])
        #tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens = re.findall(r"[\w']+", str(caption).lower())
        #words.extend(tokens)
        counter.update(tokens)

    #words = [word for word, cnt in counter.items() if cnt >= 1000]
    vocab = Vocab()
    vocab.add_word('<null>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    #for i, word in enumerate(words):
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)

    save(vocab, "data/vocab_file.pkl")
    return vocab

def save(vocab, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)
    print "Saved vocab to " + filename

def load(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

#print "Hello"
if __name__ == "__main__":
    filename = "data/annotations/captions_val2014.json"
    vocab = make_vocab(filename, 200)
    print len(vocab)
