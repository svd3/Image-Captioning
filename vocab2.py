#import nltk
import json
import pickle
import argparse, re
from collections import Counter
#from pycocotools.coco import COCO

cocofile = "data/annotations/captions_val2014.json"

class Vocab(object):
    def __init__(self, cocofile, threshold=200, load=True, loadfile):
        #assert load is not make, "Either load from file or make vocab and save to file"
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        if not load:
            with open(cocofile, 'r') as f:
                coco = json.load(f)
            anns = coco['annotations']
            words = []
            counter = Counter()
            for i in range(len(coco)):
                caption = str(anns[i]['caption'])
                tokens = re.findall(r"[\w']+", str(caption).lower())
                #words.extend(tokens)
                counter.update(tokens)
            self.add_word('<null>')
            self.add_word('<start>')
            self.add_word('<end>')
            self.add_word('<unk>')
            #for i, word in enumerate(words):
            for word, count in counter.items():
                if count >= threshold:
                    self.add_word(word)
            self.save_vocab("data/vocab_file.pkl")
        else:
            vocab = load(loadfile)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
            self.idx = vocab.idx
            print "Loaded."

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word(self, idx):
        assert idx < len(Vocab)
        return self.idx2word[idx]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print "Saved vocab to " + filename

    def load(self, filename):
        with open(filename, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
