#import nltk
import json
import cPickle as pickle
import argparse, re
from collections import Counter
#from pycocotools.coco import COCO

#cocofile = "data/annotations/captions_val2014.json"

class Vocab(object):
    def __init__(self, filename, make_vocab, savetofile="data/vocab_file.pkl", threshold=200):
        """
        make_vocab flag is same as (not load flag)
        either build vocab from coco annotations file or load from saved vocab file
        make_vocab saves file to 'savetofile' for future load 
        """

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        if make_vocab:
            assert filename.find('.json') != -1, "Provide Coco dataset annotations .json file"
            with open(filename, 'r') as f:
                coco = json.load(f)
            anns = coco['annotations']
            words = []
            counter = Counter()
            for i in range(len(anns)):
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
            self.save(savetofile)
        else:
            assert filename.find('.pkl') != -1, "Provide saved vocab .pkl file"
            vocab = self.load(filename)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
            self.idx = vocab.idx
            print "Loaded. Vocab dictionary size: ", len(vocab)

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
