import pickle
import numpy as np

class Test(object):
    def __init__(self, filename, load):
        if not load:
            self.values = np.array([0,1,2,3,4,5])
            self.save(filename)
        else:
            with open(filename, 'rb') as f:
                var = pickle.load(f)
            self.values = var.values
            print "Loaded."

    def add(self, x):
        self.values += x

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print "Saved file."

f = 'savefile123.pkl'
a = Test(f, False)
print a.values
a.add(10)
print a.values
a.save(f)

v = Test(f, True)
print v.values
v.add(-10)
print v.values
