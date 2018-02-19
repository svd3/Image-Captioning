import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
rnn = nn.RNN(3, 3)
inputs = [Variable(torch.randn((1, 3))) for _ in range(5)]  # make a sequence of length 5

hidden0 = (Variable(torch.randn(1, 1, 3)), Variable(torch.randn(1, 1, 3)))
hidden = hidden0
#print "hidden_init: ", hidden
#print "****************111*********************"
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    #print "out: ", out
    #print "*************************************"
    #print "hidden: ", hidden
    #print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print inputs.size()
hidden = (Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn(1, 1, 3)))  # clean out hidden state
out, hidden = lstm(inputs, hidden0)
print "out: ", out
print "****************111*********************"
print "hidden: ", hidden
