import torch
import torch.nn as nn

# build feed forward neural network with 2 hidden layers
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes): #num_classes is fixed
        super(NeuralNet,self).__init__()
        # create 3 linear layers
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,num_classes)
        self.relu = nn.ReLU() # activation function

    def forward(self,x):
        out= self.l1(x)
        out = self.relu(out)
        out= self.l2(out)
        out = self.relu(out)
        out= self.l3(out)
        # no activation and no softmax
        return out 