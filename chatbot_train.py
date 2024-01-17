import json 
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

from chatbot_model import NeuralNet

# load our dataset file 

with open('intents.json', 'r') as f:
    intents= json.load(f)

print(intents)

# Labeling Data with Tokenization

all_words = []
tags = []
xy = [] # holds patterns and tags

for intent in intents["intents"]:
    tag = intent["tag"] # access tag key in dictionary(dataset)
    tags.append(tag) # append to tags array
    for pattern in intent["patterns"]: #now for patterns
        w = tokenize(pattern)
        all_words.extend(w) # add specified list elements to the end of current list
        xy.append((w,tag)) #add word with its corresponding tags (labeling)

# Avoid punctuation
ignore_words = ["?","!",".",","]
all_words= [stem(w) for w in all_words if w not in ignore_words] #strip punct and stem

# Testing
print(all_words) # all the words in lowercase, no punct and stem
all_words = sorted(set(all_words)) # sort and return unique elements
tags = sorted(set(tags))
print(tags)

 # Create bag of words/training data

x_train = [] #bag of words
y_train = []

# unpack xy tuple created earlier    
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

# labels/y data
    label = tags.index(tag) #create numerical labels for tags
    print(label)

y_train.append(label) # CrossEntropyLoss

# convert to numpy array

x_train = np.array(x_train)
y_train =np.array(y_train)

# create a custom dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples # length of x_train
    
# Hyperparameters
batch_size = 2
hidden_size = 2
output_size = len(tags) #number of different text we have
input_size = len(x_train[0]) # length of bag_of_words  = all_words = first inpit of x_train because they all have the same size
learning_rate = 0.001
num_epochs= 1000

# Testing to ensure equal sizes
print(input_size, len(all_words))
print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset =dataset,batch_size=batch_size, shuffle=True) # num_workers = 8) #multi-process data loading is on


# Create model
model = NeuralNet(input_size,hidden_size,output_size)
# check GPU availability. if not, use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) # Adam optimizer

# training loop
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device) #push to device
        labels = labels.to(device) # push to device

    #forward pass
        outputs = model(words)  #words = input_data
        loss = criterion(outputs,labels)

    #backward and optimizer step
        optimizer.zero_grad() #empty the gradients
        loss.backward() #calculate back propagation
        optimizer.step()

    if (epoch +1)% 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'final loss, loss = {loss.item():.4f}') # print the final loss here

# save and load the model and data
data ={
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
     "hidden_size": hidden_size,
     "all_words": all_words,
     "tags": tags


}
# Save to file
FILE = "data.pth"
torch.save(FILE)

print(f'training complete. file saved to {FILE}')