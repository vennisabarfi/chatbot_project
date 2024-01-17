import random # random choice of possible answers
import json
import torch
from chatbot_model import NeuralNet
from nltk_utils import bag_of_words, tokenize 

# check GPU availability. if not, use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
with open('intents.json', 'r') as f:
    intents = json.load(f)

# load saved model and file
FILE = "data.pth"
data = torch.load(FILE)

# retrieving from data dictionary
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# create model
model = NeuralNet(input_size,hidden_size,output_size).to(device)
# load state_dict
model.load_state_dict(model_state)
# evaluate model
model.eval()



