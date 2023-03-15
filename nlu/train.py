import json
import sys
from os import listdir
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import DataLoader, Dataset
from utils import bag_of_words, stem, tokenize

if len(sys.argv) != 2:
    print("Usage: train.py <intents_dir>")
    exit(1)

# List of all intents that will be used to train the model
intents = []

train_files_dir = sys.argv[1]
for train_file in listdir(train_files_dir):
    path = join(train_files_dir, train_file)
    print(f"Reading intent file at path {path}")
    with open(join(train_files_dir, train_file), "r") as f:
        intent = json.load(f)
        intents.append(intent)

all_words = []
tags = []
xy = []

for intent in intents:
    tag = intent["intent"]
    tags.append(tag)
    for utterance in intent["utterances"]:
        w = tokenize(utterance)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns:", xy)
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

X_train = []
y_train = []
for utterance, tag in xy:
    bag = bag_of_words(utterance, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)


class IntentDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = IntentDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


print(f"final loss: {loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}


FILE = "data.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")
