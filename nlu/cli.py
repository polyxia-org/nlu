import torch
from model import NeuralNet
from utils import bag_of_words, tokenize

from nlu.functions import INTENTS_HANDLER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print("Enter a query to get the associated intent (type 'quit' to exit)")
while True:
    payload = input("You: ")
    if payload == "quit":
        break

    payload = tokenize(payload)
    X = bag_of_words(payload, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    out = model(X)
    _, predicted = torch.max(out, dim=1)

    intent = tags[predicted.item()]

    probs = torch.softmax(out, dim=1)
    prob = probs[0][predicted.item()]

    print(prob.item())
    if prob.item() > 0.90:
        print(f"Detected intent: {intent}")
        print(INTENTS_HANDLER.get(intent)())
    else:
        print("Cannot find an intent that match your query")
