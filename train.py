import numpy as np
import random
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r',encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 500
batch_size = 8
learning_rate = 0.01
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(f"input_size :{input_size}",f"output_size: {output_size}")

class ChatDataset(Dataset):

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

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Display the model summary
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train = model.fit(X_train, y_train, epochs=500)
# Train the model
losses = []
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Menyimpan loss untuk setiap epoch
        losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
print(f'final loss: {loss.item():.4f}')
# Menampilkan grafik loss
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

#menghitung akurasi pada data pengujian 
X_test = []
y_test = []

for (pattern_sentence, tag) in xy:
    #X:bag of word for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_test.append(bag)
    #y: pytorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

test_dataset =ChatDataset()
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)
#Menghitung akurasi 
model.eval() #set model ke mode evaluasi
with torch.no_grad():
    correct = 0
    total = 0
    for (words, labels) in test_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct/total)*100
print(f'akurasi pengujian : {accuracy:.2f}%')
model.train()

# Menghitung confusion matrix
model.eval()  # Set model ke mode evaluasi
all_predictions = []
all_labels = []
with torch.no_grad():
    for (words, labels) in test_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Visualisasi confusion matrix menggunakan heatmap
# Tag yang ingin dievaluasi
selected_tags = ['info_kampus_lokasi', 'info_kampus_akreditasi']

# Mengambil indeks tag yang terpilih
selected_tag_indices = [tags.index(tag) for tag in selected_tags]

# Menyaring prediksi dan label yang sesuai dengan tag terpilih
filtered_predictions = [pred for pred, label in zip(all_predictions, all_labels) if label in selected_tag_indices]
filtered_labels = [label for label in all_labels if label in selected_tag_indices]

# Menghitung confusion matrix
cm = confusion_matrix(filtered_labels, filtered_predictions)
print("Confusion Matrix:")
print(cm)
"""# Menampilkan confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(cm)"""

# Visualisasi confusion matrix menggunakan heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=selected_tags, yticklabels=selected_tags)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Selected Tags)')
plt.show()
# Laporan Klasifikasi
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=tags))

# Assuming the NeuralNet class is defined in model.py
from model import NeuralNet

# Load the model
loaded_data = torch.load("data.pth")
model_state = loaded_data["model_state"]
input_size = loaded_data["input_size"]
hidden_size = loaded_data["hidden_size"]
output_size = loaded_data["output_size"]

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size)

# Load the model state
model.load_state_dict(model_state)

# Print the model architecture
print(model)
#Access and print the weights
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name}, Size: {param.size()}")
        print(param.data)


# Menampilkan hasil tokenisasi, stemming, dan bag of words untuk beberapa contoh kalimat
for i in range(20):  # Menampilkan hasil untuk 20 kalimat pertama
    example_sentence = intents['intents'][i]['patterns'][0]
    tokenized_words = tokenize(example_sentence)
    stemmed_words = [stem(w) for w in tokenized_words if w not in ignore_words]
    bow_representation = bag_of_words(tokenized_words, all_words)

    print(f"\nExample Sentence: {example_sentence}")
    print(f"Tokenized Words: {tokenized_words}")
    print(f"Stemmed Words: {stemmed_words}")
    print(f"Bag of Words Representation: {bow_representation}")

