import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import argparse
import struct
import numpy as np
import gzip



#parse command line arguments from SageMaker SDK
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])

#Mapping training and test data locations from S3 to traning container environment
parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])


#Mapping hyperparameters
parser.add_argument("--batch_size", type=str, default=16)
parser.add_argument("--epochs", type=str, default=1)

#parsing all the added parameters
args = parser.parse_args()
batch_size = int(args.batch_size)

#Method to load, parse and convert the dataset into Torch Tensor objects
def convert_to_tensor(path, images_file, labels_file):
    # Open the images file and decompress it
    with gzip.open(os.path.join(path, images_file), 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)
        
    # Open the labels file and decompress it
    with gzip.open(os.path.join(path, labels_file), 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    # Convert the images and labels to tensors
    images = images.astype(np.float32) / 255.0
    images = images.reshape(-1, 28, 28, 1)
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    return images, labels
        

#Class to hold the raw dataset objects, extending from torch.utils.data.Dataset class
class FashionMNIST(Dataset):
    def __init__(self, path, train=True):
        if train:
            images_file = "train-images-idx3-ubyte.gz"
            labels_file = "train-labels-idx1-ubyte.gz"
        else:
            images_file = "t10k-images-idx3-ubyte.gz"
            labels_file = "t10k-labels-idx1-ubyte.gz"

        self.images, self.labels = convert_to_tensor(path, images_file, labels_file)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    


#Create PyTorch dataloaders for training and test datasets
train_dataloader = DataLoader(FashionMNIST(args.train, train=True), batch_size=batch_size)
test_dataloader = DataLoader(FashionMNIST(args.test, train=False), batch_size=batch_size)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#training the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

#testing the model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#You can also move this into the hyperparameters if you wish
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#save the model
path = os.path.join(args.model_dir, "model.pth")
torch.save(model.state_dict(), path)
print("Saved PyTorch Model State to model.pth")