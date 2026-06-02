import time
print("The thinking part actually just wastes time :D")
print("Thinking", end ="")
for i in range(4):
    time.sleep(0.5)
    print(".", end="")
time.sleep(1)
print(".")

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
#import libraries necessary for reading images and machine learning


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=256*8*8, out_features=256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    #same thing as in train.py

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#This is the part to use the different structures of GPUs to shorten the training time
model = CNN().to(device)
#assign the device
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval() #load the model saved from training

preprocess = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
#same way of preprocess for the image so that the model can read the image

def run_prediction(image):
    """
    Input: tensor
    Output: string
    Takes the image and run it as in the model to predict the species
    """
    rawimage = Image.open(image).convert('RGB')
    imagetensor = preprocess(rawimage).unsqueeze(0).to(device) #resize the same way, fake a batch number to make it run
    with torch.no_grad():
        output = model(imagetensor) #give its prediction
        probability = F.softmax(output, dim=1) #give percentages
        _, predicted = torch.max(output, 1) #compare number and see cat or dog wins

    classes = {0: "Cat", 1: "Dog"} #0 stands for cat, 1 for dog
    result = classes[predicted.item()] #Change 0 or 1 to cat or dog
    confidence = probability[0][predicted.item()].item() * 100 #output the confidence
    print("\n------------------------------")
    print(f"Prediction: This is a {result}!")
    print(f"Confidence score: {confidence:.4f}%")
    print("------------------------------\n")
    #Output the prediction

image = "/Users/thomasgu/PycharmProjects/Cat_vs_Dogs/Predict Pictures/Weird Cat.jpg"
img = Image.open(image)
img.show()
run_prediction(image) #run the prediction by inputting the image name