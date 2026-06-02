import torch
import torch.nn as nn
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
#Import libraries for machine learning and for importing datasets

dataset = load_dataset("microsoft/cats_vs_dogs", split="train")
#load the dataset from huggingface
splitset = dataset.train_test_split(test_size=0.2, seed=42)
trainset = splitset["train"]
testset = splitset["test"]
#split them into two sets, train and test

preprocess = transforms.Compose([ #process the images to what we want them to be for the model to be able to read the image
    transforms.Resize((128,128)), #resize all the images to the same size for training
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(0.5),# flip it or rotate it randomly to prevent overfitting
    transforms.ToTensor(), #Change it to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
]) #These make the model have similiar data so that it won't be dominated from one extreme value

def transform_batch(examples):
    """
    input: dictionary
    output: dictionary
    Converts each image to RGB format and apply the preprocess, resize, tensor convertion, etc.
    """
    examples['pixel_values'] = [preprocess(img.convert('RGB')) for img in examples['image']]
    return examples
#transform batches to tensors for training
trainset.set_transform(transform_batch)
testset.set_transform(transform_batch)
#transform both the trainset and the testset

def custom_collate(batch):
    """
    input: list
    output: dictionary
    bypass PyTorch default behavior
    ignore raw PIL images
    """
    inputs = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'pixel_values': inputs, 'labels': labels}
#It's kind of weird in this model where the default method will read my raw data
#So I have to create a function to skip these data which will mess the training up

train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=custom_collate)
test_dataloader = DataLoader(testset, batch_size=32, shuffle=False, collate_fn=custom_collate)
#Now images are loaded and the dataloader is prepared
#Now we need the model
class CNN(nn.Module):
    #We are making a CNN model
    """
    input: tensors
    output: float
    Takes in the image tensor, squash them to layers, make logical connections, output what it thinks
    """
    def __init__(self):
        #define the layers that the model will break down
        super(CNN, self).__init__() #reach it to pytorch library
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) #1st layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64) #2nd layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128) #3rd layer
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256) #4th layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #Only look at the most dominant pixels
        self.fc1 = nn.Linear(in_features=256*8*8, out_features=256) #connect the different pieces of image, make them logical clues
        self.dropout = nn.Dropout(0.2) #Discard some connections so that it won't easily overfit by focusing on a single value
        self.fc2 = nn.Linear(in_features=256, out_features=2) #make them 2 outputs, cat or dog

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        #Process what we did above
        x = x.view(-1, 256*8*8) #change the 3d block of numbers into a single line
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) #process what we did above for logical connections
        return x
        #return the value of logical predictions
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#This is the part to use the different structures of GPUs to shorten the training time
model = CNN().to(device)
#assign the device
criterion = nn.CrossEntropyLoss() #assigns the score for the loss(error) level as the model predicts a possibility value
optimizer = optim.Adam(model.parameters(), lr=0.001)
#change the weight after there's an error
epochs = 7 #How many times the model trains through the dataset

for epoch in range(epochs): #loop the dataset for the times of the epoch
    model.train() #start the train of the model
    running_loss = 0.0 #prepare variable for running loss
    total_wrong_train = 0 #store times of wrong predictions in trainset
    total_images_train = 0 #store total times it has trained
    for batchindex, batch in enumerate(train_dataloader): #loop through the train data, use enumerate to get index
        inputs = batch['pixel_values'].to(device) #take in the pixel values
        labels = batch['labels'].to(device) #take the batches
        optimizer.zero_grad() #discard old mistakes
        outputs = model(inputs) #output the prediction
        loss = criterion(outputs, labels) #use CEL to give a score for prediction
        loss.backward() #calculate weight gradient
        optimizer.step() #change the weight
        _, predicted = torch.max(outputs.data, 1) #change it to a guess of 0 and 1(cat and dog)
        wrong_predictions = (predicted != labels).sum().item() #calculate the total time of wrong predictions
        total_wrong_train += wrong_predictions #add the wrong ones to the variables
        total_images_train += labels.size(0) #calculate the total it trained through
        running_loss += loss.item() #add up the loss
        if batchindex % 50 == 0:
            batch_error = (wrong_predictions / labels.size(0)) * 100
            print(
                f"Epoch [{epoch + 1}/{epochs}], Step [{batchindex}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Batch Train Error: {batch_error:.1f}%")
        #output the result after every 50 batches to supervise
        train_error_rate = (total_wrong_train / total_images_train) * 100 #calculate the precentage
        avg_train_loss = running_loss / len(train_dataloader) #calculate the average loss

#Validations and checking error rate
    model.eval() #freeze the network
    total_wrong_test = 0
    total_images_test = 0
    val_loss = 0.0
    #initialize the variables for testset
    with torch.no_grad(): #stop tracking math histories
        for batch in test_dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
    #Take in labels and images from testset
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    #calculate test loss
            # Track validation error rate
            _, predicted = torch.max(outputs, 1)
            total_wrong_test += (predicted != labels).sum().item()
            total_images_test += labels.size(0)

    test_error_rate = (total_wrong_test / total_images_test) * 100
    avg_val_loss = val_loss / len(test_dataloader)
    #calculate percentage of wrong predictions
    print(f"\n=== Epoch {epoch + 1} Summary ===")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Error Rate: {train_error_rate:.1f}%")
    print(f"Test Loss:   {avg_val_loss:.4f} | Test Error Rate:   {test_error_rate:.1f}%")
    print("=========================\n")
    #print summary of performance
print('Finished training!')
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")
#Save the model for later prediction