from os import path

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets, transforms, models

from PIL import Image


def load_model(model, training_dataloader, model_path):
    """
    Function for loading a trained model. If no model exists yet, it will train and save one.
    
    Parameters:
        model (nn.Module): Model to be trained
        training_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset
        model_path (str): Filepath to save the trained model
        
    Returns:
        nn.Module: Loaded or trained model
    """
    if path.exists(model_path):
        return torch.load(model_path)
    else:
        return train_model(model, training_dataloader, model_path)


def train_model(model, training_dataloader, model_path):
    """
    Function for training a model. The epochs can be changed 
    by changing the epochs variable. The training loss will be printed for each epoch.
    When training has completed the function saves the model to the specified path
    and returns the trained model.
    
    Parameters:
        model (nn.Module): Model to be trained
        training_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset
        model_path (str): Filepath to save the trained model
    
    Returns:
        nn.Module: Trained model
    """
    print("Traning model...")

    model.train()
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 2),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    model.to(device)

    epochs = 1
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(training_dataloader):
            # print(ii)

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(training_dataloader)}")

    # Save the trained model to a file for future use
    torch.save(model, model_path)
    return model


def test_model(model, test_dataloader):
    """
    Test a trained model on a provided test dataset.
    Prints out the test loss and test accuracy.
    
    Parameters:
        model (nn.Module): Trained model
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loss = 0
    accuracy = 0
    model.eval()
    # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(test_dataloader):.3f}.. "
          f"Test accuracy: {accuracy/len(test_dataloader):.3f}")


def predict_image(image, model, classes, transform):
    """
    Predict the class of an image using a trained model

    Parameters:
        image (PIL.Image.Image): Image to predict the class of
        model (nn.Module): Trained model
        classes (list): List of class names, in the same order as the model's output
        transform (callable): Preprocessing function to apply to the image
        
    Returns:
        tuple: Tuple containing the class index (int) and class name (str) of the predicted class
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Apply the same preprocessing as was applied to the training and test data
    image_tensor = transform(image).to(device)
    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze_(0)

    model.to(device)
    model.eval()
    with torch.no_grad():
        # Get the class probabilities
        output = model(image_tensor)
        probs = torch.exp(output).to(device)
        # Find the class with the highest predicted probability
        _, predicted_class = probs.max(dim=1)

    return classes[predicted_class], probs[0][predicted_class]


def test_image(image_path, model):
    """
    Test the model with a single image and display the result

    Parameters:
        image_path (str): Path to the image file
        model (nn.Module): Trained model
    """
    img = Image.open(image_path)
    class_label, prob = predict_image(
        img, model, train_data.classes, test_transforms)
    print(f"The image is a {class_label} with probability {prob}")

    plt.imshow(img)  # type: ignore
    plt.title(
        f'Predicted class: {class_label}\nProbability: {round(prob.item() * 100, 2)}%')
    plt.show()


# Path to the training and testing data
data_dir = 'C:/Users/sam_y/Documents/PyTorch Data/Cat_Dog_data'
#   |- test
#       |- cat
#       |- dog
#   |- train
#       |- cat
#       |- dog

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(
    data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(
    data_dir + '/test', transform=test_transforms)

trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
testloader = DataLoader(test_data, batch_size=64)

criterion = nn.NLLLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
train_model(model, trainloader, 'catsanddogs_densenet121.pth')
# model = load_model(model, trainloader, 'catsanddogs_densenet121.pth')
model.to(device)

test_model(model, testloader)


test_image('cat_test.jpg', model)
test_image('cat_test_2.jpg', model)
test_image('dog_test.jpg', model)