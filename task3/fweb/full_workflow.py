import torch.optim
import torch.utils.data
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.utils.data
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


"""
Net
"""
class EmbeddingNet(nn.Module):
    """EmbeddingNet using the specified model in base_model()."""

    def __init__(self, resnet):
        """Initialize EmbeddingNet model."""
        super(EmbeddingNet, self).__init__()
        # Everything excluding the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs =  resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 1024)

    def forward(self, x):
        """Forward pass of EmbeddingNet."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def base_model():
    return EmbeddingNet(models.resnet50(pretrained=True))


class TripletNet(nn.Module):
    """Triplet Network."""

    def __init__(self, embeddingnet):
        """Triplet Network Builder."""
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, a, p, n):
        """Forward pass."""
        # anchor
        embedded_a = self.embeddingnet(a)

        # positive examples
        embedded_p = self.embeddingnet(p)

        # negative examples
        embedded_n = self.embeddingnet(n)

        return embedded_a, embedded_p, embedded_n


"""
Loader
"""
def image_loader(path):
    return Image.open(path.rstrip('\n')).convert('RGB')


class TripletImageLoader(Dataset):
    def __init__(self, base_path, triplets_filename, transform=None, loader=image_loader):
        """
        Image Loader Builder.
        Args:
            base_path: path to food folder
            triplets_filename: A text file with each line containing three images
            transform: To resize and normalize all dataset images
            loader: loader for each image
        """
        self.base_path = base_path
        self.transform = transform
        self.loader = loader

        # load a triplet data
        triplets = []
        for line in open(triplets_filename):
            line_array = line.split(" ")
            triplets.append((line_array[0], line_array[1], line_array[2]))
        self.triplets = triplets

    def __getitem__(self, index):
        """Get triplets in dataset."""
        path1, path2, path3 = self.triplets[index]
        path1 = path1.rstrip('\n')
        path2 = path2.rstrip('\n')
        path3 = path3.rstrip('\n')
        a = self.loader(os.path.join(self.base_path, f"{path1}.jpg"))
        p = self.loader(os.path.join(self.base_path, f"{path2}.jpg"))
        n = self.loader(os.path.join(self.base_path, f"{path3}.jpg"))
        if self.transform is not None:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)
        return a, p, n

    def __len__(self):
        """Get the length of dataset."""
        return len(self.triplets)


def DatasetImageLoader(root, batch_size_train, batch_size_test, batch_size_val):
    """
    Args:
        root: path to food folder
        batch_size_train
        batch_size_test
    Return:
        trainloader: The dataset loader for the training triplets
        testloader: The dataset loader for the test triplets
        valloader:  The dataset loader for the validation triplets
    """

    trainset_mean = torch.Tensor([0.485, 0.456, 0.406])
    trainset_std = torch.Tensor([0.229, 0.224, 0.225])

    # Normalize training set together with augmentation
    transform_train = transforms.Compose([
        transforms.Resize((242,354)),
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
    ])

    # Normalize test set same as training set without augmentation
    transform_val = transforms.Compose([
        transforms.Resize((242, 354)),
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((242, 354)),
        transforms.ToTensor(),
        transforms.Normalize(trainset_mean, trainset_std),
    ])

    # Loading Tiny ImageNet dataset
    print("Loading dataset images...")

    trainset = TripletImageLoader(
        base_path=root, triplets_filename="train_triplets_splitted.txt", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, num_workers=0)

    testset = TripletImageLoader(
        base_path=root, triplets_filename="../handout/test_triplets.txt", transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, num_workers=0)

    valset = TripletImageLoader(
        base_path=root, triplets_filename="validation_triplets_splitted.txt", transform=transform_val)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size_val, num_workers=0)

    return trainloader, testloader, valloader


"""
Train
"""
def train(net, criterion, optimizer, start_epoch, epochs, trainloader, valloader, train_batch_size, val_batch_size, is_gpu=True):
    """
    Training process.
    Args:
        net: Triplet Net
        criterion: TripletMarginLoss
        optimizer: SGD with Nesterov Momentum
        trainloader: training set loader
        valloader: validation set loader
        start_epoch: 0
        epochs: number of training epochs
        is_gpu: True since we train on GPU
    """
    print("Start training...")

    criterion_val = nn.TripletMarginLoss(margin=0.0, p=2, reduction='none')
    net.train()
    
    for epoch in range(start_epoch, epochs + start_epoch):

        running_loss = 0.0
        loss_train = 0.0
        for batch_idx, (data1, data2, data3) in enumerate(trainloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

            # compute output and loss
            embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
            loss = criterion(embedded_a, embedded_p, embedded_n)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print the loss
            running_loss += loss.data

            loss_train_cls = torch.sum(1 * (criterion_val(embedded_a, embedded_p, embedded_n) > 0)) / train_batch_size

            loss_train += loss_train_cls.data

            if batch_idx % 30 == 0:
                print(f"mini Batch Loss: {loss.data}")

        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        loss_train /= len(trainloader)

        print(f"Training Epoch: {epoch + 1} | Loss: {running_loss}")
        print(f"Training Epoch: {epoch + 1} | Class. Loss: {loss_train}")

        # Validation
        net.eval()
        val_loss = 0.0
        for batch_idx, (data1, data2, data3) in enumerate(valloader):
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

            with torch.no_grad():
                # compute output and loss
                embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
                loss_val = torch.sum(1 * (criterion_val(embedded_a, embedded_p, embedded_n) > 0)) / val_batch_size
                val_loss += loss_val.data

        val_loss /= len(valloader)
        print(f"Validation Epoch: {epoch + 1} | Class. Loss: {val_loss}")
        # Go back to training mode, see https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        net.train()

    print("Finished Training...")
    net.eval()

    return net


def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.deterministic=True

    train_triplets_splitted = np.loadtxt("../handout/train_triplets.txt", dtype=str, delimiter=" ")

    train_triplets, val_triplets = train_test_split(train_triplets_splitted, test_size=0.001, random_state=42, shuffle=True)

    np.savetxt("train_triplets_splitted.txt", train_triplets, fmt='%s %s %s')
    np.savetxt("validation_triplets_splitted.txt", val_triplets, fmt='%s %s %s')

    # Initialize the model
    net = TripletNet(base_model())

    print("Initialize CUDA support for TripletNet model...")
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    # Loss function, optimizer and scheduler
    criterion = nn.TripletMarginLoss(margin=5.0, p=2)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=2e-3, nesterov=True)

    val_batch_size = 120
    train_batch_size = 120
    test_batch_size = 120
    path = "../handout/food"

    # Load train, test, validation triplets
    trainloader, testloader, valloader = DatasetImageLoader(path.rstrip("\n"), train_batch_size, test_batch_size, val_batch_size)

    # Train model
    trained_net = train(net, criterion, optimizer, 0, 1, trainloader, valloader, train_batch_size, val_batch_size)

    # Switch the net to evaluation mode
    trained_net.eval()

    pred_test=[]

    # Predict labels 1 or 0 for each test triplet
    for batch_idx, (data1, data2, data3) in enumerate(testloader):

        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

        # wrap in torch.autograd.Variable
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        with torch.no_grad():
            # compute output and loss
            embedded_a, embedded_p, embedded_n = trained_net(data1, data2, data3)

        dist_ap = np.linalg.norm(np.squeeze((embedded_a-embedded_p).cpu().detach().numpy()), ord=2, axis=-1)
        dist_an = np.linalg.norm(np.squeeze((embedded_a-embedded_n).cpu().detach().numpy()), ord=2, axis=-1)

        pred_test.append(1*(dist_ap <= dist_an))

    predicted_labels = np.hstack(pred_test)
    try:
        np.savetxt("predictions.txt", predicted_labels, fmt='%i')
    except:
        np.savetxt("predictions.txt", predicted_labels, fmt='%s')


if __name__ == '__main__':
    main()