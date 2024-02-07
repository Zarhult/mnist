from __future__ import print_function
import os
import argparse
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image

model_filename = "mnist_cnn.pt"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.25) # drop 25% of the values
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #print(f"Start: {x.size()}")
        x = self.conv1(x)
        #print(f"Conv1: {x.size()}")
        x = F.elu(x)
        x = F.max_pool2d(x, 2)
        #print(f"Max pool: {x.size()}")
        x = self.conv2(x)
        #print(f"Conv2: {x.size()}")
        x = F.elu(x)
        x = F.max_pool2d(x, 2)
        #print(f"Max pool: {x.size()}")
        x = self.conv3(x)
        #print(f"Conv3: {x.size()}")
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        #print(f"Conv4: {x.size()}")
        x = F.elu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_image(img, model, device):
    model.eval()
    # image must be 28x28
    # convert to expected format
    transform=transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    tensor = transform(img)
    tensor = torch.unsqueeze(tensor, 0) # add batch dimension
    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        pred = output.argmax(dim=1, keepdim=True)
    print("Predicted Class:", pred.item())
    return pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--ignore-existing', action='store_true', default=False,
                        help='Ignore existing pretrained model file and train a new one in its place.')
    parser.add_argument('--input-image', type=str, help='Input filename')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)

    # After loading the MNIST dataset
    train_size = int(0.8 * len(dataset1))
    val_size = len(dataset1) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset1, 
                                                               [train_size, 
                                                                val_size])

    # Create data loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    # Load model if exists, otherwise train it
    if not os.path.exists(model_filename) or args.ignore_existing:
        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, val_loader)
            if args.input_image:
                img = Image.open(args.input_image)
                test_image(img, model, device)
            scheduler.step()
    else:
        model = Net().to(device)
        model.load_state_dict(torch.load(model_filename, map_location="cpu"))
        if args.input_image:
            img = Image.open(args.input_image)
            test_image(img, model, device)
        else:
            print("Model already exists, use --ignore-existing to train a new one.")

    if args.save_model:
        torch.save(model.state_dict(), model_filename)


if __name__ == '__main__':
    main()
