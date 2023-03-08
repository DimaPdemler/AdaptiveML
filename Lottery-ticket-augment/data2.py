from methods2 import *

batch_size = 64
num_classes = 10
learning_rate = 0.0003
weight_decay = .0001

#Loading the dataset and preprocessing, will add more datsets overtime
def get_data(dataset = 'MNIST', augment = False, validation = False):
    if dataset == 'MNIST':
        if not augment:
            train_dataset = torchvision.datasets.MNIST(root = './data',
                                                    train = True,
                                                    transform = transforms.Compose([
                                                            transforms.ToTensor(),]),
                                                    download = True)
            test_dataset = torchvision.datasets.MNIST(root = './data',
                                                    train = False,
                                                    transform = transforms.Compose([
                                                            transforms.ToTensor(),]),
                                                    download=True)
            test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True)

            if validation:
                lengths = [int(len(train_dataset) * .8), len(train_dataset) - int(len(train_dataset) * .8)]
                MNIST_train_dataset, MNIST_val_dataset = torch.utils.data.random_split(train_dataset, lengths)


                MNIST_train_loader = torch.utils.data.DataLoader(dataset = MNIST_train_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = True)

                MNIST_val_loader = torch.utils.data.DataLoader(dataset = MNIST_val_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = True)

                return MNIST_train_loader, MNIST_val_loader, test_loader

            else:
                train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = True)
                return train_loader, test_loader
            
            
        else:
            train_dataset = torchvision.datasets.MNIST(root = './data',
                                                    train = True,
                                                    transform = transforms.Compose([
                                                            transforms.RandomVerticalFlip(p=1),
                                                            transforms.RandomInvert(p=1),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                                    download = True)

            test_dataset = torchvision.datasets.MNIST(root = './data',
                                                    train = False,
                                                    transform = transforms.Compose([
                                                            transforms.RandomVerticalFlip(p=1),
                                                            transforms.RandomInvert(p=1),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                                    download=True)


            test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True)

            if validation:
                lengths = [int(len(train_dataset) * .8), len(train_dataset) - int(len(train_dataset) * .8)]
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths)


                train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = True)

                val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = True)

                return train_loader, val_loader, test_loader

            else:
                train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                        batch_size = batch_size,
                                                        shuffle = True)
                return train_loader, test_loader

    else:
        return None
