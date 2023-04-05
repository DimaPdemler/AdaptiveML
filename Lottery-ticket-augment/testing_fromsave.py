from data2 import *
from methods2 import *
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
location = '/pvc-dmitridemler/3-14-23/'
train_loader, test_loader = get_data('MNIST', augment = True, validation=False)

model3= ResNet50(num_classes = 10,channels = 1).to(device)

# Change this to whatever location you saved the model
checkpoint = torch.load(location + 'ResNet-50-MNIST-nam_iter6')

model3.load_state_dict(checkpoint)
