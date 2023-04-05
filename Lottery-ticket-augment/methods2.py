import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch import optim
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import copy
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet50, ResNet50_Weights
import gc

from data2 import *

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

        self.parameters_to_prune = [
            (self.conv1, 'weight'),
            (self.conv2, 'weight'),
            (self.conv3, 'weight'),
        ]

        for module, name in self.parameters_to_prune:
          mask = torch.ones(np.shape(module.weight))
          # Should I remove the "//5"?
          mask[:,:(np.shape(mask)[1]//5),:] = 0 #Prune input channel
          mask[:(np.shape(mask)[0]//5),:,:] = 0 #Prune output channel
          prune.custom_from_mask(module, name, mask)

        

        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x
        
    

        

        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # Delete the maksing stuff and instead initialize the mask with all 1s
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        mask = torch.ones(np.shape(self.conv1.weight))
        # mask[:(np.shape(mask)[0]//5),:,:] = 0 #Prune output channel's new growth
        prune.custom_from_mask(self.conv1, 'weight', mask)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

    def parameters_to_prune(self):
        p = [(self.conv1, 'weight')]
        for layer in [self.layer1,self.layer2, self.layer3,self.layer4]:
         for module in layer:
           p = p + module.parameters_to_prune
        return  p
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

# Define relevant variables for the ML task
batch_size = 512
num_classes = 10
learning_rate = 0.0003
weight_decay = .0001

def sparsity_print(model):
  zero = total = 0
  for module, _ in model.parameters_to_prune():
    zero += float(torch.sum(module.weight == 0))
    total += float(module.weight.nelement())
  print('Number of Zero Weights:', zero)
  print('Total Number of Weights:', total)
  print('Sparsity:', zero/total)

# Why do you habe test_loader here?
def train(model,train_loader,test_loader,num_epochs,optimizer):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  cost = nn.CrossEntropyLoss()
  scheduler = MultiStepLR(optimizer, milestones=[50,65,80], gamma=.1)
  total_step = len(train_loader)
  for epoch in range(num_epochs):
      
      for i, (images, labels) in enumerate(train_loader):  
          images = images.to(device)
          labels = labels.to(device)
          
          #Forward pass
          outputs = model(images)
          loss = cost(outputs, labels)
            
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
              
          if (i+1) % 400 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
              
      scheduler.step()

def test(model, test_loader, device):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  # Test the model
  model.eval()
  with torch.no_grad():
      correct = 0
      total = 0
      for i, (images, labels) in enumerate(test_loader): 
          images, labels = images.to(device), labels.to(device)
          test_output = model(images)
          pred_y = torch.max(test_output, 1)[1].data.squeeze()
          correct += (pred_y == labels).sum().item()
          total += labels.size(0)
      accuracy = correct / total

  print('Test Accuracy of the model on the 10000 test images: ', accuracy)
  return accuracy

def find_ticket(model, name, location, train_loader, test_loader, start_iter = 0, end_iter = 30, num_epochs = 90, learning_rate = .001, prune_amount = .2, k = 1):
  
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
  accuracy = []
  #If training has already been saved
  try:
    model.load_state_dict(torch.load(location + name + '_RewindWeights' + '_' + str(k)))
  except:
    train(model, train_loader,test_loader,num_epochs = k, optimizer = optimizer)  #Save Kth epoch model
    torch.save(model.state_dict(), location + name + '_RewindWeights' + '_' + str(k))

  model_rewind = ResNet50(num_classes = 10,channels = 1) #Save as separate model so we can rewind our weights back to this 
  model_rewind.load_state_dict(torch.load(location+ name + '_RewindWeights' + '_' + str(k)))

  if start_iter == 0:
    train(model, train_loader,test_loader,num_epochs = num_epochs - k, optimizer = optimizer) #Finish off training
  else:
    model.load_state_dict(torch.load(location + name + '_iter' + str(start_iter)))

  for i in range(start_iter, end_iter): 
    print('Rewinding Iter:', i)
    #Prune
    prune.global_unstructured(model.parameters_to_prune(),pruning_method=prune.L1Unstructured,amount=prune_amount,)
    
    #Rewind Weights
    for idx, (module, _) in enumerate(model.parameters_to_prune()):
      with torch.no_grad():
        module_rewind = model_rewind.parameters_to_prune()[idx][0]
        module.weight_orig.copy_(module_rewind.weight)
    
    
    #Train
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    train(model, train_loader,test_loader,num_epochs = num_epochs, optimizer = optimizer)
    print("testing iter: ", i, "  ", test(model, test_loader, device))
    accuracy.append(test(model, test_loader, device))

    plt.plot(np.arange(len(accuracy)), accuracy)
    plt.show()

    sparsity_print(model)
    print('Saving iteration ', str(i+1))
    torch.save(model.state_dict(), location + name + '_iter' + str(i+1)) 

# def grow(model, amount, device):
#   previous_module = model.parameters_to_prune()[0][0]

#   for i, (module, name) in enumerate(model.parameters_to_prune()[1:]):
#     mask = module.get_buffer('weight_mask').data
#     #Pick Suitable Locations
#     omega = []
#     out_channels = mask.sum(dim=(0,2,3))
#     for idx, val in enumerate(out_channels):
#       if val == 0:
#         omega.append(idx)

#     indices = np.random.choice(omega, size = min(len(omega), amount[i]), replace = False)
    
#     #Grow at these indices
#     #print(i, module.weight.size())
#     module.get_buffer('weight_mask')[:,indices] = 1
#     prune.custom_from_mask(module,'weight', torch.ones(module.weight.size(), device = device))
    
#     previous_module.get_buffer('weight_mask')[indices,:] = 1
#     prune.custom_from_mask(previous_module,'weight', torch.ones(previous_module.weight.size(), device = device))  
#     previous_module = module

def loss(model, val_loader,error_weight, structure_weight, device):
  error = 1-test(model, val_loader, device) #should be validation set, but for now we will use test set

  params = 0
  total_params = 0
  for module,_ in model.parameters_to_prune():
    params += float(torch.sum(module.weight == 0))
    total_params += float(module.weight.nelement())
  structure = 1 - params / total_params  #uses sparsity at the moment
  print('Structure: ', structure,', Error: ', error)
  return structure_weight * structure + error_weight * error 

def acceptance(current_loss, new_loss, t,d):
  p = math.exp(-1*(new_loss - current_loss) / cooling_schedule(t,d))
  return np.random.uniform() < p
    
def cooling_schedule(t,d):
  return d / math.log(t+1)

#returns module count - 1 long array
def sample_amount(model,t,T):
  amount = []
  prev_zero = float(torch.sum(model.parameters_to_prune()[0][0].weight == 0))
  prev_total = float(model.parameters_to_prune()[0][0].weight.nelement())
  for module, _ in model.parameters_to_prune()[1:]:
    zero = float(torch.sum(module.weight == 0))
    total = float(module.weight.nelement())
    layer_sparsity = (prev_zero+zero) / (prev_total+total)
    amount.append(get_amount(((T-t) / (50*T))**2,torch.where(module.weight.sum(dim=tuple(range(1, module.weight.dim()))) == 0)[0].size()[0]))
    prev_zero = zero
    prev_total = total
  return amount

#given the probability of growth, it samples to represent k-many nodes to grow
def get_amount(p,max):
  count = 0
  for _ in range(max):
    if np.random.uniform() < p:
      count += 1
  return count

#Currently 20 epochs for hyper param tuning, but lets turn it up to 90 once we done
# def simulated_annealing(model, device, train_loader, val_loader, name, d = 5, T = 30, prune_amount = .2, num_epochs = 20, structure_weight = 1, error_weight = 5):
  
#   model.to(device)

#   current_model = model.to(device)
#   current_loss = loss(model, val_loader, error_weight, structure_weight, device) #CHANGE THIS TO VALIDATION LOSS
#   losses = []
#   for t in range(1,T+1):

#     amount = sample_amount(current_model,t,T)

#     #Create next model
#     next_model = ResNet50(num_classes = 10,channels = 1).to(device)
#     next_model.load_state_dict(current_model.state_dict())
#     grow(next_model, amount, device)


#     optimizer = torch.optim.Adam(next_model.parameters(), lr=learning_rate)
#     train(next_model, train_loader,val_loader,num_epochs = num_epochs // 2, optimizer = optimizer)
#     prune.global_unstructured(next_model.parameters_to_prune(),pruning_method=prune.L1Unstructured,amount=prune_amount)
#     train(next_model, train_loader,val_loader,num_epochs = num_epochs, optimizer = optimizer)

#     next_loss = loss(next_model, val_loader, error_weight, structure_weight, device) #CHANGE THIS TO VALIDATION LOSS
#     print('Next_Loss at time ', t, ' is: ',next_loss)

#     if acceptance(current_loss, next_loss, t, d):
#       current_model = next_model
#       current_loss = next_loss
#       print('Accepted!')
#     else:
#       print('Not Accepted.')
#     print('Current Loss at end of time ', t, ' is: ', current_loss)

#     losses.append(current_loss)
#     torch.save(current_model.state_dict(), '/pvc-lukemcdermott/SA_' + name + '_t_' + str(t)) 
#   return losses, current_model