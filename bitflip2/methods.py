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
from torchvision.models import resnet50, resnet34, resnet18, wide_resnet50_2
import gc
from data import *

#Standard Training Loop: LR, Weight Decay, etc. are all the same as "One Ticket to Rule them All"
def train(model,train_loader,num_epochs, lr = .0003, weight_decay = .0001, gamma = .1, milestones = [50,65,80]):
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  cost = nn.CrossEntropyLoss()
  scheduler = MultiStepLR(optimizer, milestones=milestones, gamma= gamma)
  total_step = len(train_loader)
  for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(train_loader):  
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          loss = cost(outputs, labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if (i+1) % 400 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
      scheduler.step()

#Standard Test
def test(model, test_loader):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  # Test the model
  model.eval()
  model.to(device)
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

  print('Test Accuracy:', accuracy)
  return accuracy

#Returns prunable parameters for global pruning
def get_parameters_to_prune(model):
  parameters_to_prune = []
  for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
      parameters_to_prune.append((module, 'weight'))
  return tuple(parameters_to_prune)

#Prints Model Sparsity
def sparsity_print(model):
  prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
  zero = total = 0
  for module, _ in get_parameters_to_prune(model):
    zero += float(torch.sum(module.weight == 0))
    total += float(module.weight.nelement())
  print('Number of Zero Weights:', zero)
  print('Total Number of Weights:', total)
  print('Sparsity', zero/total)
  #TODO: Implement Node Sparsity
  return zero, total

def LotteryTicketRewinding(model, name, path, train_loader, start_iter = 0, end_iter = 30, num_epochs = 90, k = 1, amount = .2):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  #Create Rewind Weights
  try:
    model.load_state_dict(torch.load(path + name + '_RewindWeights' + '_' + str(k)))
  except:
    train(model, train_loader,num_epochs = k)  #Save Kth epoch model
    torch.save(model.state_dict(), path + name + '_RewindWeights' + '_' + str(k))

  model_rewind = copy.deepcopy(model).to(device)

  #Load Up Model at Start Iteration
  if start_iter == 0:
    train(model, train_loader,num_epochs = num_epochs - k)
  else:
    model.load_state_dict(torch.load(path + name + '_iter' + str(start_iter)))

  #Lottery Ticket Rewinding: Prune, Rewind, Train
  for i in range(start_iter,end_iter):
    print('LTR Iteration:', i+1)
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=amount)
    #Rewind Weights
    for idx, (module, _) in enumerate(get_parameters_to_prune(model)):
      with torch.no_grad():
        module_rewind = get_parameters_to_prune(model_rewind)[idx][0]
        module.weight_orig.copy_(module_rewind.weight)
    train(model, train_loader,num_epochs = num_epochs)
    sparsity_print(model)
    torch.save(model.state_dict(), path + name + '_iter' + str(i+1))

def simulated_annealing(model, model_type, name, path, rewind_name, train_loader, val_loader, start_iter = 0, end_iter = 30, num_epochs = 20, amount = .2, prune_iters = 3, d = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model = model
    prune.global_unstructured(get_parameters_to_prune(current_model),pruning_method=prune.L1Unstructured,amount=0)

    model_rewind =resnet50()
    model_rewind.fc = nn.Linear(2048, 10)
    model_rewind.load_state_dict(torch.load(path + rewind_name))
    model_rewind = model_rewind.to(device)
    prune.global_unstructured(get_parameters_to_prune(model_rewind),pruning_method=prune.L1Unstructured,amount=0)

    current_loss = weighted_loss(current_model, val_loader, 1, 1)
    for i in range(start_iter,end_iter):
        print('SA  Iteration:', i+1)
        if model_type == 'resnet18':
            next_model = resnet18()
            next_model.fc = nn.Linear(512, 10)
        elif model_type == 'resnet50':
            next_model = resnet50()
            next_model.fc = nn.Linear(2048, 10)
        elif model_type == 'resnet50w':
            next_model = wide_resnet50_2()
            next_model.fc = nn.Linear(2048, 10)
        elif model_type == 'resnet34':
            next_model = resnet34()
            next_model.fc = nn.Linear(512, 10)
        
        next_model.to(device)
        prune.global_unstructured(get_parameters_to_prune(next_model),pruning_method=prune.L1Unstructured,amount=0)
        next_model.load_state_dict(current_model.state_dict())

        #REWIND WEIGHTS
        for idx, (module, _) in enumerate(get_parameters_to_prune(next_model)):
            with torch.no_grad():
                module_rewind = get_parameters_to_prune(model_rewind)[idx][0]
                module.weight_orig.copy_(module_rewind.weight)

        #GROW!
        for module, _ in get_parameters_to_prune(next_model):
            pruned_weights = torch.nonzero(torch.where(module.get_buffer('weight_mask') == 0, 1, 0))  #get indices of pruned weights in sparsity mask
            for idx in pruned_weights[:int(len(pruned_weights)*amount*((i+1)**(-2)))]:   #int(len(pruned_weights)*amount*((end_iter-i)/end_iter))
                module.get_buffer('weight_mask')[tuple(idx)] = torch.tensor([1]).float().to(device)
            prune.custom_from_mask(module,'weight', torch.ones(module.weight.size(), device = device))  #saves the changes of the weight mask

        train(next_model,train_loader,num_epochs)
        prune.global_unstructured(get_parameters_to_prune(next_model),pruning_method=prune.L1Unstructured,amount=amount)
        train(next_model,train_loader,num_epochs)
        
        next_loss = weighted_loss(next_model, val_loader, 1, 1)
        #Keep model with Acceptance Probability
        if next_loss < current_loss or np.random.uniform() < math.exp(-1*(next_loss - current_loss) / (d / math.log(i+2)) ):
            
            current_model = next_model
        #Else, keep current model
        torch.save(current_model.state_dict(), path + name + '_iter' + str(i+1))

#Custom Loss function to account for model size and performance
def weighted_loss(model, val_loader, error_weight, structure_weight):
    zero, total = sparsity_print(model)
    structural_loss = (1 - zero / total)
    error = 1 - test(model, val_loader)
    print('Structure Loss: ', structural_loss, ', Error: ', error)
    return structure_weight * structural_loss + error_weight * error

"""
from methods import *
from data import *
path = '/pvc-lukemcdermott/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zeros = []
totals = []
acc = []

train_loader, test_loader = get_data('CIFAR10')
for i in range(1,41):
    #model = wide_resnet50_2(weights = None)
    model = resnet50()
    model.fc = nn.Linear(2048, 10)
    #model.fc = nn.Linear(512, 10)
    model = model.to(device)
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
    model.load_state_dict(torch.load(path + 'SA-RN50-exp2_iter' + str(i)))
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
    zero, total = sparsity_print(model)
    zeros.append(zero)
    totals.append(total)
    acc.append(test(model,test_loader))

print('zeros:',zeros)
print('totals:', totals)
print('acc:',acc)
"""


def old_sa(model, model_type, name, path, rewind_name, train_loader, val_loader, start_iter = 0, end_iter = 30, num_epochs = 20, amount = .2, prune_iters = 3, d = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model = model
    prune.global_unstructured(get_parameters_to_prune(current_model),pruning_method=prune.L1Unstructured,amount=0)

    model_rewind =resnet50()
    model_rewind.fc = nn.Linear(2048, 10)
    model_rewind.load_state_dict(torch.load(path + rewind_name))
    model_rewind = model_rewind.to(device)
    prune.global_unstructured(get_parameters_to_prune(model_rewind),pruning_method=prune.L1Unstructured,amount=0)

    current_loss = weighted_loss(current_model, val_loader, 1, 1)
    for i in range(start_iter,end_iter):
        print('SA  Iteration:', i+1)
        if model_type == 'resnet18':
            next_model = resnet18()
            next_model.fc = nn.Linear(512, 10)
        elif model_type == 'resnet50':
            next_model = resnet50()
            next_model.fc = nn.Linear(2048, 10)
        elif model_type == 'resnet50w':
            next_model = wide_resnet50_2()
            next_model.fc = nn.Linear(2048, 10)
        elif model_type == 'resnet34':
            next_model = resnet34()
            next_model.fc = nn.Linear(512, 10)
        
        next_model.to(device)
        prune.global_unstructured(get_parameters_to_prune(next_model),pruning_method=prune.L1Unstructured,amount=0)
        next_model.load_state_dict(current_model.state_dict())

        #REWIND WEIGHTS
        for idx, (module, _) in enumerate(get_parameters_to_prune(next_model)):
            with torch.no_grad():
                module_rewind = get_parameters_to_prune(model_rewind)[idx][0]
                module.weight_orig.copy_(module_rewind.weight)

        #GROW!
        for module, _ in get_parameters_to_prune(next_model):
            pruned_weights = torch.nonzero(torch.where(module.get_buffer('weight_mask') == 0, 1, 0))  #get indices of pruned weights in sparsity mask
            for idx in pruned_weights[:int(len(pruned_weights)*amount*((i+1)**(-2)))]:   #int(len(pruned_weights)*amount*((end_iter-i)/end_iter))
                module.get_buffer('weight_mask')[tuple(idx)] = torch.tensor([1]).float().to(device)
            prune.custom_from_mask(module,'weight', torch.ones(module.weight.size(), device = device))  #saves the changes of the weight mask

        train(next_model,train_loader,num_epochs)
        prune.global_unstructured(get_parameters_to_prune(next_model),pruning_method=prune.L1Unstructured,amount=amount)
        train(next_model,train_loader,num_epochs)
        
        next_loss = weighted_loss(next_model, val_loader, 1, 1)
        #Keep model with Acceptance Probability
        if next_loss < current_loss or np.random.uniform() < math.exp(-1*(next_loss - current_loss) / (d / math.log(i+2)) ):
            
            current_model = next_model
        #Else, keep current model
        torch.save(current_model.state_dict(), path + name + '_iter' + str(i+1))

def grow(model, amount, device):
  previous_module = get_parameters_to_prune(model)[0][0]

  for i, (module, name) in enumerate(get_parameters_to_prune(model)[1:]):
    mask = module.get_buffer('weight_mask').data
    #Pick Suitable Locations
    omega = []
    out_channels = mask.sum(dim=(0,2,3))
    for idx, val in enumerate(out_channels):
      if val == 0:
        omega.append(idx)

    indices = np.random.choice(omega, size = min(len(omega), amount[i]), replace = False)
    
    #Grow at these indices
    #print(i, module.weight.size())
    module.get_buffer('weight_mask')[:,indices] = 1
    prune.custom_from_mask(module,'weight', torch.ones(module.weight.size(), device = device))
    
    previous_module.get_buffer('weight_mask')[indices,:] = 1
    prune.custom_from_mask(previous_module,'weight', torch.ones(previous_module.weight.size(), device = device))  
    previous_module = module