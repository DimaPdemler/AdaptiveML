from data2 import *
from methods2 import *
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
location = '/pvc-dmitridemler/'
train_loader, test_loader = get_data('MNIST', augment = True, validation=False)

# scores = []
# for i in range(1,5):
#     print(str(i))
#     model = ResNet50(num_classes = 10,channels = 1).to(device)
#     model.load_state_dict(torch.load(location + 'SA_experiment_18_t_' + str(i)))
#     prune.global_unstructured(model.parameters_to_prune(),pruning_method=prune.L1Unstructured,amount=0)
#     sparsity_print(model)
#     scores.append(test(model, test_loader, device))
#     print('')

#     print(scores)
# def find_ticket(model, name, location, train_loader, test_loader, start_iter = 0, end_iter = 30, num_epochs = 90, learning_rate = .001, prune_amount = .2, k = 1):
  
model2= ResNet50(num_classes = 10,channels = 1).to(device)
find_ticket( model2,'ResNet-50-MNIST-nam', train_loader, test_loader, num_epochs = 10, end_iter=6)
  







def sample_amount(model,t,T):
  amount = []
  prev_zero = float(torch.sum(model.parameters_to_prune()[0][0].weight == 0))
  prev_total = float(model.parameters_to_prune()[0][0].weight.nelement())
  for module, _ in model.parameters_to_prune()[1:]:
    zero = float(torch.sum(module.weight == 0))
    total = float(module.weight.nelement())
    layer_sparsity = (prev_zero+zero) / (prev_total+total)
    amount.append(get_amount((layer_sparsity * ((T-t) / T) / 50),torch.where(module.weight.sum(dim=tuple(range(1, module.weight.dim()))) == 0)[0].size()[0]))
    prev_zero = zero
    prev_total = total
  return amount


# def simulated_annealing(model, device, train_loader, val_loader, name, d = 5, T = 30, prune_amount = .2, num_epochs = 20, structure_weight = 1, error_weight = 5):
  
#   model.to(device)

#   current_model = model.to(device)
#   current_loss = loss(model, val_loader, error_weight, structure_weight, device) #CHANGE THIS TO VALIDATION LOSS
#   losses = []
#   for t in range(1,T+1):

#     amount = sample_amount(current_model,t,T)

#     #Create next model
#     next_model = ResNet50(num_classes = 10,channels = 1).to(device)
#     next_model.load_state_dict(torch.load(location + 'ResNet-50-Fashion_iter30'))
#     prune.global_unstructured(next_model.parameters_to_prune(),pruning_method=prune.L1Unstructured,amount=0)
#     amount = sample_amount(next_model,1,10)
#     sparsity_print(next_model)
#     grow(next_model, amount, device)
#     prune.global_unstructured(next_model.parameters_to_prune(),pruning_method=prune.L1Unstructured,amount=0)
#     sparsity_print(next_model)


#     optimizer = torch.optim.Adam(next_model.parameters(), lr=learning_rate)
#     train(next_model, train_loader,val_loader,num_epochs = num_epochs, optimizer = optimizer)
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


#
#
# model.load_state_dict(torch.load(location + 'SA_experiment_15_t_21'))
# prune.global_unstructured(model.parameters_to_prune(),pruning_method=prune.L1Unstructured,amount=0)


# zero = total = 0
# for module, _ in model.parameters_to_prune():
#     zero += float(torch.sum(module.weight))


#     print('Number of Zero Weights:', zero)
#     print('Total Number of Weights:', total)
#     print('Sparsity with growth:', zero/total)
#     out_channels = mask.sum(dim=(0,2,3))
#     for idx, val in enumerate(out_channels):
#         if val == 0:
#             omega.append(idx)