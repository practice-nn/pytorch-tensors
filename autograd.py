#%%
import torch
print(torch.__version__)

#%%
tensor = torch.Tensor([[3, 4], [7, 5]])
tensor

#%%
tensor.requires_grad # False by default, computations for this tensor won't be tracked

#%%
tensor.requires_grad_() # implicit enable of tracking computations for this tensor to calculate gradients on the backward pass
tensor.requires_grad

#%%
print(tensor.grad) # accumulates the gradient of the computations w.r.t. this tensor after the backward pass which was used to caluclate the gradients

#%%
print(tensor.grad_fn) # no backward pass performed yet

#%%
out = tensor * tensor

#%%
out.requires_grad # derived from the original tensor

#%%
print(out.grad) # still no gradients


#%%
print(out.grad_fn) # exists because it holds the result of a computation (the mulitiplication) which required gradients, it therefore has a gradient function associated with it

#%%
print(tensor.grad_fn) # no grad_fn because this tensor is NOT the result of any computation

#%%
out = (tensor * tensor).mean()
print(out.grad_fn)

#%%
print(tensor.grad) # still no gradients associated with the original tensor

#%%
out.backward() # compute the gradients w.r.t. the "out" tensor

#%%
print(tensor.grad) # now it exists!


#%%
new_tensor = tensor * tensor
print(new_tensor.requires_grad) # if the tensors in the computation have requires_grad=True the computed output will as well

#%%
with torch.no_grad():
    # stop autograd from tracking history on newly created tensors
    # we require gradient calculation in training phase
    # we turn off calculating gradients when predicting
    new_tensor = tensor * tensor
    print('new_tensor = ', new_tensor)
    print('requires_grad for tensor', tensor.requires_grad) # True; required gradient calculation
    print('requires_grad for new_tensor', new_tensor.requires_grad) # False; does not require gradient calculation because of the torch.no_grad() block we are in

