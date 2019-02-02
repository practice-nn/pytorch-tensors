#%%
import torch
print(torch.__version__)

#%%
tensor_array = torch.Tensor([[1, 2], [4, 5]])
tensor_array

#%%
tensor_unitialized = torch.Tensor(3, 3) # unitialized
torch.numel(tensor_unitialized) # number of elements

#%%
tensor_unitialized # will throw error Overflow when unpacking as memory not yet allocated

#%%
torch_initialized = torch.rand(2, 3)
torch_initialized # random numbers

#%%
tensor_int = torch.randn(5, 3).type(torch.IntTensor) # just integers
tensor_int # has dtype=torch.int32

#%%
tensor_long = torch.LongTensor([1.0, 2.0, 3.0])
tensor_long # converted to longs even though entered as floats

#%%
tensor_byte = torch.ByteTensor([0, 261, 1, -5])
tensor_byte # has dtype=torch.uint8, so numbers past 255 loop, e.g.: 261 ends up as 5

#%%
tensor_ones = torch.ones(10) # init to 1. floats
tensor_ones

#%%
tensor_zeros = torch.zeros(10) # init zero vector (floats)
tensor_zeros

#%%
tensor_eye = torch.eye(3) # identity matrix
tensor_eye

#%%
non_zero = torch.nonzero(tensor_eye)
non_zero # finds all index positions of non-zero elements

#%%
tensor_ones_shape_eye = torch.ones_like(tensor_eye)
tensor_ones_shape_eye

#%%
# inplace operation something_ the underscore means the current tensor is getting altered, no _ postfix returns a new tensor
initial_tensor = torch.rand(3, 3)
initial_tensor.fill_(3) # overwrites all values with 3.

#%%
# out of place operation
initial_tensor.fill(3) # throws an error because it doesn't exist - not all in-place ops have out-of-place equivalents that make sense

#%%
new_tensor = initial_tensor.add(4)
new_tensor # ads 4 to a tensor copy

#%%
initial_tensor # same as before

#%%
initial_tensor.add_(5) # modifies in-place
initial_tensor # changed forever


#%%
initial_tensor # same as before, unchanged

#%%
import numpy as np
numpy_arr = np.array([1, 2, 3])
numpy_arr

#%%
tensor = torch.from_numpy(numpy_arr) # great integration
tensor # dtype=torch.int32

#%%
numpy_from_tensor = tensor.numpy()
numpy_from_tensor

#%%
numpy_arr[1] = 4
numpy_arr # same memory as the tensor!

#%%
tensor # same memory as the numpy array!

#%%
numpy_from_tensor # same memory as the tensor!

#%%
initial_tensor = torch.rand(2, 3)
initial_tensor

#%%
initial_tensor[0, 2]


#%%
initial_tensor[:, 1:] # last two columns using brackets notation

#%%
initial_tensor.size()

#%%
initial_tensor.shape # same as size()

#%%
resized_tensor = initial_tensor.view(6) # same memory as the original tensor
resized_tensor.shape

#%%
resized_tensor

#%%
initial_tensor[0, 2] = 0.1111
resized_tensor # updates the view as well

#%%
resized_tensor = initial_tensor.view(3, 2)
resized_tensor.shape # different shape

#%%
resized_matrix = initial_tensor.view(-1, 2) # infers the missing shape parameter, in this case it picks up 3
resized_matrix.shape # 3 by 2

#%%
resized_matrix = initial_tensor.view(-1, 5) # incompatible reshape ops throw errors


#%%
initial_tensor

#%%
sorted_tensor, sorted_indices = torch. sort(initial_tensor)
sorted_tensor
#%%
sorted_indices

#%%
sorted_tensor, sorted_indices = torch.sort(initial_tensor, dim=0)
sorted_tensor

#%%
sorted_indices

#%%
