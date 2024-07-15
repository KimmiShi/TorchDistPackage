import torch

def my_ln(x, dim=-1):
    means = torch.mean(x, dim, keepdim=True)
    x_sub_mean = x - means
    squared_err = torch.square(x_sub_mean)
    var_sum = torch.sum(squared_err, dim, keepdim=True)
    std = torch.sqrt(var_sum/x.shape[dim])
    # var = torch.mean(squared_err, dim, keepdim=True)
    # std = torch.sqrt(var)

    # std =  torch.std(x, dim ,keepdim=True, correction=0)
    out = x_sub_mean / std
    return out
class MyLayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(MyLayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Define input tensor
input_tensor = torch.randn(2, 3, 4)  # Shape: (batch_size=2, sequence_length=3, dimension=4)

# Apply LayerNorm along the last dimension
layer_norm = torch.nn.LayerNorm(input_tensor.size()[2], elementwise_affine=False, bias=False)  # Normalize along dimension 2 (dimension=4)
output = layer_norm(input_tensor)

print("Input tensor:")
print(input_tensor)
print("\nOutput tensor after LayerNorm:")
print(output)

output1 = my_ln(input_tensor)
print("\nOutput tensor after my_ln:")
print(output1)

my_ln = MyLayerNorm(input_tensor.shape[2])
output2 = my_ln(input_tensor)
print("\nOutput tensor after MyLayerNorm:")
print(output2)
