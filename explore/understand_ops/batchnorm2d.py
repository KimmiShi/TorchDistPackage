import torch
import torch.nn as nn

def bn2d(x,eps=1e-5,):
    mean = torch.mean(x, dim=(0,2,3), keepdim=True)
    std = torch.std(x, dim=(0,2,3), keepdim=True, correction=0)

    # var = torch.mean(torch.square(x-mean), dim=(0,2,3), keepdim=True)
    # std = torch.sqrt(var+eps)
    out = (x-mean)/std
    return out

# https://d2l.ai/chapter_convolutional-modern/batch-norm.html
# def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
def batch_norm(X, eps=1e-5,):#gamma, beta, moving_mean, moving_var, , momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    # if not torch.is_grad_enabled():
    #     # In prediction mode, use mean and variance obtained by moving average
    #     X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    # else:
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:
        # When using a fully connected layer, calculate the mean and
        # variance on the feature dimension
        mean = X.mean(dim=0)
        var = ((X - mean) ** 2).mean(dim=0)
    else:
        # When using a two-dimensional convolutional layer, calculate the
        # mean and variance on the channel dimension (axis=1). Here we
        # need to maintain the shape of X, so that the broadcasting
        # operation can be carried out later
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    # In training mode, the current mean and variance are used
    X_hat = (X - mean) / torch.sqrt(var + eps)
    return X_hat
    #     # Update the mean and variance using moving average
    #     moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
    #     moving_var = (1.0 - momentum) * moving_var + momentum * var
    # Y = gamma * X_hat + beta  # Scale and shift
    # return Y, moving_mean.data, moving_var.data

x=torch.rand(2,3,224,224)
eps=1e-8
out = bn2d(x,eps=eps)

out_dl = batch_norm(x,eps=eps)

torchbn = nn.BatchNorm2d(x.shape[1], affine=False, momentum=0., eps=eps)
outg = torchbn(x)

# print(out-outg)
print(torch.allclose(out,outg))
print(torch.allclose(out_dl,outg))
print(torch.allclose(out_dl,out))


