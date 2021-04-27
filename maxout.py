import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable
from torch.autograd import Function

class Maxout(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        x = input
        max_out=4    #Maxout Parameter
        kernels = x.shape[1]  # to get how many kernels/output
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x= x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices=indices
        ctx.max_out=max_out
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input1,indices,max_out= ctx.saved_variables[0],Variable(ctx.indices),ctx.max_out
        input=input1.clone()
        for i in range(max_out):
            a0=indices==i
            input[:,i:input.data.shape[1]:max_out]=a0.float()*grad_output
        return input

nf = 16
palette = 64

class MaxoutCNN(nn.Module):

    def __init__(self): #Input: (B, 3, 256, 256), Output: (B, palette * 3)
        super(MaxoutCNN, self).__init__()
        self.conv1 = torch.nn.Sequential( #declaring all of our conv layers, Input: (B, 3, 256, 256), Output: (B, nf*8, 1, 1)
            torch.nn.Conv2d(3, nf * 4, 3, 4, 1),
            torch.nn.LeakyReLU(0.2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(nf, nf * 4, 3, 4, 1),
            torch.nn.LeakyReLU(0.2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(nf, nf * 4, 3, 4, 1),
            torch.nn.LeakyReLU(0.2),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(nf, nf * 4, 3, 4, 1),
            torch.nn.LeakyReLU(0.2),
        )
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.linear = torch.nn.Sequential( #declaring all of our linear layers, Input: (B, nf*8), Output: (B, palette * 3)
            torch.nn.Linear(nf, nf),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(nf, palette * 3),
            torch.nn.Tanh(),
        )

    def forward(self, input): #runs input through layers of model, returns output
        conv_output = input
        for i in range(4):
            conv_output = Maxout.apply(self.convs[i](conv_output))
        linear_output = (self.linear(conv_output.view(input.size(0), -1))+1)/2
        return linear_output
