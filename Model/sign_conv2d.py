import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def _pair(inputs):
    if isinstance(inputs, int):
        outputs = (inputs, inputs)
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) != 2:
            raise ValueError("Length of parameters should be TWO!")
        else:
            outputs = tuple(int(item) for item in inputs)
    else:
        raise TypeError("Not proper type!")

    return outputs


class SignConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample, use_bias):
        super(SignConv2d, self).__init__()

        if (kernel_size, stride) not in [(9, 4), (5, 2), (3, 1)]:
            raise ValueError("This pair of parameters (kernel_size, stride) has not been checked!")


        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.upsample = bool(upsample)
        self.use_bias = bool(use_bias)


        if self.upsample:
            self.weight = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels, *self.kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("bias", None)


        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            init.xavier_normal_(self.weight)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, inputs):
        up = self.kernel_size[0]//2
        down = (self.kernel_size[0]-self.stride[0]) - self.kernel_size[0]//2
        left = self.kernel_size[1]//2
        right = (self.kernel_size[1]-self.stride[1]) - self.kernel_size[1]//2

        if self.upsample:
            outputs = F.conv_transpose2d(inputs, self.weight, self.bias, self.stride, 0)
            outputs = outputs[:, :, up:-down, left:-right]
        else:
            inputs = F.pad(inputs, [up, down, left, right], "constant", 0)
            outputs = F.conv2d(inputs, self.weight, self.bias, self.stride, 0)

        return outputs
