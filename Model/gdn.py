import torch
import torch.nn as nn
import torch.nn.functional as F

from .math_ops import lower_bound


class NonnegativeParameterizer(nn.Module):
    def __init__(self, shape, initializer, minimum=0, reparam_offset=2**-18):
        super(NonnegativeParameterizer, self).__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)


        self._pedestal = self.reparam_offset ** 2
        self._bound = (self.minimum + self.reparam_offset ** 2) ** 0.5


        self._var = nn.Parameter(torch.sqrt(initializer(shape) + self._pedestal))

    def forward(self):
        var = lower_bound(self._var, self._bound)
        var = torch.pow(var, 2) - self._pedestal

        return var


class GDN2d(nn.Module):
    def __init__(self, channels, inverse):
        super(GDN2d, self).__init__()
        self.channels = int(channels)
        self.inverse = bool(inverse)

        self._beta = NonnegativeParameterizer(
            shape=(self.channels,),
            initializer=lambda shape: torch.ones(*shape),
            minimum=1e-6
        )
        self._gamma = NonnegativeParameterizer(
            shape=(self.channels, self.channels),
            initializer=lambda shape: torch.eye(*shape)*0.1
        )

    @property
    def beta(self):
        return self._beta()

    @property
    def gamma(self):
        return self._gamma().view(self.channels, self.channels, 1, 1)

    def forward(self, inputs):
        norm_pool = F.conv2d(inputs ** 2, self.gamma, self.beta)

        if self.inverse:
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = torch.rsqrt(norm_pool)

        outputs = inputs * norm_pool

        return outputs
