"""
Activation functions for complex-valued networks (e.g., zReLU, modReLU).

- Implementations are numerically stable and PyTorch-friendly.
"""
import torch
from torch import nn
import torch.nn.functional as F

############### Complex Activation Functions ###############
# Complex ReLU
class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x.real).type(torch.complex64) + 1j * F.relu(x.imag).type(torch.complex64)

# Complex PReLU
class CPReLU(nn.Module):
    def __init__(self, num_channels= 1):
        super().__init__()
        self.num_channels= num_channels
        self.real_prelu = nn.PReLU(num_parameters= self.num_channels)
        self.imag_prelu = nn.PReLU(num_parameters= self.num_channels)

    def forward(self, x):
        return self.real_prelu(x.real).type(torch.complex64) + 1j * self.imag_prelu(x.imag).type(torch.complex64)

# zReLU Activation Function
class zReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        real = x.real
        imag = x.imag
        mask = (real > 0) & (imag > 0)
        return (real * mask).type(torch.complex64) + 1j * (imag * mask).type(torch.complex64)   

# Naive Complex Sigmoid and Tanh
class Naive_ComplexSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.sigmoid(x.real).type(torch.complex64) + 1j * F.sigmoid(x.imag).type(torch.complex64)

# Naive Complex Tanh
class Naive_ComplexTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.tanh(x.real).type(torch.complex64) + 1j * F.tanh(x.imag).type(torch.complex64)

