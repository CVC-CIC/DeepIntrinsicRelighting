import torch
import torch.nn as nn


class SigLinear(nn.Module):
    """
    SigLinear is a custom PyTorch activation module that combines the Sigmoid function for the
    negative half-axis and a linear function for the positive half-axis.

    Specifically:
    - For input values x < 0, the output is given by the Sigmoid function.
    - For input values x >= 0, the output is given by the linear function y = 0.5 + 0.25 * x.

    The activation function is continuous and differentiable at x = 0.
    """
    def __init__(self):
        super(SigLinear, self).__init__()

    def forward(self, x):
        sigmoid_mask = x < 0
        linear_mask = ~sigmoid_mask

        sigmoid_part = torch.sigmoid(x[sigmoid_mask])
        linear_part = 0.5 + 0.25 * x[linear_mask]

        output = torch.empty_like(x)
        output[sigmoid_mask] = sigmoid_part
        output[linear_mask] = linear_part

        return output


"""

"""
class ClampWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_value, max_value):
        ctx.save_for_backward(x)
        ctx.min_value = min_value
        ctx.max_value = max_value
        return x.clamp(min_value, max_value)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < ctx.min_value) | (x > ctx.max_value)] = 1
        return grad_input, None, None

class CustomClamp(nn.Module):
    """
    CustomClamp is a custom module that applies element-wise clamping to the input tensor.
    It ensures that the output values are within a specified range [min_value, max_value],
    while maintaining gradients outside the clamping range for learning purposes.
    Attributes:
        min_value (float): The lower bound of the clamping range.
        max_value (float): The upper bound of the clamping range.
    """
    def __init__(self, min_value, max_value):
        super(CustomClamp, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return ClampWithGradient.apply(x, self.min_value, self.max_value)

