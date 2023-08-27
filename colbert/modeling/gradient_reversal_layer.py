import torch
from torch.autograd import Function

# Mostly sourced from https://github.com/uds-lsv/da-lang-id/blob/master/nn_speech_models.py
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -_lambda the gradient
        output = grad_output.neg() * ctx._lambda

        # Must return same number as inputs to forward()
        return output, None