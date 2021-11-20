import torch

from contextlib import contextmanager
from colbert.utils.utils import NullContextManager
from packaging import version

v = version.parse
PyTorch_over_1_6  = v(torch.__version__) >= v('1.6')

class MixedPrecisionManager():
    def __init__(self, activated):
        assert (not activated) or PyTorch_over_1_6, "Cannot use AMP for PyTorch version < 1.6"

        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)

            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
