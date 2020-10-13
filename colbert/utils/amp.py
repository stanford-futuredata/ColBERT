import torch
from contextlib import contextmanager

PyTorch_over_1_6 = float('.'.join(torch.__version__.split('.')[0:2])) >= 1.6


class MixedPrecisionManager():
    def __init__(self, activated):
        assert (not activated) or PyTorch_over_1_6, "Cannot use AMP for PyTorch version < 1.6"

        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        # see https://stackoverflow.com/a/45187287
        import sys
        PY_36 = sys.version_info < (3, 7)
        if PY_36:
            class NullContextManager(object):
                def __init__(self, dummy_resource=None):
                    self.dummy_resource = dummy_resource
                def __enter__(self):
                    return self.dummy_resource
                def __exit__(self, *args):
                    pass
            nullcontext = NullContextManager
        else:
            from contextlib import nullcontext

        return torch.cuda.amp.autocast() if self.activated else nullcontext()

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
