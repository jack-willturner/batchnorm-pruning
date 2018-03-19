import torch
from torch.optim import SGD


class BatchNormSGD(SGD):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        ista ([float], required) : list of ista penalties for each layer
    """

    def __init__(self, params, ista, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.ista = ista
        super(BatchNormSGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # not allowed to use weight_decay, should add a check

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum     = group['momentum']
            dampening    = group['dampening']
            nesterov     = group['nesterov']


            for p, ista in zip(group['params'], self.ista):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.new().resize_as_(p.data).zero_()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply group lasso
                x      = p.data.add(-group['lr'],d_p)
                x      = torch.clamp((torch.abs(x) - ista), min=0.) # second elem is index location
                p.data = x * torch.sign(x)
                
        return loss
