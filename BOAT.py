import math
import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor, threshold
from typing import List, Optional


class BOAT(Optimizer):

    def __init__(self, params, eta= 0.1, weight_decay=1e-3):
        defaults = dict( eta=eta, weight_decay=weight_decay)
        super(BOAT, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:  
            for p in group['params']:
                if hasattr(p,'org'):
                    if p.grad is not None:
                        grad = p.grad.data
                        
                        eta=group['eta']
                        weight_decay=group['weight_decay']

                        if not hasattr(p,'m'):
                            p.m = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if not hasattr(p,'v'):
                            p.v = torch.zeros_like(p, memory_format=torch.preserve_format)
   
                        v_t = p.v.addcmul_(grad, grad, value=eta)

                        temp = v_t.sqrt()

                        p.m.mul_(1-weight_decay).add_(grad,alpha=-1)

                        p.data = torch.sign(torch.sign(torch.where(((p.m.abs())>temp),p.m, p.pre_binary_data)).add(0.1)) 
                        
        return loss
