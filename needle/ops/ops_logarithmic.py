from typing import Optional
from ..autograd import NDArray
from ..autograd import Tensor, TensorOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        return Z - array_api.log(array_api.sum(array_api.exp(Z), axis=-1, keepdims=True))

    def gradient(self, out_grad, node):
        # how much this is changing with respect to the input
        ...
        


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = None if axes is None else (axes,) if isinstance(axes, int) else axes


    def compute(self, Z):
        max_z = array_api.max(Z, axis=self.axes, keepdims=True) # (M,) where M is the number of samples
        return array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes)) + max_z.squeeze() # (M,)

    def gradient(self, out_grad, node):
        # The following lines are to get it back to the actual input Z_ in compute()
        inp = node.inputs[0].numpy()
        
        max_z = array_api.max(inp, axis=self.axes, keepdims=True)
        Z_ = Tensor(inp - max_z)
        # gradient with respect to log
        log_grad = out_grad / summation(exp(Z_), axes=self.axes) # (M,) where M is the number of samples
        # I have to get this to the right shape so I can use it on Z_
        base_shape = list(Z_.shape) # (M, N) where M is the number of samples and N is the number of classes
        for ax in self.axes or range(len(base_shape)):
            base_shape[ax] = 1 # (M, 1) 
        
        log_grad = reshape(log_grad, base_shape) # (M, 1)
        log_grad = broadcast_to(log_grad, Z_.shape)
        sum_exp_grad = exp(Z_) 
        out_grad = log_grad * sum_exp_grad
        return out_grad
        

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

