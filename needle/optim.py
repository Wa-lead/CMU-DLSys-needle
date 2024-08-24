from abc import abstractmethod
import numpy as array_api


class Optimizer:
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def step(self): ...

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}  # velocity
        self.weight_decay = weight_decay  # regularzation

    def step(self):
        for idx, p in enumerate(self.params):
            grad = p.grad.numpy() + self.weight_decay * p.numpy()
            if idx not in self.u.keys():
                self.u[idx] = 0
            self.u[idx] = self.momentum * self.u[idx] + (1 - self.momentum) * grad
            self.params[idx].data = p.data - self.lr * self.u[idx]

    def clip_grad_norm(self, max_norm=0.25): ...


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for idx, p in enumerate(self.params):
            grad = p.grad.numpy() + self.weight_decay * p.numpy()
            self.m[idx] = self.beta1 * self.m.get(idx, 0.0) + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v.get(idx, 0.0) + (1 - self.beta2) * grad**2

            # correct bias
            m_hat = self.m[idx] / (1 - self.beta1**self.t)
            v_hat = self.v[idx] / (1 - self.beta2**self.t)
            self.params[idx].data = p.data - self.lr * m_hat / (v_hat**0.5 + self.eps)
