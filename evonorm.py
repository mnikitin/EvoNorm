import mxnet as mx
from mxnet import nd

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class EvoNormB0(nn.HybridBlock):
    def __init__(self, in_channels, momentum=0.9, eps=1e-5, training=True):
        super(EvoNormB0, self).__init__()
        self.training = training
        self.momentum = momentum
        self.eps = eps
        with self.name_scope():
            shape = (1, in_channels, 1, 1)
            self.gamma = self.params.get('gamma', grad_req='write',
                                         shape=shape, init=mx.init.Constant(1),
                                         differentiable=True)
            self.beta = self.params.get('beta', grad_req='write',
                                        shape=shape, init=mx.init.Constant(0),
                                        differentiable=True)
            self.v = self.params.get('v', grad_req='write',
                                     shape=shape, init=mx.init.Constant(1),
                                     differentiable=True)
            self.running_var = self.params.get('running_var', grad_req='null',
                                               shape=shape, init=mx.init.Constant(1),
                                               differentiable=False)

    def instance_std(self, F, x):
        _, var = F.moments(x, axes=(2, 3), keepdims=True)
        std = F.sqrt(var + self.eps)
        return std

    def hybrid_forward(self, F, x, gamma, beta, running_var, v):
        if self.training:
            _, var = F.moments(x, axes=(0, 2, 3), keepdims=True)
            running_var = running_var * self.momentum + var * (1.0 - self.momentum)
        else:
            var = running_var
        batch_std = F.sqrt(var + self.eps)
        instance_std = self.instance_std(F, x)
        den = F.broadcast_maximum(F.broadcast_add(F.broadcast_mul(v, x), instance_std), batch_std)
        x = x / den
        # affine transformation
        x = F.broadcast_add(F.broadcast_mul(x, gamma), beta)
        return F.broadcast_add(x, 0 * running_var) # nasty workaround for "unused parameter" crash


class EvoNormS0(nn.HybridBlock):
    def __init__(self, in_channels, groups=32, eps=1e-5):
        super(EvoNormS0, self).__init__()
        self.insize = in_channels
        self.groups = groups
        self.eps = eps
        with self.name_scope():
            shape = (1, self.insize, 1, 1)
            self.gamma = self.params.get('gamma', grad_req='write',
                                         shape=shape, init=mx.init.Constant(1),
                                         differentiable=True)
            self.beta = self.params.get('beta', grad_req='write',
                                        shape=shape, init=mx.init.Constant(0),
                                        differentiable=True)
            self.v = self.params.get('v', grad_req='write',
                                     shape=shape, init=mx.init.Constant(1),
                                     differentiable=True)

    def group_std(self, F, x):
        x = F.reshape(x, shape=(0, -4, self.groups, self.insize // self.groups, -2))
        _, var = F.moments(x, axes=(2, 3, 4), keepdims=True)
        std = F.sqrt(var + self.eps)
        std = F.broadcast_like(std, x)
        std = F.reshape(std, shape=(0, -3, -2))
        return std

    def hybrid_forward(self, F, x, gamma, beta, v):
        swish = x * F.Activation(F.broadcast_mul(v, x), act_type='sigmoid')
        x = swish / self.group_std(F, x)
        # affine transformation
        x = F.broadcast_add(F.broadcast_mul(x, gamma), beta)
        return x
