# -*- coding: utf-8 -*-

import numpy as np

import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv3d2d
from theano.tensor.nnet.Conv3D import conv3D
from theano.tensor.signal import downsample

from breze.arch.component import transfer as _transfer, loss as _loss
from breze.arch.construct.base import Layer
from breze.arch.util import lookup


class AffineNonlinear(Layer):

    @property
    def n_inpt(self):
        return self._n_inpt

    @property
    def n_output(self):
        return self._n_output

    def __init__(self, inpt, n_inpt, n_output, transfer='identity',
                 use_bias=True, declare=None, name=None):
        self.inpt = inpt
        self._n_inpt = n_inpt
        self._n_output = n_output
        self.transfer = transfer
        self.use_bias = use_bias
        super(AffineNonlinear, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.weights = self.declare((self.n_inpt, self.n_output))

        self.output_in = T.dot(self.inpt, self.weights)

        if self.use_bias:
            self.bias = self.declare(self.n_output)
            self.output_in += self.bias

        f = lookup(self.transfer, _transfer)

        self.output = f(self.output_in)


class Split(Layer):

    def __init__(self, inpt, lengths, axis=1, declare=None, name=None):
        self.inpt = inpt
        self.lengths = lengths
        self.axis = axis
        super(Split, self).__init__(declare, name)

    def _forward(self):
        starts = [0] + np.add.accumulate(self.lengths).tolist()
        stops = starts[1:]
        starts = starts[:-1]

        self.outputs = [self.inpt[:, start:stop] for start, stop
                        in zip(starts, stops)]


class Concatenate(Layer):

    def __init__(self, inpts, axis=1, declare=None, name=None):
        self.inpts = inpts
        self.axis = axis
        super(Concatenate, self).__init__(declare, name)

    def _forward(self):
        concatenated = T.concatenate(self.inpts, self.axis)
        self.output = concatenated


class SupervisedLoss(Layer):

    def __init__(self, target, prediction, loss, comp_dim=1, imp_weight=None,
                 declare=None, name=None):
        self.target = target
        self.prediction = prediction
        self.loss_ident = loss

        self.imp_weight = imp_weight
        self.comp_dim = comp_dim

        super(SupervisedLoss, self).__init__(declare, name)

    def _forward(self):
        f_loss = lookup(self.loss_ident, _loss)

        self.coord_wise = f_loss(self.target, self.prediction)

        if self.imp_weight is not None:
            self.coord_wise *= self.imp_weight

        self.sample_wise = self.coord_wise.sum(self.comp_dim)

        self.total = self.sample_wise.mean()


class Conv2d(Layer):

    def __init__(self, inpt, inpt_height, inpt_width, n_inpt,
                 filter_height, filter_width,
                 n_output, transfer='identity',
                 n_samples=None,
                 subsample=(1, 1),
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.n_inpt = n_inpt

        self.filter_height = filter_height
        self.filter_width = filter_width

        self.n_output = n_output
        self.transfer = transfer
        self.n_samples = n_samples
        self.subsample = subsample

        # self.output_height, _ = divmod(inpt_height, filter_height)
        # self.output_width, _ = divmod(inpt_width, filter_width)
        self.output_height = (inpt_height - filter_height) / subsample[0] + 1
        self.output_width = (inpt_width - filter_width) / subsample[1] + 1

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than filter height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than filter width')

        super(Conv2d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.weights = self.declare((
            self.n_output, self.n_inpt,
            self.filter_height, self.filter_width))
        self.bias = self.declare((self.n_output,))

        self.output_in = conv.conv2d(
            self.inpt, self.weights,
            image_shape=(
                self.n_samples, self.n_inpt, self.inpt_height, self.inpt_width),
            subsample=self.subsample,
            border_mode='valid',
            )

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)


class MaxPool2d(Layer):

    def __init__(self, inpt, inpt_height, inpt_width, pool_height, pool_width,
                 n_output,
                 transfer='identity',
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.transfer = transfer

        self.output_height, _ = divmod(inpt_height, pool_height)
        self.output_width, _ = divmod(inpt_width, pool_width)

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than pool height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than pool width')

        self.n_output = n_output

        super(MaxPool2d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.output_in = downsample.max_pool_2d(
            input=self.inpt, ds=(self.pool_height, self.pool_width),
            ignore_border=True)

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)

class Conv3d(Layer):
    def __init__(self, inpt, inpt_height, inpt_width,
                 inpt_depth, n_inpt, filter_height,
                 filter_width, filter_depth, n_output,
                 transfer='identity', n_samples=None,
                 declare=None, name=None,
                 implementation='conv3D'):
        """
        Create one layer of 3d convolution.
        """
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.inpt_depth = inpt_depth
        self.n_inpt = n_inpt

        self.filter_height = filter_height
        self.filter_width = filter_width
        self.filter_depth = filter_depth 

        self.n_output = n_output
        self.transfer = transfer
        self.n_samples = n_samples
     
        self.output_height = inpt_height - filter_height + 1
        self.output_width = inpt_width - filter_height + 1
        self.output_depth = inpt_depth - filter_depth + 1

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than filter height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than filter width')
        if not self.output_depth > 0:
            raise ValueError('inpt depth smaller than filter depth')

        self.implementation = implementation

        super(Conv3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.weights = self.declare(
            (self.n_output, self.filter_depth, self.n_inpt,
             self.filter_height, self.filter_width)
        )
        self.bias = self.declare((self.n_output,))

        inpt_shape = (self.n_samples, self.inpt_depth, self.n_inpt,
                      self.inpt_height, self.inpt_width)
        filter_shape= (self.n_output, self.filter_depth, self.n_inpt,
                       self.filter_height, self.filter_width)        
        
        if self.implementation == 'conv3d2d':
            self.output_in = conv3d2d.conv3d(
                signals=self.inpt, 
                filters=self.weights, 
                filters_shape=filter_shape    
            )
        elif self.implementation == 'conv3D':
            filters_flip = self.weights[:,::-1,:,::-1,::-1]
            self.output_in = conv3D(
                V=self.inpt.dimshuffle(0,3,4,1,2),
                W=filters_flip.dimshuffle(0,3,4,1,2),
                b=self.bias,
                d=(1,1,1)
            )
            self.output_in = self.output_in.dimshuffle(0,3,4,1,2)
        else:
            msg = 'This class only supports conv3d2d and conv3D'
            raise NotImplementedError(msg)

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)

def max_pool_3d(inpt, inpt_shape, ds, ignore_border=True):
    # Downsize 'into the depth' by downsizing twice.
    inpt_shape_4d = (
        inpt_shape[0]*inpt_shape[1],
        inpt_shape[2],
        inpt_shape[3],
        inpt_shape[4]
    )

    inpt_as_tensor4 = T.reshape(inpt, inpt_shape_4d, ndim=4)
    
    # The first pooling only downsizes the height and the width.
    pool_out1 = downsample.max_pool_2d(inpt_as_tensor4, (ds[1], ds[2]), 
                                       ignore_border=True) 
    out_shape1 = T.join(0, inpt_shape[:-2], pool_out1.shape[-2:])

    inpt_pooled_once = T.reshape(pool_out1, out_shape1, ndim=5)

    # Shuffle dimensions so the depth is the last dimension.
    inpt_shuffled = inpt_pooled_once.dimshuffle(0, 4, 2, 3, 1)
    
    shuffled_shape = inpt_shuffled.shape
    # Reshape input to be 4 dimensional.
    shuffle_shape_4d = (
        shuffled_shape[0] * shuffled_shape[1],
        shuffled_shape[2],
        shuffled_shape[3],
        shuffled_shape[4]
    )

    inpt_shuffled_4d = T.reshape(inpt_shuffled, shuffle_shape_4d, ndim=4)

    pool_out2 = downsample.max_pool_2d(inpt_shuffled_4d, (1,ds[0]), 
                            ignore_border=True)
    out_shape2 = T.join(0, shuffled_shape[:-2], pool_out2.shape[-2:]) 

    inpt_pooled_twice = T.reshape(pool_out2, out_shape2, ndim=5)
    pool_output_fin = inpt_pooled_twice.dimshuffle(0, 4, 2, 3, 1)

    return pool_output_fin

class MaxPool3d(Layer):

    def __init__(self, inpt, inpt_height, inpt_width, inpt_depth,
                 pool_height, pool_width, pool_depth, n_output,
                 transfer='identity', declare=None, name=None):
        """
        One layer of 3D max pooling.
        """

        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.inpt_depth = inpt_depth

        self.pool_height = pool_height
        self.pool_width = pool_width
        self.pool_depth = pool_depth

        self.transfer = transfer
        self.output_height, _ = divmod(inpt_height, pool_height)
        self.output_width, _ = divmod(inpt_width, pool_width)
        self.output_depth, _ = divmod(inpt_depth, pool_depth)

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than pool height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than pool width')
        if not self.output_depth > 0:
            raise ValueError('inpt depth smaller than pool depth')

        self.n_output = n_output

        super(MaxPool3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        poolsize = (self.pool_depth, self.pool_height, self.pool_width)
        
        self.output_in = max_pool_3d(
            inpt=self.inpt, 
            inpt_shape=self.inpt.shape, 
            ds=poolsize, 
            ignore_border=True
        )

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)
