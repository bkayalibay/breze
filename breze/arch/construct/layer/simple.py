# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T

from breze.arch.component import transfer as _transfer, loss as _loss
from breze.arch.construct.base import Layer
from breze.arch.util import lookup, get_named_variables


class AffineNonlinear(Layer):

    @property
    def n_inpt(self):
        return self._n_inpt

    @property
    def n_output(self):
        return self._n_output

    def __init__(self, n_inpt, n_output, transfer='identity', bias=True,
                 name=None):
        self._n_inpt = n_inpt
        self._n_output = n_output
        self.transfer = transfer
        self.bias = True
        super(AffineNonlinear, self).__init__(name=name)

    def forward(self, inpt):
        super(AffineNonlinear, self).forward(inpt)

        weights = self.parameterized('weights', (self.n_inpt, self.n_output))

        output_pre_transfer = T.dot(inpt, weights)

        if self.bias:
            bias = self.parameterized('bias', (self.n_output,))
            output_pre_transfer += bias

        f_transfer = lookup(self.transfer, _transfer)
        output = f_transfer(output_pre_transfer)

        E = self.exprs = get_named_variables(locals())
        self.output = [output]


class Split(Layer):

    def __init__(self, lengths, axis=1, name=None):
        self.lengths = lengths
        self.axis = axis
        super(Split, self).__init__(name)

    def forward(self, inpt):
        starts = [0] + np.add.accumulate(self.lengths).tolist()
        stops = starts[1:]
        starts = starts[:-1]

        E = self.exprs = get_named_variables(locals())
        self.output = [inpt[:, start:stop]
                       for start, stop in zip(starts, stops)]


class Concatenate(Layer):

    def __init__(self, axis=1, name=None):
        self.axis = axis
        super(Concatenate, self).__init__(name)

    def forward(self, *inpts):
        concatenated = T.concatenate(inpts, self.axis)
        E = self.exprs = get_named_variables(locals())
        self.output = [concatenated]


class SupervisedLoss(Layer):

    def __init__(self, loss, target, comp_dim=1, imp_weight=None,
                 name=None):
        self.loss = loss
        self.target = target
        self.imp_weight = imp_weight
        self.comp_dim = comp_dim

        super(SupervisedLoss, self).__init__(name)

    def forward(self, inpt):
        super(SupervisedLoss, self).forward(inpt)
        f_loss = lookup(self.loss, _loss)

        coord_wise = f_loss(self.target, inpt)
        if self.imp_weight is not None:
            coord_wise *= self.imp_weight
        sample_wise = coord_wise.sum(self.comp_dim)
        total = sample_wise.mean()

        E = self.exprs = get_named_variables(locals())
        E['target'] = self.target
        if self.imp_weight is not None:
            E['imp_weight'] = self.imp_weight
        self.output = total,


class UnsupervisedLoss(Layer):

    def __init__(self, loss, comp_dim=1, imp_weight=None,
                 name=None):
        self.loss = loss
        self.imp_weight = imp_weight
        self.comp_dim = comp_dim

        super(UnsupervisedLoss, self).__init__(name)

    def forward(self, inpt):
        super(UnsupervisedLoss, self).forward(inpt)
        f_loss = lookup(self.loss, _loss)

        coord_wise = f_loss(inpt)
        if self.imp_weight is not None:
            coord_wise *= self.imp_weight
        sample_wise = coord_wise.sum(self.comp_dim)
        total = sample_wise.mean()

        E = self.exprs = get_named_variables(locals())
        if self.imp_weight is not None:
            E['imp_weight'] = self.imp_weight
        self.output = total,