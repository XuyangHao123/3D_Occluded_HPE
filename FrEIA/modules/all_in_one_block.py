from . import InvertibleModule

import warnings
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group


class AllInOneBlock(InvertibleModule):
    '''Module combining the most common operations in a normalizing flow or similar model.

    It combines affine coupling, permutation, and global affine transformation
    ('ActNorm'). It can also be used as GIN coupling block, perform learned
    householder permutations, and use an inverted pre-permutation (see
    constructor docstring for details).'''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 affine_clamping: float = 2.,
                 gin_block: bool = False,
                 global_affine_init: float = 1.,
                 global_affine_type: str = 'SOFTPLUS',
                 permute_soft: bool = False,
                 learned_householder_permutation: int = 0,
                 reverse_permutation: bool = False):
        '''
        Args:
          subnet_constructor:
            class or callable f, called as f(channels_in, channels_out) and
            should return a torch.nn.Module
          affine_clamping:
            clamp the output of the multiplicative coefficients (before
            exponentiation) to +/- affine_clamping.
          gin_block:
            Turn the block into a GIN block from Sorrenson et al, 2019
          global_affine_init:
            Initial value for the global affine scaling beta
          global_affine_init:
            'SIGMOID', 'SOFTPLUS', or 'EXP'. Defines the activation to be used
            on the beta for the global affine scaling.
          permute_soft:
            bool, whether to sample the permutation matrices from SO(N), or to
            use hard permutations in stead. Note, permute_soft=True is very slow
            when working with >512 dimensions.
          learned_householder_permutation:
            Int, if >0,  use that many learned householder reflections. Slow if
            large number. Dubious whether it actually helps.
          reverse_permutation:
            Reverse the permutation before the block, as introduced by Putzky
            et al, 2019.
        '''

        super().__init__(dims_in, dims_c)

        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        try:
            self.permute_function = {0: F.linear,
                                     1: F.conv1d,
                                     2: F.conv2d,
                                     3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        self.in_channels         = channels
        self.clamp               = affine_clamping
        self.GIN                 = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder         = learned_householder_permutation

        if permute_soft and channels > 512:
            warnings.warn(("Soft permutation will take a very long time to initialize "
                           f"with {channels} feature channels. Consider using hard permutation instead."))

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        if global_affine_type == 'SIGMOID':
            global_scale = 2. - np.log(10. / global_affine_init - 1.)
            self.global_scale_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif global_affine_type == 'SOFTPLUS':
            global_scale = 2. * np.log(np.exp(0.5 * 10. * global_affine_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.global_scale_activation = (lambda a: 0.1 * self.softplus(a))
        elif global_affine_type == 'EXP':
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = (lambda a: torch.exp(a))
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.global_scale = nn.Parameter(torch.ones(1, self.in_channels, *([1] * self.input_rank)) * float(global_scale))
        self.global_offset = nn.Parameter(torch.zeros(1, self.in_channels, *([1] * self.input_rank)))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, j] = 1.

        if self.householder:
            # instead of just the permutation matrix w, the learned housholder
            # permutation keeps track of reflection vectors vk, in addition to a
            # random initial permutation w_0.
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w_perm = nn.Parameter(torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
                                       requires_grad=False)
            self.w_perm_inv = nn.Parameter(torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
                                           requires_grad=False)

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor"
                             "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None

    def _construct_householder_permutation(self):
        '''Computes a permutation matrix from the reflection vectors that are
        learned internally as nn.Parameters.'''
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        for i in range(self.input_rank):
            w = w.unsqueeze(-1)
        return w

    def _permute(self, x, rev=False):
        '''Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.'''
        if self.GIN:
            scale = 1.
            perm_log_jac = 0.
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale,
                    perm_log_jac)
        else:
            return (self.permute_function(x * scale + self.global_offset, self.w_perm),
                    perm_log_jac)

    def _pre_permute(self, x, rev=False):
        '''Permutes before the coupling block, only used if
        reverse_permutation is set'''
        if rev:
            return self.permute_function(x, self.w_perm)
        else:
            return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x, a, rev=False):
        '''Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet.'''

        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        # b, 26
        a *= 0.1
        # 13
        ch = x.shape[1]
        # b, 13
        sub_jac = self.clamp * torch.tanh(a[:, :ch])
        if self.GIN:
            sub_jac -= torch.mean(sub_jac, dim=self.sum_dims, keepdim=True)

        if not rev:
            return (x * torch.exp(sub_jac) + a[:, ch:],
                    torch.sum(sub_jac, dim=self.sum_dims))
        # b, 13, b
        else:
            return ((x - a[:, ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=self.sum_dims))
        # b, 13

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            # tuple(b, 26)
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)
        # b, 13, b, 13
        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            # b, 26
            a1 = self.subnet(x1c)
            # b, 13, b
            x2, j2 = self._affine(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)
        # b
        log_jac_det = j2
        # b, 26
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        # 1
        n_pixels = x_out[0, :1].numel()
        # b,
        log_jac_det += (-1)**rev * n_pixels * global_scaling_jac
        # tuple(b, 26), b
        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims
