import torch
from torch import nn as nn


class NanBatchNorm1dNaive(nn.Module):
    def __init__(self,
                 n_feature,
                 eps=1e-5,
                 momentum=0.1,
                 affine=False,
                 track_running_stats=True):
        super(NanBatchNorm1dNaive, self).__init__()
        self.num_features = n_feature
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, 1, self.num_features))
            self.register_buffer('running_std', torch.ones(1, 1, self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0.))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_std', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, input):
        """
        :param input: [bs, sl, fd]
        """
        if self.training:
            mask = torch.isnan(input)
            nan_mean_per_feat = torch.nanmean(input, dim=(0, 1), keepdim=True)
            count_no_nan_per_feat = (~mask).float().sum(dim=(0, 1), keepdim=True)
            sum_no_mean_per_feat = torch.nansum((input - nan_mean_per_feat) ** 2, dim=(0, 1), keepdim=True)
            nan_std_per_feat = torch.sqrt(sum_no_mean_per_feat / count_no_nan_per_feat)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = self.momentum * nan_mean_per_feat + \
                                        (1 - self.momentum) * self.running_mean
                    self.running_std = self.momentum * nan_std_per_feat + \
                                       (1 - self.momentum) * self.running_std

        input_norm = (input - self.running_mean) / (self.running_std + self.eps)

        return input_norm
