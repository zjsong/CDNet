"""
Define the cross covariance (XCov) loss/function between two groups of vectors.
"""


import torch
import torch.nn as nn


class XCov(nn.Module):
    def __init__(self):
        super(XCov, self).__init__()

    def forward(self, x, y):
        batch_size = x.size()[0]
        if len(x.size()) == 3:
            x = x.view(batch_size, -1)
        else:
            pass

        diff_x = x - x.mean(0)
        diff_y = y - y.mean(0)

        XCov_value = torch.sum(torch.mm(torch.t(diff_x), diff_y)**2) / (2 * batch_size**2)

        return XCov_value
