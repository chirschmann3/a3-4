import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        # based off implementation from https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574
        h_var = torch.sum((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2))
        w_var = torch.sum((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2))

        return torch.multiply((h_var + w_var), tv_weight)
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################