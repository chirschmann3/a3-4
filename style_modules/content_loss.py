import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
            Compute the content loss for style transfer.

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        ##############################################################################
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        img, num_filts, h, w = content_current.size()
        current_reshaped = content_current.view(img, num_filts, -1)
        original_reshaped = content_original.view(img, num_filts, -1)
        vect_diff = (current_reshaped - original_reshaped)**2
        loss = content_weight * torch.sum(vect_diff)
        return loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

