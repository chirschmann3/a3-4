import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np
import torch
import torch
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from helpers.data_utils import *
from helpers.image_utils import *


class SaliencyMap:
    def compute_saliency_maps(self, X, y, model):
        """
        Compute a class saliency map using the model for images X and labels y.

        Input:
        - X: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the saliency map.

        Returns:
        - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
        images.
        """
        # Make sure the model is in "test" mode
        model.eval()

        # Wrap the input tensors in Variables
        X_var = Variable(X, requires_grad=True)
        y_var = Variable(y, requires_grad=False)
        saliency = None

        ##############################################################################
        # TODO: Implement this function. Perform a forward and backward pass through #
        # the model to compute the gradient of the correct class score with respect  #
        # to each input image.                                                       #    
        #                                                                            #
        # There are two approaches to performing backprop using the PyTorch command  #
        # tensor.backward() when computing the gradient of a non-scalar  tensor. One #
        # approach is listed in PyTorch docs. Alternatively, one can take the sum of #
        # all the elements of the tensor and do a single backprop with the resulting #
        # scalar. This second approach is simpler and preferable as it lends itself  #  
        # vectorization, and this is the one you should implement.                   #
        #                                                                            #
        # Note: Only a single back-propagation pass is required                      #
        ##############################################################################
        score = model(X_var)[range(len(y)), y]
        score_sums = score.sum()
        score_sums.backward()

        grad_abs = X_var.grad.abs()
        saliency = grad_abs.max(axis=1)[0]
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return saliency

    def show_saliency_maps(self, X, y, labels, model, folder="visualization"):
        # Convert X and y from numpy arrays to Torch Tensors
        X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
        y_tensor = torch.LongTensor(y)

        # Compute saliency maps for images in X
        saliency = self.compute_saliency_maps(X_tensor, y_tensor, model)
        # Convert the saliency map from Torch Tensor to numpy array and show images
        # and saliency maps together.
        saliency = saliency.numpy()

        # Create a figure and a subplot with 2 rows and 4 columns
        fig, ax = plt.subplots(2, 5, figsize=(12, 6))
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)

        # Loop over the subplots and plot an image in each one
        for i in tqdm(range(2), desc="Creating plots", leave=True):
            for j in tqdm(range(5), desc="Processing image", leave=True):
                # Load image
                if i == 0:
                    image = X[j]
                elif i == 1:
                    image = saliency[j]

                # Plot the image in the current subplot
                ax[i, j].imshow(image)
                ax[i, j].axis('off')

                # Add a label above each image in the bottom row
                if i == 1:
                    ax[i, j].set_title(labels[j].title(), fontsize=12, y=1.2)

        # Save and display the subplots
        plt.savefig(f"./{folder}/saliency_visualization.png")
        plt.show()


if __name__ == '__main__':
    # Check for GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.DEFAULT').to(device)
    for param in model.parameters():
        param.requires_grad = False

    # Get data and instantiate SaliencyMap
    X, y, labels, class_names = load_images(num=5)
    sm = SaliencyMap()
    sm.show_saliency_maps(X, y, labels, model)

