import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from helpers.data_utils import *
from helpers.image_utils import *
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.use('TkAgg')

class GradCAM():
    
    def __init__(self, X):
        self.X = X

    def gradcam(self):

        # Define a preprocessing function to resize the image and normalize its pixels
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Check for GPU support
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.DEFAULT').to(device)

        ##############################################################################
        # TODO: Define a hook function to get the feature maps from the last         #
        # convolutional layer. Then register the hook to get the feature maps        #
        ##############################################################################
        gradcams = []
        grads = {}
        acts = {}

        # Stores the gradient
        def back_hook(mod, grad_in, grad_out):
            grads['val'] = grad_out[0]
            return None

        # Stores output of ReLU
        def fwrd_hook(mod, input, output):
            acts['val'] = output
            return None

        model.features[-1].expand3x3_activation.register_forward_hook(fwrd_hook)
        model.features[-1].expand3x3_activation.register_backward_hook(back_hook)

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        for x in self.X:
            # Load an input image and perform pre-processing
            image = Image.fromarray(x)
            x = preprocess(image).unsqueeze(0)

            # Make a forward pass through the model
            logits = model(x)

            # Get the class with the highest probability
            class_idx = torch.argmax(logits)

            ###############################################################################
            # TODO: To generate a Grad-CAM heatmap, first compute the gradients of the    #
            # output class with respect to the feature maps. Then, calculate the weights  #
            # of the feature maps. Using these weights, compute the Grad-CAM heatmap.     #
            # Use the cv2 for resizing the heatmap if necessary.                          #
            ###############################################################################
            # below based on ideas from https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/gradcam.py

            # get score and send backwards to get hooked activations and gradients
            score = logits[0, class_idx]
            model.zero_grad()
            score.backward()
            gradnts = grads['val']
            activtns = acts['val']

            # get avg gradient per "pixel" per filter at last layer and reshape it
            weights = gradnts.view(gradnts.size()[0], gradnts.size()[1], -1)
            weights = weights.mean(2)
            weights = weights.view(gradnts.size()[0], gradnts.size()[1], 1, 1)

            # multiply activation of each "pixel" in each kernel by avg weight for that kernel
            # then sum across kernels to reduce to single depth dimension
            weighted_act = (weights*activtns)
            weighted_act = weighted_act.sum(1, keepdim=True)

            # remove negative values with ReLU & upsample to match size of original image
            map = F.relu(weighted_act)
            map = F.interpolate(map, size=(x.size()[2], x.size()[3]), mode='bicubic')

            # normalize result
            with torch.no_grad():
                map = (map - map.min()).div(map.max() - map.min())

            # clip to match img dimensions
            heatmap = torch.squeeze(map)

            ##############################################################################
            #                             END OF YOUR CODE                               #
            ##############################################################################

            # store gradcams
            gradcams.append([x.squeeze(0).permute(1,2,0).numpy(), heatmap])

        return gradcams


if __name__ == '__main__':

    # Retrieve images
    X, y, labels, class_names = load_images(num=5)
    gc = GradCAM(X)
    gradcams = gc.gradcam()
    # Create a figure and a subplot with 2 rows and 4 columns
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)

    # Loop over the subplots and plot an image in each one
    for i in tqdm(range(2), desc="Creating plots", leave=True):
        for j in tqdm(range(5), desc="Processing image", leave=True):
            # Load image
            if i == 0:
                item = gradcams[j]
                image = item[0].clip(0,1)
                ax[i, j].imshow(image, alpha=.87, vmin=.5)
                ax[i, j].axis('off')
            elif i == 1:
                item = gradcams[j]
                image = item[0].clip(0,1)
                overlay = item[1]

                # Plot the image in the current subplot
                ax[i, j].imshow(image, alpha=1, vmin=100.5, cmap='twilight_shifted')
                ax[i, j].imshow(overlay, cmap='viridis', alpha=0.779)
                ax[i, j].axis('off')

            # Add a label above each image in the bottom row
            if i == 1:
                ax[i, j].set_title(labels[j].title(), fontsize=12, y=1.2)

    # Save and display the subplots
    plt.savefig("./visualization/gradcam_visualization.png")
    plt.show()