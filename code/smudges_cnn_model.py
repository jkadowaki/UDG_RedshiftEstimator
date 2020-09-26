#!/usr/bin/env python

from __future__ import print_function, division
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import skimage
from skimage import io, transform, metrics
import sklearn.metrics

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

### NOTE: Confirmed to work for these package versions:
# NumPy 1.19.1
# pandas 0.25.1
# Scikit-Image 0.17.2
# PyTorch 1.5.0, 1.6.0


################################################################################
"""
CONTACT: Jennifer Kadowaki (jkadowaki@email.arizona.edu)
LAST UPDATED: 2020 SEPT 25

TODO:
[-1] Aggregate more data. (!!!!)
[0] Track performance metrics & training loss.
[1] Implement early stopping.
[2] Increase model size since it looks like it's still underfitting the data.
[3] Test whether custom loss function will help.
[4] Implement transfer learning if [3] doesn't work.
"""
################################################################################

################################################################################
#                                                                              #
#                                 DATASET CLASS                                #
#                                                                              #
################################################################################

class SMUDGesDataset(Dataset):
    """
    Custom map-style dataset class for the SMUDGes dataset.
    
    SMUDGesDataset inherits torch.utils.data.Dataset and overrides
    the __len__ and __getitem__ methods.
    """
    
    def __init__(self, csv_file, root_dir, transform=None,
                       zoom=15, image_type='dr8_data'):
        """
        Reads csv_file

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            zoom (int): Image resolution. Each integer-step increases the
                        the resolution by 2x, but images remain fixed pixel size.
                        default: 15 --> 0.12"/pixel. (16 --> 0.06"/pixel)
            image_type (str): "r", "resid", "residual" --> use residual image.
                              "m", "model" --> use model image.
                              anything else --> use Legacy Survey DR8 image.
        """
        self.smudges_cz = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.zoom = zoom
        self.image_type = image_type
    
    
    def __len__(self):
        return len(self.smudges_cz)
    
    
    def __getitem__(self, idx):
        """
        Reads images; separate function allows for more efficient memory usage 
        since all images do not need to be read in at once.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Image Suffixes
        residual = f"_dr8-resid_zoom{self.zoom}.jpeg"
        model    = f"_dr8-model_zoom{self.zoom}.jpeg"
        dr8_data = f"_dr8_zoom{self.zoom}.jpeg"
        suffix   = residual  if self.image_type in ["r", "resid", "residual"] \
                   else model if self.image_type in ["m", "model"] \
                   else dr8_data

        # Retrieves the name of the file associated with object in index idx
        obj_name = self.smudges_cz.loc[self.smudges_cz.index[idx],"NAME"]
        img_name = os.path.join(self.root_dir, obj_name + suffix)
        image    = io.imread(img_name)
        
        # Retrieves the redshift of object at index idx
        cz  = self.smudges_cz.loc[self.smudges_cz.index[idx],"cz"]
        udg = {'image': image.astype(np.float32), 'cz': cz.astype(np.float32)}
        
        if False:
            print("NAME: ", obj_name)
        
        if self.transform:
            udg = self.transform(udg)

        return udg


################################################################################

################################################################################
#                                                                              #
#                          IMAGE AUGMENTATION CLASSES                          #
#                                                                              #
################################################################################

class Random90Rotation:
    """Randomly rotates image by angles of 90 degree increments."""

    def __call__(self, udg):
        image, cz     = udg['image'], udg['cz']
        num_rotations = np.random.randint(4)
        
        image = np.rot90(image, k=num_rotations)
        
        return {'image': image, 'cz': cz}


################################################################################

class RandomFlip:
    """Randomly rotates image by angles of 90 degree increments."""

    def __call__(self, udg):
        image, cz = udg['image'], udg['cz']
        image     = image if random.getrandbits(1) else np.fliplr(image)
        
        return {'image': image, 'cz': cz}


################################################################################

class RandomShift(object):
    """Shifts the image by up to `shift_pix` number of pixels horizontally and
        vertically. Image is then edge padded to its original size.

    Args:
        shift_pix (int): The maximum number of pixels to shift.
    """

    def __init__(self, shift_pix):
        self.shift_pix = shift_pix
    
    def __call__(self, udg):
        
        def shift_image(image, shift, vertical=True):
            """
            Shifts the image by `shift` pixels in the specified direction.
            Direction is specified by the sign of `shift` and the `vertical` flag.

            Args:
                image (np.ndarray): 2D or 3D Image
                shift (int): Number of pixels to shift the image.
                             Negative value shifts the image up or to the left
                                 depending on whether `vertical` is True.
                             Positive values shifts image down or to the right.
                vertical (bool): Flag to indicate whether to shift an image
                                 vertically or horizontally.
            """
            new_image = image.copy() if vertical else \
                        image.copy().transpose((1,0,2)) \
                        if len(image.shape)==3 else image.copy().T
            dimension = new_image.shape[0]

            if shift < 0:
                new_image[:dimension+shift,:,:] = new_image[-shift:,:,:]
                new_image[dimension+shift:] = new_image[dimension+shift-1]
            else:
                new_image[shift:,:,:] = new_image[:dimension-shift,:,:]
                new_image[:shift] = new_image[shift]

            return new_image if vertical else \
                  new_image.transpose((1,0,2)) \
                  if len(image.shape)==3 else new_image.T

        # Generates Number of Pixels for Horizontal & Vertical Shifts
        horizontal_shift = np.random.randint(-self.shift_pix,
                                              self.shift_pix+1)
        vertical_shift   = np.random.randint(-self.shift_pix,
                                              self.shift_pix+1)
        
        # Computes Image Shift
        image, cz = udg['image'], udg['cz']
        image     = shift_image(image, vertical_shift)
        image     = shift_image(image, horizontal_shift, vertical=False)

        return {'image': image, 'cz': cz}


################################################################################

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, udg):
        image, cz = udg['image'], udg['cz']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        new_image = image.copy().transpose((2, 0, 1))
        
        return { 'image': torch.from_numpy(new_image),
                  'cz': torch.Tensor([[cz]]) }


################################################################################

################################################################################
#                                                                              #
#                             IMAGE DISPLAY METHODS                            #
#                                                                              #
################################################################################

def display_smudges(smudges_dataset, max_display=8,
                    output='smudges_sample.pdf'):
    """
     Display `max_display` number of images
     """
    
    num_display = min(max_display, len(smudges_dataset))
    
    fig, axes = plt.subplots(1,num_display,figsize=(2*num_display,2))
    print("Index \tDimension \tRedshift (km/s)")

    for i in range(num_display):
        udg = smudges_dataset[i]
        
        print(f"{i}  \t{udg['image'].shape} \t{udg['cz']}")
        
        axes[i].axis('off')
        axes[i].imshow(udg['image']/256)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(output, bbox_inches='tight')


################################################################################

def display_batches(smudges_dataset, num_display=8,
                    output='smudges_sample.pdf'):


    fig, axes = plt.subplots(2, num_display,
                             sharex=True, sharey=True,
                             figsize=(2*num_display,0.5*num_display))
    print("Index \tOriginal \tTransformed \tRedshift (km/s)")

    for i in range(num_display):
        original_udg    = smudges_dataset[i]
        transformed_udg = visual_dataset[i]
        
        print(f"{i}   \t{original_udg['image'].shape} \t{transformed_udg['image'].shape} \t{original_udg['cz']}")
        
        axes[0,i].set_ylabel('UDG #{0}'.format(i))
        axes[0,i].axis('off')
        axes[1,i].axis('off')
        axes[0,i].imshow(original_udg['image']/256)
        axes[1,i].imshow(transformed_udg['image']/256)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(output, bbox_inches='tight')


################################################################################

# Helper function to show a batch
def show_batch(sample_batched, max_value=256):
    """Show images in a batch of samples."""
    images_batch, cz_batch = sample_batched['image'], sample_batched['cz']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0))/max_value)

    for i in range(batch_size):
        plt.title('Batch from dataloader')

def display_batch(dataloader, batch_size=16):
    for i_batch, sample_batched in enumerate(dataloader):
        print("\n", i_batch, sample_batched['image'].size(),
              sample_batched['cz'].size())
        
        plt.figure()
        show_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()

        if i_batch == batch_size:
            break


################################################################################

################################################################################
#                                                                              #
#                           REDSHIFT ESTIMATION MODEL                          #
#                                                                              #
################################################################################

class SMUDGes_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1       = nn.Conv2d(3, 3, kernel_size=8, stride=2, padding=2)
        self.leaky_relu1 = nn.LeakyReLU(0.05)
        self.batch_norm1 = nn.BatchNorm2d(3)
        self.conv2       = nn.Conv2d(3, 3, kernel_size=6, stride=2, padding=2)
        self.leaky_relu2 = nn.LeakyReLU(0.01)
        self.batch_norm2 = nn.BatchNorm2d(3)
        self.conv3       = nn.Conv2d(3, 3, kernel_size=4, stride=1, padding=2)
        self.leaky_relu3 = nn.LeakyReLU(0.01)
        self.batch_norm3 = nn.BatchNorm2d(3)
        self.conv4       = nn.Conv2d(3, 1, kernel_size=2, stride=1, padding=2)
        self.leaky_relu4 = nn.LeakyReLU(0.01)
        #self.batch_norm4 = nn.BatchNorm2d(3)
        self.dropout4    = nn.Dropout(p=0.05)
        self.fc3         = nn.Linear(16,16)
        self.fc4         = nn.Linear(16,4)
        self.dropout5    = nn.Dropout(p=0.05)
        self.fc5         = nn.Linear(4,1)

    def forward(self, image):       
        image = image.view(-1, 3, 256, 256)
        image = self.batch_norm1(image)
        #print("Step 0", image.shape)
        
        image = self.conv1(image)
        image = self.leaky_relu1(image)
        #image = self.batch_norm1(image)
        #print("Step 1", image.shape)
        
        image = self.conv2(image)
        image = self.leaky_relu2(image)
        image = self.batch_norm2(image)
        #print("Step 2", image.shape)        
        
        image = F.max_pool2d(image, 4)
        image = self.conv3(image)
        image = self.leaky_relu3(image)
        #image = self.batch_norm3(image)
        #print("Step 3", image.shape)

        image = self.conv4(image)
        image = self.leaky_relu4(image)
        image = F.max_pool2d(image, 4)
        #print("Step 4", image.shape)  
        
        # Fully Connected Layers
        image = image.view(-1, 1, image.shape[2] * image.shape[3])
        image = self.dropout4(image)
        image = self.fc3(image)
        image = self.dropout4(image)
        image = self.fc4(image)
        image = self.dropout5(image)
        image = self.fc5(image)
        #print("Step 5", image.shape)  
        
        return image.view(-1,1,1)


################################################################################

def loss_batch(model, loss_func, image, cz, opt=None):
    
    loss = loss_func(model(image), cz)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(image)


################################################################################

def fit(epochs, model, loss, opt, train_dl, valid_dl):
    
    print("EPOCH\t LOSS")
    
    for epoch in range(epochs):
        
        model.train()
        for batch_idx, batched_data in enumerate(train_dl):
            image = batched_data['image']
            cz    = batched_data['cz']
            
            loss_batch(model, loss, image, cz, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[ loss_batch( model,               \
                                              loss,                \
                                              batch_data['image'], \
                                              batch_data['cz'])    \
                                  for batch_data in valid_dl] )
        
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, "\t", val_loss)


################################################################################

################################################################################
#                                                                              #
#                                 PLOT RESULTS                                 #
#                                                                              #
################################################################################

def generate_predictions(MODEL, dataloader, verbose=True):
    if verbose:
        print("True\t Predicted")
    true = []
    pred = []

    for idx in range(10):
        for batch in dataloader:
            image = batch['image']
            cz    = batch['cz']

            pred  = MODEL(image)
            for item in zip(cz, pred):
                true.append(int(item[0].item()))
                pred.append(np.round(item[1].item(),2))
                if verbose:
                    print(true[-1], "\t", pred[-1])

    return true, pred


################################################################################

def plot_true_vs_predicted(MODEL, train_dataloader, valid_dataloader,
                           plot_fname="true_vs_pred.pdf", verbose=True):

    train_true, train_pred = generate_predictions(MODEL, train_dataloader, verbose=verbose)
    valid_true, valid_pred = generate_predictions(MODEL, valid_dataloader, verbose=verbose)
    
    # Create Figure
    plt.figure(figsize=(10,10))
    
    # Plot 1:1 Line
    plt.plot([-2000,12000], [-2000,12000], c='g', linestyle='-')
    
    # Plot Training & Validation Results
    plt.scatter(training_true, training_pred,
                c='b', marker='.', s=15, label="Training")
    plt.scatter(validation_true, validation_pred,
                c='r', marker='o', s=50, label="Validation")
    
    # Plot Formatting
    plt.xlabel("True cz (km/s)",      fontsize=20)
    plt.ylabel("Predicted cz (km/s)", fontsize=20)
    plt.xlim(-2000,12000)
    plt.ylim(-2000,12000)
    plt.legend()
    
    plt.savefig(plot_fname)


################################################################################

def pipeline():

    # LOAD DATA
    ROOT_DIR            = '/Users/jkadowaki/dataset'
    transformed_dataset = SMUDGesDataset(
                          csv_file=os.path.join(ROOT_DIR, "training.csv"),
                          root_dir=os.path.join(ROOT_DIR, "training", "zoom15"),
                          transform=transforms.Compose([
                                        RandomFlip(),
                                        Random90Rotation(),
                                        RandomShift(2),
                                        ToTensor() ]),
                          zoom=15,
                          image_type='dr8_data' )

    """
    smudges_dataset = SMUDGesDataset(
                        csv_file=os.path.join(ROOT_DIR, "training.csv"),
                        root_dir=os.path.join(ROOT_DIR, "training", "zoom15"),
                        transform=transforms.Compose([
                                   RandomFlip(),
                                   Random90Rotation(),
                                   RandomShift(50))
    display_smudges(smudges_dataset)
    
    # split data into training/validation
    # create dataloaders for both datasets
    display_batch(train_dataloader)
    display_batch(valid_dataloader)
    """

    # DATA PARAMETERS
    DATASET_SIZE      = 68
    BATCH_SIZE        = 16
    TRAINING_FRACTION = 0.8
    NUM_NODES         = 2

    # MODEL PARAMETERS
    MODEL          = SMUDGes_CNN()
    AUGMENT_FACTOR = 2 * 4 * 25
    ITERATIONS     = 2
    EPOCHS         = int(ITERATIONS * AUGMENT_FACTOR * DATASET_SIZE/BATCH_SIZE)
    LEARNING_RATE  = 0.0002
    DECAY          = 0.25
    LOSS_FUNC      = nn.MSELoss()
    OPTIMIZER      = optim.Adam( MODEL.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=DECAY, amsgrad=False )

    # SPLIT DATA INTO TRAINING / VALIDATION
    train_size = int(TRAINING_FRACTION * len(transformed_dataset))
    valid_size = len(transformed_dataset) - train_size
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
                                   transformed_dataset, [train_size, valid_size])

    train_dataloader  = DataLoader( train_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=NUM_NODES )
    valid_dataloader  = DataLoader( valid_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=NUM_NODES )

    # TRAIN & EVALUATION
    fit(EPOCHS, MODEL, LOSS_FUNC, OPTIMIZER, train_dataloader, valid_dataloader)

    # SAVE MODEL
    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    MODEL_PATH   = 'smudges_redshift_{0}.pt'.format(datetime_str)

    torch.save(MODEL.state_dict(), MODEL_PATH)
    torch.save({
                'epoch':                EPOCHS,
                'model_state_dict':     MODEL.state_dict(),
                'optimizer_state_dict': OPTIMIZER.state_dict(),
                'loss': LOSS_FUNC
                }, MODEL_PATH)
                
    # GENERATE PREDICTIONS
    plot_true_vs_predicted(MODEL, train_dataloader, valid_dataloader,
                           plot_fname="true_vs_pred.pdf", verbose=True)


################################################################################

if __name__ == "__main__":
    pipeline()


################################################################################

# This call sets the network to 'evaluation' mode, which effects
# batchnorm layers slightly. In training mode, the network can't
# operate on a single example at a time, it needs batches to be
# well defined.
"""
net.eval()
with torch.no_grad(): # `torch.no_grad` explained in automatic differentiation
    all_outs = net(test_features)
test_prediction = all_outs.argmax(dim=1)

accuracy = sklearn.metrics.accuracy_score(test_labels,test_prediction)
print("CONVOLUTIONAL NETWORK")
print("Overall accuracy:",accuracy*100,"%")
print("Classification Report:")
print(sklearn.metrics.classification_report(test_labels,test_prediction,digits=4))

print("Parameter count:", sum(p.nelement() for p in net.parameters()))
for name,param in net.named_parameters():
    print(list(param.shape),name)
"""

################################################################################
