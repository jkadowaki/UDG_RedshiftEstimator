#!/usr/bin/env python

################################################################################
"""
CONTACT: Jennifer Kadowaki (jkadowaki@email.arizona.edu)
LAST UPDATED: 2020 SEPT 26

TODO:
[0] Aggregate more data. (!!!!)
[1] Increase model size since it looks like it's still underfitting the data.
[2] Test whether custom loss function will help.
[3] Implement transfer learning if [3] doesn't work.

Recently Implemented:
[0] Track performance metrics & training loss. (2020 SEPT 25)
[1] Implement early stopping. (2020 SEPT 26)
"""
################################################################################

from __future__ import print_function, division
#import argparse
from collections import OrderedDict
import copy
import csv
from datetime import datetime
import glob as g
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
#import skimage
from skimage import io, transform, metrics
#import sklearn.metrics

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Matplotlib Fonts
plt.rcParams.update({ "text.usetex": True,
                      "text.latex.unicode": True,
                      "font.family": "sans-serif",
                      "font.sans-serif": ["Helvetica"] })

# CONSTANTS
SPEED_OF_LIGHT = 299792.458  # [in units of km/s]

### NOTE: Confirmed to work for these package versions:
# NumPy 1.19.1
# pandas 0.25.1
# Scikit-Image 0.17.2
# PyTorch 1.5.0, 1.6.0

# sudo pip install --user virtualenv  # Install venv (Python3)/ virtual environment (Python2)
# python3 -m venv smudges_env         # Create a new virtual environment
# source activate smudges_env         # Activate virtual environment
# sudo pip install numpy==1.19.1
# sudo pip install pandas==0.25.1
# sudo pip install scikit-image==0.17.2
# sudo pip install torch==1.6.0

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
        self.smudges_cz  = pd.read_csv(csv_file)
        self.root_dir   = root_dir
        self.transform  = transform
        self.zoom       = zoom
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
        suffix   = residual   if self.image_type in ["r", "resid", "residual"] \
                   else model if self.image_type in ["m", "model"] \
                   else dr8_data

        # Retrieves the name of the file associated with object in index idx
        obj_name = self.smudges_cz.loc[self.smudges_cz.index[idx],"NAME"]
        img_name = os.path.join(self.root_dir, obj_name + suffix)
        image    = io.imread(img_name)
        
        # Retrieves the redshift of object at index idx
        cz   = self.smudges_cz.loc[self.smudges_cz.index[idx],"cz"]
        udg = { 'image': image.astype(np.float32),
                'cz':    cz.astype(int) }
        
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
    """
    Show images in a batch of samples.
    """

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
    
    # Model needs updating. See link below:
    # https://medium.com/@sundeep.laxman/perform-regression-using-transfer-learning-to-predict-house-prices-97e432a66ba5
    models.resnet18(pretrained=True)
    """
    
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
        #image = self.leaky_relu2(image)
        #image = self.batch_norm2(image)
        #print("Step 2", image.shape)        
        
        image = F.max_pool2d(image, 4)
        image = self.conv3(image)
        image = self.leaky_relu3(image)
        image = self.batch_norm3(image)
        #print("Step 3", image.shape)

        image = self.conv4(image)
        image = self.leaky_relu4(image)
        image = F.max_pool2d(image, 4)
        #print("Step 4", image.shape)  
        
        # Fully Connected Layers
        image = image.view(-1, 1, image.shape[2] * image.shape[3])
        #image = self.dropout4(image)
        image = self.fc3(image)
        #image = self.dropout4(image)
        image = self.fc4(image)
        #image = self.dropout5(image)
        image = self.fc5(image)
        #print("Step 5", image.shape)  
        
        return image.view(-1,1,1)
    
    """
################################################################################

def build_model(arch, num_classes=1, hidden_units=1024):
    """
    Load a pretrained model with only the final layer
    replaced by a user-defined classifier
    :param arch - a string specifying the type of model architecture
    :param num_classes - an integer specifying the number of class labels
    :param hidden_units - an integer specifying the size
    return - a pretrained model with a user-defined classifier
    """
    in_features = 0
    try:
          # model = eval("models." + arch + "(pretrained=True)")
          model = models.__dict__[arch](pretrained=True)
    except:
        raise Exception('Invalid architecture specified')
      
    # Freeze parameters as only the final layer is being trained
    for param in model.parameters():
        param.require_grad = False
    # extract the last layer in the model
    last_layer = list(model.children())[-1]
    if isinstance(last_layer, nn.Sequential):
        count = 0
        for layer in last_layer:
            if isinstance(layer, nn.Linear):
                # fetch the first of the many Linear layers
                count += 1
                in_features = layer.in_features
            if count == 1:
                break
    elif isinstance(last_layer, nn.Linear):
        in_features = last_layer.in_features
    # define the new classifier
    classifier = nn.Sequential(OrderedDict([
                            ('bc1', nn.BatchNorm1d(in_features)),
                            ('relu1', nn.ReLU()),
                            ('fc1', nn.Linear(in_features, num_classes, bias=True)),
    ]))
    # replace the existing classifier in thelast layer with the new one
    if model.__dict__['_modules'].get('fc', None):
        model.fc = classifier
    else:
        model.classifier = classifier

    return model


################################################################################

################################################################################
#                                                                              #
#                                 TRAINING MODEL                               #
#                                                                              #
################################################################################

def fit(epochs, model, loss_func, opt, patience, train_dl, valid_dl,
        model_directory='checkpoints'):

    # Tracks Model Metrics
    metrics_dict = {}
    
    # Early Stopping Metrics
    last_val_model = None
    min_val_loss   = 99999999999999999999999 # Initialize to Large Number
    no_improvement = 0
    
    
    print("\n---------------------------------------------------------------------------")
    print(  "EPOCH    TRAINING_LOSS   VALIDATION_LOSS   TRAINING_ERROR  VALIDATION_ERROR")
    print(  "---------------------------------------------------------------------------")
    for epoch in range(epochs):
        
        # Training Metrics
        training_loss   = 0.
        training_images = 0
        training_error  = 0.
        
        # Switches Model to Training Mode
        model.train()
        
        # Computes Gradients & Updates Model After Every Mini-Batch
        for batched_data in train_dl:
            
            # Load Batched Data
            image = batched_data['image']
            cz    = batched_data['cz']
            
            # Computes Total Loss between Model Prediction & Labels
            # Note: This is total loss bc we define it as the sum (i.e., not the
            #       default mean) with parameter reduction='sum' in LOSS_FUNC.
            prediction = model(image)
            loss = loss_func(prediction, cz)
            # Removes Gradients from Previous Batch
            opt.zero_grad()
            # Compute Gradients
            loss.backward()
            # Update Model
            opt.step()
            
            # Tracks Cumulative Loss Per Epoch
            training_loss   += loss.item()
            training_images += len(image)

            # Tracks Cumulative Training Percent Error
            training_error  += th.sum(100 * th.abs((prediction - cz) / cz)).item()
    


        # Validation Metrics
        validation_loss   = 0.
        validation_images = 0
        validation_error  = 0.

        # Switches Model to Evaluation Mode
        model.eval()
        
        # Turn off Gradient Calculations (i.e., Do not update model parameters)
        with torch.no_grad():
            
            # Tracks Validation Losses Across Every Mini-Batch
            for batched_data in valid_dl:
            
                # Load Batched Data
                image = batched_data['image']
                cz    = batched_data['cz']
            
                # Computes Total Loss between Model Prediction & Labels
                prediction = model(image)
                loss       = loss_func(prediction, cz)
                
                # Tracks Cumulative Loss Per Epoch
                validation_loss   += loss.item()
                validation_images += len(image)
    
                # Tracks Cumulative Validation Percent Error
                validation_error  += th.sum(100 * th.abs((prediction - cz) / cz)).item()
    
    
        # Compute Training & Evaluation Losses Per Epoch
        train_loss_per_epoch = training_loss   / training_images
        valid_loss_per_epoch = validation_loss / validation_images
        training_error      /= training_images
        validation_error    /= validation_images
        metrics_dict[epoch]  = [train_loss_per_epoch, valid_loss_per_epoch, training_error, validation_error]
        
        
        # Early Stopping
        if valid_loss_per_epoch < min_val_loss:
            
            # Resets Counter & Minimum Validation Loss Observed
            min_val_loss   = valid_loss_per_epoch
            no_improvement = 0
            try:
                os.remove(last_val_model)
            except:
                pass

            # SAVE MODEL
            datetime_str   = datetime.now().strftime("%Y%m%d%H%M%S")
            MODEL_PATH     = os.path.join( model_directory,
                                'smudges_redshift_{0}.pt'.format(datetime_str) )
            last_val_model = MODEL_PATH
            
            torch.save({ 'epoch':                epoch,
                         'model_state_dict':     model.state_dict(),
                         'optimizer_state_dict': opt.state_dict(),
                         'loss':                 loss
                       }, MODEL_PATH)
        else:
            no_improvement += 1
        
        print(f"{epoch} \t {train_loss_per_epoch:>10.5f} \t {valid_loss_per_epoch:>10.5f} \t {training_error:>9.3f}% \t {validation_error:>9.3f}%" )

        # Stops Training if Number of Epochs without Validation Loss Improvement > Patience
        if no_improvement >= patience:
            print(f"Training Ended Early Due to No Performance Gains Since Epoch {epoch-patience}.")
            break
    
    print("---------------------------------------------------------------------------\n")

    return metrics_dict


################################################################################

################################################################################
#                                                                              #
#                                 PLOT RESULTS                                 #
#                                                                              #
################################################################################

def generate_predictions(MODEL, dataloader, verbose=True, num_augmentations=10):
    
    if verbose:
        print("True\t Predicted")

    true = []
    pred = []

    for idx in range(num_augmentations):
        
        for batch in dataloader:
            image    = batch['image']
            cz       = batch['cz']
            estimate = MODEL(image)
            
            for item in zip(cz, estimate):
                #print(true)
                #print(pred)
                
                true.append(np.round(item[0].item(),4))
                pred.append(np.round(item[1].item(),4))
                if verbose:
                    print(true[-1], "\t", pred[-1])

    return true, pred


################################################################################

def plot_true_vs_predicted(MODEL, train_dataloader, valid_dataloader,
                           plot_fname="true_vs_pred.pdf", verbose=True,
                           save_predictions=True):

    train_true, train_pred = generate_predictions(MODEL, train_dataloader, verbose=verbose)
    valid_true, valid_pred = generate_predictions(MODEL, valid_dataloader, verbose=verbose)
    
    # Create Figure
    plt.figure(figsize=(10,10))
    
    # Plot Training & Validation Results
    plt.scatter(train_true, train_pred, c='b', marker='.', s=15, label="Training")
    plt.scatter(valid_true, valid_pred, c='r', marker='o', s=50, label="Validation")
    
    # Adjust Plot Bounds
    all_values = train_true + train_pred + valid_true + valid_pred
    minimum   = min(all_values)
    maximum   = max(all_values)
    min_bound = min(0.9 * minimum, 1.1 * minimum)
    max_bound = max(0.9 * maximum, 1.1 * maximum)
    plt.xlim(min_bound, max_bound)
    plt.ylim(min_bound, max_bound)
    
    # Plot 1:1 Line w/in Plot Bounds
    plt.plot([min_bound,max_bound], [min_bound,max_bound], c='g', linestyle='-')
    
    # Plot Formatting
    plt.xlabel(r"True cz (km/s)")
    plt.ylabel(r"Predicted cz (km/s)")
    plt.legend()
    
    plt.savefig(plot_fname, bbox_inches='tight')

    if save_predictions:
    
        def z_to_cz(z_list):
            return list(SPEED_OF_LIGHT * np.array(z_list))
        
        directory     = os.path.dirname(plot_fname)
        train_results = os.path.join(directory,  "train_results.csv")
        valid_results = os.path.join(directory,  "valid_results.csv")
        train_rows    = zip(train_true, train_pred) #zip(z_to_cz(train_true), z_to_cz(train_pred))
        valid_rows    = zip(valid_true, valid_pred) #zip(z_to_cz(valid_true), z_to_cz(valid_pred))
        header        = ("true_cz", "predicted_cz")
        write_to_file(train_results, train_rows, header)
        write_to_file(valid_results, valid_rows, header)


################################################################################

def plot_learning_curve(metrics_dict, plot_fname='learning_curve.pdf',
                        save_metrics=True):

    # Load Metrics
    epochs = list(metrics_dict.keys())
    training_loss, validation_loss, training_error, validation_error = list(zip(*metrics_dict.values()))

    # Create Figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    # Plot Learning Curves
    axes[0].plot(epochs, training_loss,    c='b', label=r"Training")
    axes[0].plot(epochs, validation_loss,  c='r', label=r"Validation")
    axes[1].plot(epochs, training_error,   c='b', label=r"Training")
    axes[1].plot(epochs, validation_error, c='r', label=r"Validation")
    
    # Plot Early Stopping Epoch
    vl_epoch   = [e for e,vl in zip(epochs, validation_loss)  if vl==min(validation_loss)][0]
    ve_epoch   = [e for e,ve in zip(epochs, validation_error) if ve==min(validation_error)][0]
    ymin, ymax = axes[0].get_ylim()
    axes[0].plot([vl_epoch, vl_epoch], [ymin, ymax], c='g', linestyle='--', label=r'Val Loss  Epoch')
    axes[0].plot([ve_epoch, ve_epoch], [ymin, ymax], c='y', linestyle='-.', label=r'Val Error Epoch')
    ymin, ymax = axes[1].get_ylim()
    axes[1].plot([vl_epoch, vl_epoch], [ymin, ymax], c='g', linestyle='--', label=r'Val Loss  Epoch')
    axes[1].plot([ve_epoch, ve_epoch], [ymin, ymax], c='y', linestyle='-.', label=r'Val Error Epoch')


    # Plot Formatting
    axes[0].set_xlabel(r"Epochs")
    axes[0].set_ylabel(r"MSE Loss")
    axes[1].set_xlabel(r"Epochs")
    axes[1].set_ylabel(r"Percent Error (\%)")
    axes[0].legend()
    axes[1].legend()
    
    # Save Figure
    plt.savefig(plot_fname, bbox_inches='tight')

    # Write Metrics to CSV File
    if save_metrics:
        csv_file  = plot_fname.replace('pdf', 'csv')
        data_rows = zip( epochs, training_loss,  validation_loss,
                                 training_error, validation_error )
        header    = ("epoch", "training_loss",  "validation_loss",
                              "training_error", "validation_error")
        write_to_file(csv_file, data_rows, header)


################################################################################

def write_to_file(filename, data_rows, header):
    """
    filename (str): Filename to save data
    data_rows (zip): Zip object containing data column lists (e.g., zip(colA, colB)
    header (list): List with column names for header
    """
    
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)


################################################################################

################################################################################
#                                                                              #
#                   MODEL PIPELINE FOR TRAINING & EVALUATION                   #
#                                                                              #
################################################################################

def pipeline(train_model=True, load_checkpoint=True):


    # LOAD DATA
    ROOT_DIR            = "/Users/jennifer_kadowaki/Documents/GitHub/UDG_RedshiftEstimator/dataset" #"/Users/jkadowaki/dataset"
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
    NUM_NODES         = 4

    # MODEL PARAMETERS
    PROJECT         = "/Users/jennifer_kadowaki/Documents/GitHub/UDG_RedshiftEstimator" #"/Users/jkadowaki/Documents/github/UDG_RedshiftEstimator"
    MODEL_DIRECTORY = os.path.join(PROJECT, "checkpoints")
    MODEL           = build_model("resnet18", hidden_units=64) # SMUDGes_CNN()
    AUGMENT_FACTOR  = 2 * 4 * 25
    ITERATIONS      = 2
    EPOCHS          = int(ITERATIONS * AUGMENT_FACTOR * DATASET_SIZE/BATCH_SIZE)
    LEARNING_RATE   = 10**-4
    DECAY           = 0.25
    LOSS_FUNC       = nn.MSELoss(reduction='sum')
    OPTIMIZER       = optim.Adam( MODEL.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=DECAY, amsgrad=False )
    PATIENCE        = 15


    # RESULTS PARAMETERS
    RESULTS_DIRECTORY   = os.path.join(PROJECT, "results")
    RESULTS_FILE        = os.path.join(RESULTS_DIRECTORY, "true_vs_pred.pdf")
    LEARNING_CURVE_FILE = os.path.join(RESULTS_DIRECTORY, "learning_curve.pdf")


    # SPLIT DATA INTO TRAINING / VALIDATION
    train_size = int(TRAINING_FRACTION * len(transformed_dataset))
    valid_size = len(transformed_dataset) - train_size
        
    train_dataset, valid_dataset = torch.utils.data.random_split(
                                   transformed_dataset, [train_size, valid_size])
            
    train_dataloader  = DataLoader( train_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True,  num_workers=NUM_NODES )
    valid_dataloader  = DataLoader( valid_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True,  num_workers=NUM_NODES )


    # LOAD PRETRAINED MODEL TO CONTINUE TRAINING OR
    if load_checkpoint:
        
        LATEST_CHECKPOINT = max(g.glob(os.path.join(MODEL_DIRECTORY,'*')))
        checkpoint_dict   = torch.load(LATEST_CHECKPOINT)
        
        EPOCHS    = checkpoint_dict['epoch']
        LOSS_FUNC = checkpoint_dict['loss']
        MODEL.load_state_dict(checkpoint_dict['model_state_dict'])
        OPTIMIZER.load_state_dict(checkpoint_dict['optimizer_state_dict'])


    # TRAIN MODEL
    if train_model:

        # TRAIN & EVALUATION
        MODEL.train()
        metrics_dict = fit(EPOCHS, MODEL, LOSS_FUNC, OPTIMIZER, PATIENCE,
                           train_dataloader, valid_dataloader, MODEL_DIRECTORY)
        plot_learning_curve(metrics_dict, plot_fname=LEARNING_CURVE_FILE)



    # GENERATE PREDICTIONS
    MODEL.eval()
    plot_true_vs_predicted(MODEL, train_dataloader, valid_dataloader,
                           plot_fname=RESULTS_FILE, verbose=True)


################################################################################

################################################################################

if __name__ == "__main__":
    
    # Train & Predict
    pipeline(train_model=True, load_checkpoint=False)

    # Predict only from pre-trained model
    # pipeline(train_model=False, load_checkpoint=True)


################################################################################
"""
https://scikit-learn.org/stable/modules/model_evaluation.html
print("Parameter count:", sum(p.nelement() for p in net.parameters()))
"""
################################################################################
