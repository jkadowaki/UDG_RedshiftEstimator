#!/usr/bin/env python

import argparse
import glob as g
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

# Import Custom Packages
sys.path.append('/Users/jkadowaki/Documents/github/paper_plots/redshift_paper/code')
from read_data import read_data

# Plotting Style
plt.rc('text', usetex=True)
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)

################################################################################

def parse_args(verbose=True):
    """
    Parses Command Line Arguments
    """
    ######################   Checks for Valid Arguments   ######################
    
    def is_valid_file(parser, arg):
        if not os.path.isfile(arg):
            parser.error("The file {0} does not exist!".format(arg))
        else:
            return str(arg)

    def is_valid_directory(parser, arg):
        if not os.path.isdir(arg):
            parser.error("The directory {0} does not exist!".format(arg))
        else:
            return str(arg)

    ############################################################################

    parser = argparse.ArgumentParser(
                 description="Generates Annotated Figures")
    
    # Required Arguments
    parser.add_argument('data', help="Data File", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('image_dir', help="Image Directory", metavar="DIR",
                        type=lambda x: is_valid_directory(parser, x))

    parser.add_argument('figure_name', help="Figure Filename", metavar="FILE",
                        type=str)

    # Optional Arguments
    parser.add_argument('--table', nargs='+',
                        default=['all'],
                        choices=['2','3','4','smudges','all'],
                        help="Choose from {2,3,4,smudges,all} " \
                             "separated with spaces.")

    parser.add_argument('--environment', nargs='+',
                        default=['all'],
                        choices=['dense',   'sparse',
                                 'cluster', 'non-cluster',
                                 'high',    'low',         'all'],
                        help="Choose one or more from {dense, sparse, " \
                             "cluster, non-cluster, high, low, all} " \
                             "separated with spaces.")

    parser.add_argument('--layer', nargs=1, default='dr8', 
                        choices=['dr8', 'model', 'resid'],
                        help="Choose from {'dr8', 'model', 'resid'}")

    parser.add_argument('--sortby', nargs=1,
                        default='Re',
                        choices=['ra',  'dec', 'cz',  'sepMpc', 'MUg0', 'Mg',
                                 'g-r', 'Re',  'b/a', 'n'],
                        help="Choose from {'ra',  'dec', 'cz',  'sepMpc', " \
                             "'MUg0', 'Mg', 'g-r', 'Re',  'b/a', 'n'}")

    # Flag to select UDGs/candidates
    parser.add_argument('--udgs',       dest='udgs_only', action='store_true')
    parser.add_argument('--candidates', dest='udgs_only', action='store_false')
    
    # Flag to run script verbosely/silently
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--silent',  dest='verbose', action='store_false')
    
    # Set Defaults & Return Command Line Arguments
    parser.set_defaults(udgs_only=True, verbose=False)
    args = parser.parse_args()

    if verbose:
        print("\nParameters:", args)

    return args


################################################################################

def get_data(data_file, table=[2,3,4], udgs_only=True, environment=['all'],
             sort_param='Re', verbose=True):
    """
    """

    if verbose:
        print('\n{0}\n'.format('-'*150))
        print("File:       ", data_file)
        print("Table:      ", table)
        print("Objects:    ", "UDGs" if udgs_only else "Candidates")
        print("Environment:", environment)
        print("Sort By:    ", sort_param, '\n')

    # Load Data from Appropriate Tables
    df_results = read_data(data_file, udg_only=udgs_only)
    df_subset  = df_results.loc[df_results["TABLE"].isin(table)]

    # Filter for Environment
    for env in environment:
        if   env.lower() in ['sparse',  'dense']:
            df_subset = df_subset.loc[df_subset["LocalEnv"]  == env.title()]
        elif env.lower() in ['cluster', 'non-cluster']:
            df_subset = df_subset.loc[df_subset["GlobalEnv"] == env.title()]
        elif env.lower() in ['high',    'low']:
            df_subset = df_subset.loc[df_subset["Density"]   == env.title()]

    # Sort Data
    df_subset = df_subset.sort_values(by=sort_param)
    df_subset = df_subset.reset_index(drop=True)
    
    return df_subset
    

################################################################################

def make_figure(df, image_dir, figure_name, layer='dr8', verbose=True,
                num_cols=5, img_size=5):

    # Annotations
    labels = {"NAME": ("",              ""),
              "MUg0": ("$\mu(g,0)$ = ", " mag arcsec$^{-2}$"),
              "b/a":  ("$b/a$ = ",      " "),
              "n":    ("$n$ = ",        " "),
              "cz":   ("$cz$ = ",       " km/s"),
              "Re":   ("$r_e$ = ",      " kpc")}

    def generate_label(obj, key):
        variable, units = labels[key]
        value = str(np.round(obj[key],1)) if isinstance(obj[key], float) else \
                str(obj[key])             if isinstance(obj[key], int)   else \
                obj[key]
        label = variable + value + units
        return label


    if verbose:
        print(df[list(labels.keys())])

    num_rows = int(np.ceil( len(df) / num_cols ))

    if verbose:
        print("\nFigure Rows:   ", num_rows)
        print(  "Figure Columns:", num_cols)

    fig = plt.figure(figsize=(num_cols*img_size, num_rows*img_size))

    for idx,obj in df[list(labels.keys())].iterrows():
        image_root = os.path.join(image_dir, '*' + obj["NAME"] + '*' + layer + '_*')
        image_file = g.glob(image_root)[0]
        img        = mpimg.imread(image_file)

        subfig  = fig.add_subplot(num_rows, num_cols, idx+1)
        imgplot = subfig.imshow(img, interpolation='none')
        subfig.axis("off")

        # Add Figure Annotations
        subfig.text(0.05, 0.90, generate_label(obj,"NAME"), c='w', transform=subfig.transAxes, fontsize=24, weight='bold')
        subfig.text(0.05, 0.83, generate_label(obj,"cz"),   c='w', transform=subfig.transAxes)
        subfig.text(0.05, 0.29, generate_label(obj,"n"),    c='w', transform=subfig.transAxes)
        subfig.text(0.05, 0.21, generate_label(obj,"b/a"),  c='w', transform=subfig.transAxes)
        subfig.text(0.05, 0.12, generate_label(obj,"Re"),   c='w', transform=subfig.transAxes)
        subfig.text(0.05, 0.05, generate_label(obj,"MUg0"), c='w', transform=subfig.transAxes)
    
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.savefig(figure_name, bbox_inches='tight')

################################################################################


################################################################################

if __name__ == "__main__":

    # EXAMPLE CALL
    """
    $CODE/correlation.py  $DATA/kadowaki2019.csv    \
                          $IMAGE_DIR                \
                          $DATA/../plot/sort_n.pdf  \
                          --udgs                    \
                          --table       all         \
                          --environment all         \
                          --layer       dr8         \
                          --sortby      n           \
                          --verbose
    """
    # Parse Command Line Arguments
    args = parse_args()

    # Computes a separate correlation/p-value matrix for each data subset.
    for table in args.table:
        try:
            subset = [int(table)]
            if int(table) not in [2,3,4]:
                raise Exception("{0} is not a valid table number".format(subset))
        except:
            if table.lower() == 'smudges':
                subset = [2,3]
            elif table.lower() == 'all':
                subset = [2,3,4]
            else:
                raise Exception("{0} is not a valid table name.".format(table))

    df_data = get_data(args.data, table=subset, udgs_only=args.udgs_only,
                       environment=args.environment,
                       sort_param=args.sortby, verbose=args.verbose)
    
    make_figure(df_data, args.image_dir, args.figure_name, layer=args.layer[0],
                num_cols=6, verbose=args.verbose)



################################################################################
