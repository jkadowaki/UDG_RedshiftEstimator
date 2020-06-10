#!/bin/bash

PROJECT_DIR=/Users/jkadowaki/Documents/github/paper_plots/redshift_paper
CODE=.
DATA=$PROJECT_DIR/data
IMG_DIR=/Users/jkadowaki/data/redshifts/stamps15 
PLOT_DIR=$PROJECT_DIR/plots

$CODE/display_postage_stamps.py  $DATA/kadowaki2019.tsv        \
                                 $IMG_DIR                      \
                                 $PLOT_DIR/udgs_sortby-RA.pdf  \
								 --udgs                        \
								 --table all                   \
								 --environment all             \
								 --layer dr8                   \
								 --sortby ra                   \
								 --verbose
