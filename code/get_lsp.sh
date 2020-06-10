#!/bin/bash

NEW_DIR=/Users/jkadowaki/data/redshifts
DATA_DIR=/Users/jkadowaki/Documents/github/paper_plots/redshift_paper/data
DATA_FILE=$DATA_DIR/kadowaki2019.tsv

STAMPS=$NEW_DIR/stamps13
mkdir -p $STAMPS


function get_postage_stamp () {   
# Retrieves Postage Stamps for UDGs in $DATA_FILE
    LINK="http://legacysurvey.org/viewer/cutout.jpg?ra=%.4f&dec=%.4f&layer=%s&pixscale=%.2f"
    LAYER=$1
    PS_ZOOM13=0.50
    #PS_ZOOM14=0.25
    #PS_ZOOM15=0.13
    #PS_ZOOM16=0.06
    awk -F'\t' -v LINK=$LINK -v PS=$PS_ZOOM13 -v LAYER=$LAYER -v STAMPS=$STAMPS \
        'FNR>1 {printf LINK,$6,$7,LAYER,PS; print " " STAMPS"/"$5"_"LAYER"_zoom13.jpeg"}' $DATA_FILE | \
        xargs -n2 bash -c 'curl $1 --output $2' bash
}

# Get Postage Stamp Cutouts for Legacy Survey DR8 Images, Models, & Residuals
for LAYER in dr8 dr8-model dr8-resid; 
do
    get_postage_stamp $LAYER
done 
