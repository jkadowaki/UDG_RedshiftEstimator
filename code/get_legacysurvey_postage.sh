#!/bin/bash

DATA_DIR=/Users/jkadowaki/data/smudges
DATA_FILE=$DATA_DIR/ComaJK.csv

FITS=$DATA_DIR/original
JPEG=$DATA_DIR/stamps
mkdir -p $JPEG


function get_postage_stamp () {   
# Retrieves Postage Stamps for UDGs in $DATA_FILE
    LINK="http://legacysurvey.org/viewer/cutout.jpg?ra=%.4f&dec=%.4f&layer=%s&pixscale=%.2f"
    LAYER=$1
    #PS_ZOOM13=0.50
    #PS_ZOOM14=0.25
    #PS_ZOOM15=0.13
    PS_ZOOM16=0.06
    awk -F, -v LINK=$LINK -v PS=$PS_ZOOM16 -v LAYER=$LAYER -v JPEG=$JPEG \
        'FNR>1 {printf LINK,$3,$4,LAYER,PS; print " " JPEG"/"$1"_"LAYER"_zoom15.jpeg"}' $DATA_FILE | \
        xargs -n2 bash -c 'curl $1 --output $2' bash
}

# Get Postage Stamp Cutouts for Legacy Survey DR8 Images, Models, & Residuals
for LAYER in dr8 dr8-model dr8-resid; 
do
    get_postage_stamp $LAYER
done 
