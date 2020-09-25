#!/bin/bash

# CHANGE LINES 4 & 5 TO YOUR DATA.
DATA_DIR=/Users/jkadowaki/data/smudges
DATA_FILE=$DATA_DIR/ComaJK.csv

FITS=$DATA_DIR/original
JPEG=$DATA_DIR/stamps
mkdir -p $JPEG

# DO NOT CHANGE THESE. THESE ARE SET RESOLUTIONS IN THE LEGACY SURVEY.
PS_ZOOM13=0.50
PS_ZOOM14=0.25
PS_ZOOM15=0.13
PS_ZOOM16=0.06

# CHANGE LINES 17 & 18 IF YOU WANT TO CHANGE THE RESOLUTION.
ZOOM=PS_ZOOM16
ZOOM_LABEL=zoom16

function get_postage_stamp () {   
# Retrieves Postage Stamps for UDGs in $DATA_FILE
    LINK="http://legacysurvey.org/viewer/cutout.jpg?ra=%.4f&dec=%.4f&layer=%s&pixscale=%.2f"
    LAYER=$1
	
	# Step 1: Uses awk to format the link and destination file for each UDG
	# Step 2: Uses curl to download and uses the output from Step 1 as the inputs 
	
	# CHANGE $1, $3, $4 IN LINE 35 AS NEEDED.
	# Column $1 in $DATA_FILE: Name
	# Column $3 in $DATA_FILE: RA
	# Column $4 in $DATA_FILE: dec
    awk -F, -v LINK=$LINK -v PS=$ZOOM -v LABEL=$ZOOM_LABEL -v LAYER=$LAYER -v JPEG=$JPEG \
        'FNR>1 {printf LINK,$3,$4,LAYER,PS; print " " JPEG"/"$1"_"LAYER"_"LABEL".jpeg"}' $DATA_FILE | \
        xargs -n2 bash -c 'curl $1 --output $2' bash
}

# Get Postage Stamp Cutouts for Legacy Survey DR8 Images, Models, & Residuals
for LAYER in dr8 dr8-model dr8-resid; 
do
    get_postage_stamp $LAYER
done 

