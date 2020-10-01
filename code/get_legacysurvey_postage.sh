#!/bin/bash

# CHANGE LINES 4-5 TO YOUR DATA.
DATA_DIR=/Users/jennifer_kadowaki/Documents/GitHub/UDG_RedshiftEstimator/dataset #/Users/jkadowaki/data/smudges
DATASET=test #training
DATA_FILE=$DATA_DIR/$DATASET.csv

# DO NOT CHANGE THESE. THESE ARE SET RESOLUTIONS IN THE LEGACY SURVEY.
PS_ZOOM13=0.50
PS_ZOOM14=0.25
PS_ZOOM15=0.13
PS_ZOOM16=0.06

# CHANGE LINES 14-15 IF YOU WANT TO CHANGE THE RESOLUTION.
ZOOM=$PS_ZOOM14
ZOOM_LABEL=zoom14
OUTPUT_DIR=$DATA_DIR/$DATASET/$ZOOM_LABEL
mkdir -p $OUTPUT_DIR

# Create TMP file for 2-column $DATA_FILE with Name in Column 1 & cz (or another parameter) in Column 2
# Column $1 in $TMP: Name --> Corresponds to $1 on Line 39
# Column $2 in $TMP: RA   --> Corresponds to $2 on Line 39
# Column $3 in $TMP: dec  --> Corresponds to $3 on Line 39
TMP=tmp.txt
awk -F, '{ RA=substr($1,5,7); DEC=substr($1,13,6); DECSIGN=substr($1,12,1); \
           RADEG=(substr(RA,1,2) + substr(RA,3,2)/60 + substr(RA,5,2)/3600 \
                                 + substr(RA,7)/36000)/24*360; \
           DECDEG=substr(DEC,1,2) + substr(DEC,3,2)/60 + substr(DEC,5,2)/3600; \
         printf "%s,%.5f,%s%.5f\n", $1,RADEG,DECSIGN,DECDEG}' $DATA_FILE > $TMP

function get_postage_stamp () {   
# Retrieves Postage Stamps for UDGs in $DATA_FILE
    LINK="http://legacysurvey.org/viewer/jpeg-cutout?ra=%.5f&dec=%.5f&layer=%s&pixscale=%.2f&bands=grz"
    LAYER=$1
	
	# Step 1: Uses awk to format the link and destination file for each UDG
	# Step 2: Uses curl to download and uses the output from Step 1 as the inputs
    awk -F, -v LINK=$LINK -v PS=$ZOOM -v LABEL=$ZOOM_LABEL -v LAYER=$LAYER -v OUTPUT_DIR=$OUTPUT_DIR \
        'FNR>1 {printf LINK,$2,$3,LAYER,PS; print " " OUTPUT_DIR"/"$1"_"LAYER"_"LABEL".jpg"}' $TMP | \
        xargs -n2 bash -c 'wget -O $2 $1' bash #xargs -n2 bash -c 'curl $1 --output $2' bash
}

# Get Postage Stamp Cutouts for Legacy Survey DR8 Images, Models, & Residuals
for LAYER in dr8 dr8-model dr8-resid; 
do
    get_postage_stamp $LAYER
done 

# Remove TMP file
#rm $TMP
