#!/bin/bash

PROJECT_DIRECTORY=/Users/jennifer_kadowaki/Documents/GitHub/paper_plots/redshift_paper
DATA_DIRECTORY=$PROJECT_DIRECTORY/data
DATA_FILE=$DATA_DIRECTORY/ST82JK.csv  # kadowaki2019.tsv
OUTPUT_DIRECTORY=$PROJECT_DIRECTORY/redshifts
OUTPUT_FILE=$OUTPUT_DIRECTORY/results.csv
mkdir -p $OUTPUT_DIRECTORY

rm OUTPUT_FILE

CONESIZE=3 #arcsec radius

# 2) Search for objects within 1.0 arcmin of Arp 220, sort them by their distances from Arp 220, and display their basic data as a tab-separated ASCII table:
# http://ned.ipac.caltech.edu/cgi-bin/objsearch?search_type=Near+Name+Search&objname=arp+220&radius=1.0&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=Distance+to+search+center&of=ascii_tab



function get_redshift () {

    CONE_DEGREE=$(bc -l <<< 'scale=5;'$CONESIZE'/3600')  # Floating-point Division (%.5f)
   
   QUERY="https://ned.ipac.caltech.edu/tap/sync?query=SELECT+*+FROM+objdir+WHERE+CONTAINS(POINT('J2000',ra,dec),CIRCLE('J2000',%.5f,%.5f,%.5f))=1&REQUEST=doQuery&LANG=ADQL&FORMAT=csv"
    
    echo 'ra,dec,cz' >> $OUTPUT_FILE
   
    # Step 1: Uses awk to print (ra,dec) and format the link and destination
    #         file for each UDG as inputs for Step 2.
    # Step 2: Uses curl to download link and appends result to destination file.
    
    # CHANGE $6,$7 IN LINE 25 AS NEEDED
    # Column $6 in $DATA_FILE: RA
    # Column $7 in $DATA_FILE: dec
    awk -F, -v QUERY=$QUERY -v SEARCH_RADIUS=$CONE_DEGREE -v FILE=$OUTPUT_FILE \
        'FNR>1 {print " "$6" "$7; printf "\""QUERY, $6,$7,SEARCH_RADIUS; print "\" " FILE}' $DATA_FILE | \
        xargs -n4 bash -c 'curl $3 | tail -n+2 | cut -d, -f2,3,12 >> $4' bash
    } #'echo curl $1' bash #


get_redshift

exit 1
