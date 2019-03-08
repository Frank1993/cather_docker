#!/usr/bin/env bash

set -e
OUTPUT_PATH="/home/docker/apollo/test_download/"
WAY_ID_PATH="/home/docker/apollo/csv-inputs/us-new/lane-4.csv"
OSC_DB_PATH="/home/docker/apollo/csv-inputs/db/osc_db_with_new_matching.csv"
MAX_NR_IMGS_PER_WAY_ID=10
MAX_NR_IMGS_PER_SEQ_ID_PER_WAY_ID=5
TOTAL_NR_IMGS=500

echo 'Parameters:'
echo 'OUTPUT_PATH='$OUTPUT_PATH
echo 'WAY_ID_PATH='$WAY_ID_PATH
echo 'OSC_DB_PATH='$OSC_DB_PATH
echo 'MAX_NR_IMGS_PER_WAY_ID='$MAX_NR_IMGS_PER_WAY_ID
echo 'MAX_NR_IMGS_PER_SEQ_ID_PER_WAY_ID='$MAX_NR_IMGS_PER_SEQ_ID_PER_WAY_ID
echo 'TOTAL_NR_IMGS='$TOTAL_NR_IMGS

PYTHONPATH=../../:$PYTHONPATH
export PYTHONPATH


python image_extraction.py \
    --output_path $OUTPUT_PATH \
    --way_id_path $WAY_ID_PATH \
    --osc_db_path $OSC_DB_PATH \
    --max_nr_imgs_per_way_id $MAX_NR_IMGS_PER_WAY_ID \
    --max_nr_imgs_per_seq_id_per_way_id $MAX_NR_IMGS_PER_SEQ_ID_PER_WAY_ID \
    --total_nr_images $TOTAL_NR_IMGS