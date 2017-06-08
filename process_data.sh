#!/bin/bash
set -e
# Set up file choices
IN_FULL_FILE_PREFIXES=( /mnt/ini-nas/DrivingFordMondeo/run3/rec1487433587 /mnt/ini-nas/DrivingFordMondeo/run5/rec1487858093 )
for IN_FULL_FILE_PREFIX in "${IN_FULL_FILE_PREFIXES[@]}"
do
BASE_ID=`basename ${IN_FULL_FILE_PREFIX}`
OUT_FULL_FILE_PREFIX=~/datasets/driving/${BASE_ID}
echo "Working on $OUT_FULL_FILE_PREFIX"
# Export data
ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --keep_frames 1 --keep_events 0 --out_file ${OUT_FULL_FILE_PREFIX}_frames.hdf5
ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize -5000 --keep_frames 0 --keep_events 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin5k.hdf5
ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.010 --keep_frames 0 --keep_events 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin10ms.hdf5
# Prepare and resize
ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_frames.hdf5 --rewrite 1
ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin5k.hdf5 --rewrite 1
ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin10ms.hdf5 --rewrite 1
# Train a network
ipython ./train_test_cnn.py -- --h5file ${OUT_FULL_FILE_PREFIX}_frames.hdf5 --dataset_key aps_frame_60x80 --run_id aps_${BASE_ID}
ipython ./train_test_cnn.py -- --h5file ${OUT_FULL_FILE_PREFIX}_bin5k.hdf5 --dataset_key dvs_frame_60x80 --run_id dvs5k_${BASE_ID}
ipython ./train_test_cnn.py -- --h5file ${OUT_FULL_FILE_PREFIX}_bin10ms.hdf5 --dataset_key dvs_frame_60x80 --run_id dvs10ms_${BASE_ID}
ipython ./train_test_rnn.py -- --h5file ${OUT_FULL_FILE_PREFIX}_bin10ms.hdf5 --dataset_key dvs_frame_60x80 --run_id dvs10ms_${BASE_ID}
ipython ./train_test_rnn.py -- --h5file ${OUT_FULL_FILE_PREFIX}_frames.hdf5 --dataset_key aps_frame_60x80 --run_id aps_${BASE_ID}
ipython ./train_test_rnn.py -- --h5file ${OUT_FULL_FILE_PREFIX}_bin5k.hdf5 --dataset_key dvs_frame_60x80 --run_id dvs5k_${BASE_ID}
done
