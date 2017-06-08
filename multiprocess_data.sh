#!/bin/bash
set -e
OUT_DIR=/home/dneil/datasets/driving
ORIGIN_DIR=/mnt/ini-nas/DrivingFordMondeo
TODO_FILES=( run5/rec1487839456.hdf5 run5/rec1487844247.hdf5 run5/rec1487849151.hdf5 run5/rec1487856408.hdf5 run5/rec1487858093.hdf5 run5/rec1487842276.hdf5 run5/rec1487846842.hdf5 run5/rec1487849663.hdf5 run5/rec1487857941.hdf5 run5/rec1487860613.hdf5 run5/rec1487864316.hdf5 )
for TODO_FILE in "${TODO_FILES[@]}"
do
    IN_FULL_FILE_PREFIX=${ORIGIN_DIR}/${TODO_FILE%.*}
    BASE_ID=`basename ${IN_FULL_FILE_PREFIX}`
    OUT_FULL_FILE_PREFIX=${OUT_DIR}/${BASE_ID}
    echo "Working on $OUT_FULL_FILE_PREFIX"
    # Export data
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --keep_frames 1 --keep_events 0 --out_file ${OUT_FULL_FILE_PREFIX}_frames.hdf5
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize -5000 --keep_frames 0 --keep_events 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin5k.hdf5
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.010 --keep_frames 0 --keep_events 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin10ms.hdf5
    # Prepare and resize
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_frames.hdf5 --rewrite 1 --skip_mean_std 1
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin5k.hdf5 --rewrite 1 --skip_mean_std 1
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin10ms.hdf5 --rewrite 1 --skip_mean_std 1
done
for filename in ${OUT_DIR}/*_frames.hdf5
do
    frames_h5list="$frames_h5list $filename"
    type_list="$type_list aps_frame_60x80"
done
# Train a network
echo --h5file ${frames_h5list[@]} --type_list ${type_list[@]}
ipython ./multitrain_test_cnn.py -- --h5file ${frames_h5list[@]} --dataset_keys ${type_list[@]} --run_id aps_multi 
