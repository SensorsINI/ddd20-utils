#!/bin/bash
set -e
OUT_DIR=/home/dneil/temp/datasets
ORIGIN_DIR=/mnt/ini-nas/DDD17-DavisDrivingDataset2017
#TODO_FILES=( run5/rec1487839456.hdf5 run5/rec1487844247.hdf5 run5/rec1487849151.hdf5 run5/rec1487856408.hdf5 run5/rec1487858093.hdf5 run5/rec1487842276.hdf5 run5/rec1487846842.hdf5 run5/rec1487849663.hdf5 run5/rec1487857941.hdf5 run5/rec1487860613.hdf5 run5/rec1487864316.hdf5 )
TODO_FILES=( run5/rec1487839456.hdf5 run5/rec1487844247.hdf5 )
for TODO_FILE in "${TODO_FILES[@]}"
do
    IN_FULL_FILE_PREFIX=${ORIGIN_DIR}/${TODO_FILE%.*}
    BASE_ID=`basename ${IN_FULL_FILE_PREFIX}`
    OUT_FULL_FILE_PREFIX=${OUT_DIR}/${BASE_ID}
    echo "### Working on $OUT_FULL_FILE_PREFIX ####"

    # Export data
    echo "### Exporting frames... ###"
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --export_aps 1 --export_dvs 0 --out_file ${OUT_FULL_FILE_PREFIX}_frames.hdf5
    echo "### Exporting constant events... ###"
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize -5000 --export_aps 0 --export_dvs 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin5k.hdf5
    echo "### Exporting constant time slices... ###"
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.010 --export_aps 0 --export_dvs 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin10ms.hdf5

    # Prepare and resize
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_frames.hdf5 --rewrite 1 --skip_mean_std 1
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin5k.hdf5 --rewrite 1 --skip_mean_std 1
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin10ms.hdf5 --rewrite 1 --skip_mean_std 1
done

# Find all frame datasets
for filename in ${OUT_DIR}/*_frames.hdf5
do
    frames_h5list="$frames_h5list $filename"
    frames_type_list="$frames_type_list aps_frame_60x80"
    echo "### Found the following APS datasets: ${frames_h5list} ###"
done
# Find all constant event datasets
for filename in ${OUT_DIR}/*_bin5k.hdf5
do
    dvs5k_h5list="$dvs5k_h5list $filename"
    dvs5k_type_list="$dvs5k_type_list dvs_frame_60x80"
    echo "### Found the following constant event datasets: ${dvs5k_h5list} ###"
done
# Find all constant timeslice datasets
for filename in ${OUT_DIR}/*_bin10ms.hdf5
do
    dvs10ms_h5list="$dvs10ms_h5list $filename"
    dvs10ms_type_list="$dvs10ms_type_list dvs_frame_60x80"
    echo "### Found the following constant time datasets: ${dvs5k_h5list} ###"
done

# Train the networks
echo "### Working on: --h5file ${frames_h5list[@]} --type_list ${frames_type_list[@]} ###"
ipython ./multitrain_test_cnn.py -- --h5file ${frames_h5list[@]} --dataset_keys ${frames_type_list[@]} --run_id aps_multi
ipython ./multitrain_test_rnn.py -- --h5file ${frames_h5list[@]} --dataset_keys ${frames_type_list[@]} --run_id aps_multi
echo "### Working on: --h5file ${dvs5k_h5list[@]} --type_list ${dvs5k_type_list[@]} ###"
ipython ./multitrain_test_cnn.py -- --h5file ${dvs5k_h5list[@]} --dataset_keys ${dvs5k_type_list[@]} --run_id dvs5k_multi
ipython ./multitrain_test_rnn.py -- --h5file ${dvs5k_h5list[@]} --dataset_keys ${dvs5k_type_list[@]} --run_id dvs5k_multi
echo "### Working on: --h5file ${dvs10ms_h5list[@]} --type_list ${dvs10ms_type_list[@]} ###"
ipython ./multitrain_test_cnn.py -- --h5file ${dvs10ms_h5list[@]} --dataset_keys ${dvs10ms_type_list[@]} --run_id dvs10ms_multi
ipython ./multitrain_test_rnn.py -- --h5file ${dvs10ms_h5list[@]} --dataset_keys ${dvs10ms_type_list[@]} --run_id dvs10ms_multi
echo "### Working on: --h5file ${dvs10ms_h5list[@]} ${dvs5k_h5list[@]} ${frames_h5list[@]} --type_list ${dvs10ms_type_list[@]} ${dvs5k_type_list[@]} ${frames_type_list[@]} ###"
ipython ./multitrain_test_cnn.py -- --h5file ${dvs10ms_h5list[@]} ${dvs5k_h5list[@]} ${frames_h5list[@]} --dataset_keys ${dvs10ms_type_list[@]} ${dvs5k_type_list[@]} ${frames_type_list[@]} --run_id everything_multi
ipython ./multitrain_test_rnn.py -- --h5file ${dvs10ms_h5list[@]} ${dvs5k_h5list[@]} ${frames_h5list[@]} --dataset_keys ${dvs10ms_type_list[@]} ${dvs5k_type_list[@]} ${frames_type_list[@]} --run_id everything_multi
