# DDD20 End-to-End Event Camera Driving Dataset

See https://sites.google.com/view/davis-driving-dataset-2020/home for details.

![view.py](ddd20-view.gif)

Software released as part of the publication

 * Hu, Y., Binas, J., Neil, D., Liu, S.-C., and Delbruck, T. (2020).  "DDD20 End-to-End Event Camera Driving Dataset: Fusing Frames and Events with Deep Learning for Improved Steering Prediction".  Special session Beyond Traditional Sensing for Intelligent Transportation, The 23rd IEEE International Conference on Intelligent Transportation Systems, September 20 – 23, 2020, Rhodes, Greece.   arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2005.08605 

 * Binas, J., Neil, D., Liu, S.-C., and Delbruck, T. (2017). DDD17: **End-To-End DAVIS Driving Dataset**. in _ICML’17 Workshop on Machine Learning for Autonomous Vehicles (MLAV 2017)_, Sydney, Australia.  Available at: arXiv:1711.01458 [cs]  http://arxiv.org/abs/1711.01458 

See https://github.com/SensorsINI/ddd20-itsc20 for the code used for the Hu paper above.

<!-- # Prerequisites -->

<!-- Note: Tested with python 3.7. -->
<!--  -->
<!-- If using conda, install pip to your conda environment first. -->
<!--  -->
<!-- These tools require -->
<!--  * openCV (pip install opencv-python), -->
<!--  * openxc (pip install openxc) -->
<!--  * h5py (pip install h5py). -->
<!--  -->
<!-- Or (inside your python 3.7 environment) -->
<!-- ```bash -->
<!-- pip install openxc opencv-python h5py -->
<!-- ``` -->

# Installation instructions using conda and Python 2.7

This project currently works with Python 2.7 under linux.  Lasted tested working *view.py* November 2024.

*(Don't try to run this code in python 3; it depends on some cryptic multiprocessing code that is not portable to python3! Trust us, we tried to port it.)*

1. First, create an Python 2.7 environment

    ```bash
    conda create -n ddd20 python=2.7
    conda activate ddd20
    ```

2. Install all dependencies:

    ```bash
    pip install future
    pip install numpy h5py opencv-python==4.2.0.32 openxc==0.15.0
    ```

3. There is no step 3, have fun! :tada:

# Usage:

See https://sites.google.com/view/davis-driving-dataset-2020/home for details
## viewing

- Play a file from the beginning

    ```bash
    $ python view.py <recorded_file.hdf5>
    ```

- Print usage
    ```bash
    $ python view.py --help
    usage: view.py [-h] [--start START] [--rotate ROTATE] filename

    positional arguments:
    filename

    optional arguments:
    -h, --help            show this help message and exit
    --start START, -s START
                            Examples:
                            -s 50% - play file starting at 50%
                            -s 66s - play file starting at 66s
    --rotate ROTATE, -r ROTATE
                            Rotate the scene 180 degrees if True, Otherwise False
    ```

While viewing, hit ? or h for help in console:
```
space pause
b brighter
d darker
s slower
f faster
i toggle/rotate info
r rotate 180 deg
```

## Exporting raw data into standard data types

The DDD20 recordings are recorded using a custom data structure in HDF5.
This design choice made the batch processing restricted without reformatting/exporting.

We prepared a script that can convert the original HDF5 recording into a
nicer data strcture that user can directly work on. __However, this file will not contain the car CAN bus steering/throttle/GPS, etc.__

```bash
$ python export_ddd20_hdf.py [-h] [--rotate ROTATE] filename
```

The newly exported file is an HDF5 file that is called `filename.exported.hdf5`.
This file is saved at the same folder of the `filename`.
This HDF5 file has a very simple structure, it has three datasets:

```
event: (N events x 4)  # each row is an event.
frame: (M frames x 260 x 346)
frame_ts: (M frames x 1)
```
Added now is option to turn off the display (thanks youkaichao) so that issue #4 can be resolved by simply adding the option: 
```
python export_ddd20_hdf.py filename --display 0
```
## Exporting to frame-based representation

```bash
$ python export.py [-h] [--tstart TSTART] [--tstop TSTOP] [--binsize BINSIZE]
                 [--update_prog_every UPDATE_PROG_EVERY]
                 [--export_aps EXPORT_APS] [--export_dvs EXPORT_DVS] [--display 0]
                 [--out_file OUT_FILE]
                 filename
```


# License

This software is released under the GNU LESSER GENERAL PUBLIC LICENSE Version 3.

