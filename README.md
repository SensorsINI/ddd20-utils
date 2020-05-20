# DDD20 End-to-End Event Camera Driving Dataset

See https://sites.google.com/view/davis-driving-dataset-2020/home for details.

Software released as part of the publication

 * Hu, Y., Binas, J., Neil, D., Liu, S.-C., and Delbruck, T. (2020).  "DDD20 End-to-End Event Camera Driving Dataset: Fusing Frames and Events with Deep Learning for Improved Steering Prediction".  Special session Beyond Traditional Sensing for Intelligent Transportation, The 23rd IEEE International Conference on Intelligent Transportation Systems, September 20 – 23, 2020, Rhodes, Greece.   arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2005.08605 

 * Binas, J., Neil, D., Liu, S.-C., and Delbruck, T. (2017). DDD17: **End-To-End DAVIS Driving Dataset**. in _ICML’17 Workshop on Machine Learning for Autonomous Vehicles (MLAV 2017)_, Sydney, Australia.  Available at: arXiv:1711.01458 [cs]  http://arxiv.org/abs/1711.01458 



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

This project was designed to work with Python 2.7

1. First, create an Python 2.7 environment

    ```bash
    conda create -n ddd20 python=2.7
    ```

2. Install all dependencies:

    ```bash
    pip install future
    pip install numpy h5py opencv-python-headless openxc==0.15.0
    ```

3. There is no step 3, have fun! :tada:

# Usage:

See https://sites.google.com/view/davis-driving-dataset-2020/home for details
## viewing

- Play a file from the beginning

    ```bash
    $ python view.py <recorded_file.hdf5>
    ```

- Play a file, starting at X percent

    ```bash
    $ python view.py <recorded_file.hdf5> X%
    ```

- Play a file starting at second X

    ```bash
    $ python view.py <recorded_file.hdf5> Xs
    ```

## Exporting to frame-based representation

```bash
$ python export.py [-h] [--tstart TSTART] [--tstop TSTOP] [--binsize BINSIZE]
                 [--update_prog_every UPDATE_PROG_EVERY]
                 [--export_aps EXPORT_APS] [--export_dvs EXPORT_DVS]
                 [--out_file OUT_FILE]
                 filename
```


# License

This software is released under the GNU LESSER GENERAL PUBLIC LICENSE Version 3.

