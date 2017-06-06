
Software released as part of the publication

# "DDD17: End-To-End DAVIS Driving Dataset"
Jonathan Binas, Daniel Niel, Shih-Chii Liu, Tobi Delbruck
Institute of Neuroinformatics, University of Zurich and ETH Zurich, Switzerland

See https://docs.google.com/document/d/1HM0CSmjO8nOpUeTvmPjopcBcVCk7KXvLUuiZFS6TWSg/pub for details.

Note: the software has been tested with python 2.7, support for newer versions will follow.


# Prerequisites

These tools require
 * openCV (pip install opencv-python),
 * h5py (pip install h5py).


# Usage:

## viewing

### Play a file from the beginning
$ python view.py <recorded_file.hdf5>

### Play a file, starting at X percent
$ python view.py <recorded_file.hdf5> X%

### Play a file starting at second X
$ python view.py <recorded_file.hdf5> Xs


## Exporting to frame-based representation

$ python export.py [-h] [--tstart TSTART] [--tstop TSTOP] [--binsize BINSIZE]
                 [--update_prog_every UPDATE_PROG_EVERY]
                 [--keep_frames KEEP_FRAMES] [--keep_events KEEP_EVENTS]
                 [--out_file OUT_FILE]
                 filename


# License

This software is released under the GNU LESSER GENERAL PUBLIC LICENSE Version 3.

