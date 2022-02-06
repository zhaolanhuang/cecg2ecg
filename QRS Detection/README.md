The codes in this folders were mainly adapted from https://www.medit.hia.rwth-aachen.de/fileadmin/user_upload/05_Publication/UnoVis/CNN_Fusion.zip

C. Hoog Antink, E. Breuer, D. U. Uguz, and S. Leonhardt: “Signal-Level Fusion with Convolutional Neural Networks for Capacitively Coupled ECG in the Car”. Computing in Cardiology 2018;45: accepted for publication

The required UnoViS_auto2012.mat can be downloaded at: https://www.medit.hia.rwth-aachen.de/unovis

usage: run_me.py [-h] [-e [EPOCHS [EPOCHS ...]]] [-t [THRESH [THRESH ...]]]
                 [-c CACHE] [--reprint_only] [-d]
                 mat_file

e.g. python run_me.py ./UnoViS_auto2012.mat -e 2 4 -t 0.6 0.95

positional arguments:
  mat_file              Path to the UnoViS_auto2012.mat file. (Can be
                        downloaded at: 'link einfuegen!')

optional arguments:
  -h, --help            show this help message and exit
  -e [EPOCHS [EPOCHS ...]], --epochs [EPOCHS [EPOCHS ...]]
                        Defines the number of epochs per generation. Should be
                        a list of integers. e.g. 1 2 4 8
  -t [THRESH [THRESH ...]], --thresh [THRESH [THRESH ...]]
                        Define the thresholds for the final peak detection.
                        Should be a list of floats, e.g. 0.6 0.7 0.8 0.9
  -c CACHE, --cache CACHE
                        Path to a directory to hold temporary files.
  --reprint_only        Does not train and evaluate a network but print
                        results of an earlier run stored in the cache
                        directory.
  -d, --debug           Print additional debug information to the console.
