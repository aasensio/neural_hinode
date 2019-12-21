# Neural 3D LTE inversor

## Run the code

The code needs as input an HDF5 file with the Stokes profiles downloaded from the Hinode database and will produce as output another HDF5 file with the resulting physical conditions.
The code is run as:

    python neural_inversion.py --input input.h5 --output output.h5 --normalize 0 100 0 100

The arguments are the input and output files, together with the definition of a box which is used as the quiet Sun for normalizing the Stokes profiles.

## Download Hinode data

Search for the initial date and time of the Hinode observation you want by
searching in the Hinode database ``http://sdc.uio.no/search/form``. Once found, search on the Hinode level 1 database that the observation you want is present, for instance: 
``http://www.lmsal.com/solarsoft/hinode/level1hao/2014/02/04/SP3D/20140204_190005`. Once located, you can call the ``download_hinode.py``code that will download all the FITS files and generate an HDF5 file appropriate for the input to the neural inversor:
    
    python download_hinode.py --url http://www.lmsal.com/solarsoft/hinode/level1hao/2014/02/04/SP3D/20140204_190005 --output output.h5 --downloader curl

By default, the files are downloaded with ``wget``. If you do not have it available in your system, which happens in many cases in MacOS, you can use ``curl``. You can find some help of the options of the downloader by using ``python download_hinode -h``.

## Requirements
    bs4
    numpy
    argparse
    h5py
    tqdm
    astropy
    torch

## Use with Anaconda

We recommend to use Anaconda to run this code. We also recommend to generate a new environment in which all the packages will be installed. Then, install ``PyTorch``as indicated in the webpage ``https://pytorch.org/`` depending on your system . A typical process would be:

    conda create -n inversor python=3.7
    conda activate inversor
    conda install -c conda-forge numpy h5py tqdm astropy bs4 argparse
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch