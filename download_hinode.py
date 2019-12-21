from bs4 import BeautifulSoup
import requests
import numpy as np
import os
import argparse
import glob
import h5py
from tqdm import tqdm
from astropy.io import fits

def listFD(url, ext=''):
    page = requests.get(url).text    
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def download(url, output, downloader):
    ext = 'fits'
    
    # print("Finding files to download...")

    # fout = open('file_list', 'w')
    # for file in listFD(url, ext):
    #     fout.write('{0}\n'.format(file))

    # fout.close()

    print("Downloading files...")
    if (downloader == 'wget'):
        os.system("cat file_list | xargs -n 1 -P 8 wget -q")
    if (downloader == 'curl'):
        os.system("cat file_list | xargs -n 1 -P 8 curl -O")

    print("Generating HDF5 file...")
    files = glob.glob('*.fits')
    files.sort()
    nfiles = len(files)

    f = h5py.File(output, 'w')
    db = f.create_dataset('stokes', shape=(4,nfiles,408,112))

    for i in tqdm(range(nfiles)):
        ff = fits.open(files[i])
        img = ff[0].data.astype('float32')
        stokesi = img[0,:,:]
        ind = np.where(stokesi < 0)
        stokesi[ind] += 65536.0
        img[0,:,:] = stokesi
        db[:,i,:,:] = img 
        ff.close()

    f.close()

    print("Removing all FITS files")
    os.system("rm -f *.fits")

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Download Hinode from http://www.lmsal.com/solarsoft/hinode/level1hao')
    parser.add_argument('--url', default=None, type=str,
                    metavar='URL', help='Hinode URL - Example: http://www.lmsal.com/solarsoft/hinode/level1hao/2014/02/04/SP3D/20140204_190005', required=True)
    parser.add_argument('--output', default=None, type=str,
                    metavar='OUTPUT', help='Output HDF5 file', required=True)
    parser.add_argument('--downloader', default='wget', type=str,
                    metavar='DOWNLOADER', help='Downloader (wget/curl)', required=False)

    parsed = vars(parser.parse_args())

    download(parsed['url'], parsed['output'], parsed['downloader'])
