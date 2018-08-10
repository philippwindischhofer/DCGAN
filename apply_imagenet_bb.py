# applies the bounding boxes defined in the xml files from imagenet
from __future__ import print_function

import imageio
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

def main():
    expr = "imagenet_data/n07594066/*.JPEG"
    
    filelist = glob.glob(expr)

    for cur_file in filelist:
        print(cur_file)
        cur_metadata = cur_file.replace('.JPEG', '.xml')
        print("checking if metadata file '{}' exists".format(cur_metadata))

        cur_im = imageio.imread(cur_file)
        xmin = 0
        ymin = 0
        xmax = np.shape(cur_im)[0]
        ymax = np.shape(cur_im)[1]
        
        if os.path.exists(cur_metadata):
            tree = ET.parse(cur_metadata)
            root = tree.getroot()
            bndbox = root.find('object').find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)

            print("bounding box found: ({}, {}) -> ({}, {})".format(xmin, ymin, xmax, ymax))

        cur_im = cur_im[xmin:xmax,ymin:ymax,:]

        outpath = cur_file.replace(".JPEG", "_bb.JPEG")
        imageio.imwrite(outpath, cur_im)
        print("wrote {}".format(outpath))

if __name__ == "__main__":
    main()
