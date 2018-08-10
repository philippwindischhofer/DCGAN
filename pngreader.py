import glob
import imageio
import numpy as np

class pngreader:

    @staticmethod
    def load_files(expr):
        filelist = glob.glob(expr)
        number_files = len(filelist)
        
        print("found {} files matching '{}'".format(number_files, expr))

        print("loading images")
        images = []        
        for index, cur_file in enumerate(filelist):
            cur_im = imageio.imread(cur_file)

            # if this is a grayscale image, convert to a quasi-RGB one
            if cur_im.ndim == 2:
                cur_im = np.array([cur_im, cur_im, cur_im])
                cur_im = np.moveaxis(cur_im, 0, 2)
            
            #print("loading images: {:3.0f}% done".format(100 * index / number_files), end = '\r')

            if np.shape(cur_im) != (64,64,3):
                print("{}: {} -> is not used".format(cur_file, np.shape(cur_im)))
            else:
                images.append(cur_im)

        #print("loading images: 100% done")
        print("done")

        imagearr = np.array(images)

        imagearr = imagearr / 127.5 - 1.0
        
        return imagearr
