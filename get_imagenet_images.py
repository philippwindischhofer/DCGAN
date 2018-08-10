import urllib
import os

def main():
    urlpath = "imagenet.synset.txt"
    outdir = "imagenet_data/"
    with open(urlpath) as urlfile:
        cnt = 0
        for line in urlfile:
            outpath = os.path.join(outdir, "{}.jpg".format(cnt))
            print("writing {}".format(outpath))
            urllib.urlretrieve(line, outpath)
            cnt += 1

if __name__ == "__main__":
    main()
