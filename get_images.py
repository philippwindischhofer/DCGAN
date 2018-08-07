import sys
sys.path.append('externals/google-images-download/google_images_download/')

from google_images_download import googleimagesdownload

def main():
    dloader = googleimagesdownload()
    dloader.download({"keywords": "cake", "limit": 200, "chromedriver": "/usr/bin/chromedriver", "no_numbering": True})
    print("test")

if __name__ == "__main__":
    main()
