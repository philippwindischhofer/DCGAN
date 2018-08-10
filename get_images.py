import sys
sys.path.append('externals/google-images-download/google_images_download/')

from google_images_download import googleimagesdownload

def main():
    dloader = googleimagesdownload()
    dloader.download({"keywords": "herrentorte", "limit": 70000, "chromedriver": "/usr/bin/chromedriver", "no_numbering": True, "output_directory": "data_raw"})
    print("test")

if __name__ == "__main__":
    main()
