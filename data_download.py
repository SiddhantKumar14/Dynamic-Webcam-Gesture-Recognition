import utils
from fastai.vision.all import *

key = os.environ.get('AZURE_SEARCH_KEY','XXX')

search_name = input('Enter the name of the images you want to download')
dest = input('Enter the destination Folder')
path = Path(dest)

results = utils.search_images_bing(key, search_name)
ims = results.attrgot('content_url')

dest.mkdir(exist_ok=True)
results = utils.search_images_bing(key, 'label_name')
download_images(dest, urls=results.attrgot('content_url'))


