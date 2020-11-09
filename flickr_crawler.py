import flickrapi
from urllib.request import urlretrieve
from PIL import Image

import os

# Flickr api access key 
flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)

keyword = 'tennis'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_c',
                     per_page=100,           # may be you can try different numbers..
                     sort='relevance')

urls = []
for i, photo in enumerate(photos):
    print (i)
    
    url = photo.get('url_c')
    urls.append(url)
    
    # get n urls
    n = 200
    if i > n:
        break

print (urls)

# Download image from the url and save it to '00001.jpg'
for i in range(n):
    urlretrieve(urls[n], '0000{}.jpg'.format(n))

# Resize the image and overwrite it
image = Image.open('00001.jpg') 
image = image.resize((256, 256), Image.ANTIALIAS)
image.save('00001.jpg')