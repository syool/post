import flickrapi
from urllib.request import urlretrieve
from PIL import Image

import os
from tqdm import tqdm

# Flickr api access key 
flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)

keyword = 'soccer kick'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     extras='url_c',
                     per_page=100, # may be you can try different numbers..
                     sort='relevance')

urls = []
for i, photo in enumerate(photos):
    url = photo.get('url_c')
    urls.append(url)
    
    # get n urls
    n = 1000
    if i > n:
        break

print (urls)
print ('{} urls are fetched'.format(len(urls)))

if not os.path.isdir('./{}'.format(keyword)):
    os.mkdir('./{}'.format(keyword))

# Download image from the url and save it as 'n.jpg'
for i in tqdm(range(n)):
    if urls[i] == None:
        continue
    else:
        try:
            urlretrieve(urls[i], './{}/{}.jpg'.format(keyword, i))

            # Resize the image and overwrite it
            image = Image.open('./{}/{}.jpg'.format(keyword, i)) 
            image = image.resize((256, 256), Image.ANTIALIAS)
            image.save('./{}/{}.jpg'.format(keyword, i))

            # print('{}.jpg saved'.format(i))
        
        except HTTPError:
            print('Error ignored')