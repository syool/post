# from selenium.webdriver import Firefox
# from selenium.webdriver.firefox.options import Options

# opts = Options()
# opts.headless = True

# driver = Firefox(options=opts)

# try:

#     driver.get('https://www.flickr.com/search/')

#     elem = driver.find_element_by_name('text')
 
#     elem.send_keys('tennis')
#     elem.submit()

# finally:

#     driver.quit()

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options

from urllib.request import urlretrieve
from tqdm import tqdm

import os
import time

def get_images(keyword):
    # browser setting
    opts = Options()
    opts.headless = True

    driver = webdriver.Firefox(options=opts)
    driver.implicitly_wait(30)

    # target url
    url='https://www.flickr.com/search/?text={}'.format(keyword)
    driver.get(url)

    # page auto scrolling
    body = driver.find_element_by_css_selector('body')
    for i in range(3):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)

    # collect image links
    imgs = driver.find_element_by_class_name('view.photo-list-view.awake')
    result = []
    for i in tqdm(imgs):
        if 'http' in img.get_attribute('src'):
            result.append(img.get_attribute('src'))

    driver.close()
    print('crawled successfully')

    # create a download location
    if not os.path.isdir('~/Downloads/{}'.format(keyword)):
        os.mkdir('~/Downloads/{}'.format(keyword))

    # download from urls
    for index, link in tqdm(enumerate(result)):
        start = link.rfind('.')
        filetype = link[start:end]

        urlretrieve(link, '~/Downloads/{}/{}{}{}'.format(keyword, keyword, index, filetype))

    print('downloaded successfully')


get_images('tennis')