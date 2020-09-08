from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
import os
import pathlib



def get_page(url):
    user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.101 Safari/537.36'
    user_agent2 = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'
    req = Request(url, headers={'User-Agent': user_agent2})

    web_byte = urlopen(req)
    page = web_byte.read()
    web_byte.close()
    #webpage = page.decode('utf-8')
    return page

def download_music():
    page = soup(get_page(),"html.parser")

print("hello "*10000)