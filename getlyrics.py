import socket
import json
import requests
import re
import sys
from bs4 import BeautifulSoup

CLIENT_TOKEN = "FAlLLpDDvipZuMDOBM07qRkDgw_kQs3_l3KYJuMsXCE3EJIE77MdWC8FP2x2ieCO"
URL = "http://api.genius.com"
headers = {"Authorization": "Bearer " + CLIENT_TOKEN, "User-Agent": "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)"}

def getRequest(url, headers):
    return requests.request(url=url, headers=headers, method="GET")

def writeLyrics(url, headers, filename):
    req = getRequest(url, headers)
    html = BeautifulSoup(req.text, "html.parser")
    [h.extract() for h in html('script')]
    lyrics = html.find("div", class_="lyrics").get_text()
    lyrics = lyrics.replace("\n", " ")
    lyrics = re.sub(r'\[.+?\]', "",lyrics)
    songFile = open(filename, "w")
    songFile.write(lyrics)
    songFile.close()


def main():
    args = sys.argv[1:]
    if len(args) == 2:
        fn = args[0]
        url = args[1]
        writeLyrics(url, headers, fn)

if __name__ == "__main__":
    main()

    # page = 1
    # while page < 5:
    #     url = "http://api.genius.com/search?q=" + urllib2.quote(query) + "&page=" + str(page)
    #     req = urllib2.Request(url)
    #     req.add_header("Authorization", "Bearer " + CLIENT_TOKEN)
    #     req.add_header("User-Agent", "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)")
    #
    #
    #
    # search_url = URL + "/songs/" + "70324"
    # req = requests.request(url=search_url, headers=headers, method="GET")
    #
    # res = req.text
    #
    # print(res)
    # json_obj = json.loads(raw)
    # body = json_obj['response']['song']
    # lyrics_path = body['path']
    # lyrics_url = "https://genius.com/Kendrick-lamar-poetic-justice-lyrics"

    # print(raw)
    # json_obj = json.loads(raw)
    # body = json_obj
    # print(body)
