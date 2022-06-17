#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python

import argparse
import asyncio
import logging
from os.path import exists

import pyppeteer.connection
import requests
from bs4 import BeautifulSoup

URL = "https://www.ardmediathek.de/sendung/tagesschau-in-100-sekunden/Y3JpZDovL2Rhc2Vyc3RlLmRlL3RzMTAwcw"
PATH = "/Users/tihmels/TS/ts100/"


def process_page(page):
    soup = BeautifulSoup(page, "html.parser")

    video = soup.find('video').find('source')
    url = video['src']
    name = url.split('/')[-1]

    r = requests.get(url)

    file = '{}/{}'.format(PATH, name)

    if not exists(file):
        with open(file, 'wb') as fd:
            fd.write(r.content)


async def main():
    browser = await pyppeteer.connect(
        browserURL='http://localhost:9222', slowMo=3, logLevel=logging.INFO)

    pages = await browser.pages()
    page = pages[-1]

    await page.setViewport({'height': 900, 'width': 1200})

    await page.goto(URL, {'waitUntil': 'networkidle0'})

    links = {}

    for i in range(12):
        await page.evaluate("""{window.scrollBy(0, 800);}""")
        await page.waitFor(1000)

        soup = BeautifulSoup(await page.content(), "html.parser")
        divs = soup.select('div[data-bb-id]')
        for div in divs:
            key = div['data-bb-id']
            links[key] = div

        print(len(links.keys()))

    links = [div.findChild("a")['href'] for div in links.values()]

    print("Downloading {} videos".format(len(links)))

    for link in links:
        url = "https://www.ardmediathek.de" + link
        await page.goto(url, {'waitUntil': 'networkidle0'})

        await page.waitForSelector("video.ardplayer-mediacanvas")
        await page.waitForSelector("button.ardplayer-button-settings")
        await page.waitForSelector("source")

        process_page(await page.content())

    await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(main())
