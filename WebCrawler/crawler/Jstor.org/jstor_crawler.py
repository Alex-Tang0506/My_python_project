import json
import os
from Main_Crawler import MainCrawler
import csv
import pandas as pd
import time

request_headers = {
    # "Connection": "close",
    "content-type": "application/json"
}


payloads = '{ \
        "acc": "",\
        "allowEmptyQuery": false,\
        "endDate": "",\
        "filter": "",\
        "forwardedAdvancedSearchParams": {\
            "Query": "+George+Eliot",\
            "so": "rel"\
        },\
        "getFlagName": "enable_search_results_infinite_scrolling",\
        "isAdvancedSearch": false,\
        "msFacetFields": [],\
        "pageParams": {},\
        "referer": "",\
        "refreqid": "login:7d8404d7bfcc800cdbe619d9ebdaa699",\
        "searchTerm": "George Eliot",\
        "sortOrder": "rel",\
        "startDate": "",\
        "filterQueries": []\
    }'

maincrawler = MainCrawler()
url_grouped_search = 'https://www.jstor.org/search-results/grouped-search/'

if __name__ == "__main__":
    print('===  ' + 'Scraping data from: ' + url_grouped_search + '  ===')
    time.sleep(1)

    current_path = os.getcwd()
    # f = open(current_path + "\\narrow_search.csv", mode="w", encoding='utf-8-sig', newline='')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(['Title_URL'])
    resp = maincrawler.post_source(url_grouped_search, request_headers, payloads)
    print(resp)
    print(resp.json())
    # content_dict = pd.json_normalize(resp.json())
    # print(content_dict)
    # print(type(content_dict))



