import requests
from bs4 import BeautifulSoup
import pandas as pd

#URL page for amazon product reviews, starting from page 2
url = "https://www.amazon.co.uk/product-reviews/0241425441/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2"


#Taken from curlthrillwork to bypass amazon
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.5',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-GPC': '1',
    'TE': 'Trailers',
     }

#Function for soup so I can scrape multiple pages
def get_soup(url):
    page = requests.get(url, headers=headers)
    #Soup parser
    soup = BeautifulSoup(page.text, "html.parser")
    return soup

#Function for Assigning the review section of the html page
def get_reviews(soup):
    #Reviews
    review_list = []
    reviews = soup.find_all("div", {"data-hook": "review"})
    #Looping through and splitting each section of the review and adding to a dict
    for item in reviews:
        review = {
        "product": soup.title.text.replace("Amazon.co.uk:Customer reviews:", "").strip(),
        "title": item.find("a", {"data-hook": "review-title"}).text.strip(),
        "rating": float(item.find("i", {"data-hook": "review-star-rating"}).text.replace("out of 5 stars","").strip()),    
        "text_review": item.find("span", {"data-hook": "review-body"}).text.strip(),
        }
        review_list.append(review)
    return review_list

#Function with page scraper, for loop to go through page numbers
#WIll scrape the first 2500 reviews
def multiple_page_scraper(url):
    page_counter = 2
    for page_counter in range(1000):
        new_url = url.replace("next_2", "next_{}".format(page_counter)).replace("pageNumber=2","pageNumber={}".format(page_counter))
        soup = get_soup(new_url)
        get_reviews(soup)
        page_counter += 1
        
        if soup.find("li", {"class": "a-disabled a-last"}):
            break