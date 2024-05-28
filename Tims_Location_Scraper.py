import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import numpy as np

url = "https://locations.timhortons.ca/en/"

response = requests.get(url)
soup = BeautifulSoup(response.content, "lxml")

ul_element = soup.find("ul", class_="sb-directory-list sb-directory-list-states")
li_elements = ul_element.find_all("li")

provs = {'Provience' : [], 'hrefs_prov' : []}
for li in li_elements:
  link = li.find("a")
  href = link.get("href")
  name = link.text
  provs['Provience'].append(name)
  provs['hrefs_prov'].append(urljoin(url,href))

Provience_df = pd.DataFrame(provs)

cities = {'Provience' : [], 'cities' : [], 'hrefs_city' : []}
for Provience, hrefs_prov in zip(provs['Provience'],provs['hrefs_prov']):
  response = requests.get(hrefs_prov)
  soup = BeautifulSoup(response.content, "lxml")

  ul_element = soup.find("ul", class_="sb-directory-list sb-directory-list-states")
  li_elements = ul_element.find_all("li")

  for li in li_elements:
    link = li.find("a")
    href = link.get("href")
    name = link.text
    cities['Provience'].append(Provience)
    cities['cities'].append(name)
    cities['hrefs_city'].append(urljoin(url,href))

cities_df = pd.DataFrame(cities)

sites = {'Provience': [], 'cities' : [], 'address': [], 'hrefs_site' : [],
         'Has WiFi': [], 'Catering Available': [], 'Dine In': [], 'Take Out': [],
         'Dine-In Sunday': [], 'Dine-In Monday': [], 'Dine-In Tuesday': [], 'Dine-In Wednesday': [], 'Dine-In Thursday': [], 'Dine-In Friday': [], 'Dine-In Saturday': [],
         'Drive-Thru Sunday': [], 'Drive-Thru Monday':[], 'Drive-Thru Tuesday': [], 'Drive-Thru Wednesday': [],
         'Drive-Thru Thursday': [], 'Drive-Thru Friday': [], 'Drive-Thru Saturday': []
         }

for Provience, city, hrefs_city in zip(cities['Provience'], cities['cities'],cities['hrefs_city']):
  try:
    response = requests.get(hrefs_city)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")

  except Exception as e:
    print(f"An error occurred: {e}")
    sites['Provience'].append(Provience)
    sites['cities'].append(city)
    sites['address'].append(np.nan)
    sites['hrefs_site'].append(np.nan)
    continue


  ul_element = soup.find("ul", class_="sb-directory-list sb-directory-list-sites")
  li_elements = ul_element.find_all("li")

  for li in li_elements:
    link = li.find("a")
    href = link.get("href")
    sites['Provience'].append(Provience)
    sites['cities'].append(city)
    sites['hrefs_site'].append(urljoin(url,href))

    address = li.text.replace("Tim Hortons\n -","").replace('\n','')
    sites['address'].append(address)

    features_list = ['Has WiFi', 'Catering Available', 'Dine In', 'Take Out']

    hrefs_site = urljoin(url,href)
    response_site = requests.get(hrefs_site)
    response_site.raise_for_status()
    soup_site = BeautifulSoup(  response_site.content, "lxml")
    features_div = soup_site.find("div", class_="lp-banner-features")

    features_avail = []
    for feature in features_div.find_all("li"):
      features_avail.append(feature.text.strip())

    for feature in features_list:
      if feature in features_avail:
        sites[feature].append(True)
      else:
        sites[feature].append(False)

    dine_in_div = soup_site.find('div', class_='lp-label', string='Dine-In Hours')
    if dine_in_div:
      dine_in_container = dine_in_div.find_next_sibling('div', class_='lp-hours')
      days = dine_in_container.find_all(class_='lp-day')
      hours = dine_in_container.find_all(class_='lp-hours')
      for day, hour in zip(days, hours):
        sites['Dine-In'+' '+day.get_text(strip=True).replace(':','')].append(hour.get_text(strip=True))

    dine_in_div = soup_site.find('div', class_='lp-label', string='Drive-Thru Hours')
    if dine_in_div:
      dine_in_container = dine_in_div.find_next_sibling('div', class_='lp-hours')
      days = dine_in_container.find_all(class_='lp-day')
      hours = dine_in_container.find_all(class_='lp-hours')
      for day, hour in zip(days, hours):
        sites['Drive-Thru'+' '+day.get_text(strip=True).replace(':','')].append(hour.get_text(strip=True))

sites_df = pd.DataFrame(sites)
sites_df = pd.merge(cities_df, sites_df, on = ['Provience','cities'])

final_df = pd.merge(Provience_df, sites_df, on='Provience', how='left')
final_df.to_csv('./Tim_Hortons_Locations.csv', index=False)