# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:18:59 2024

@author: ParkBumCheol
"""


import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

location = "수원시 영통구 영통동"
search_query = location + " 날씨"
search_url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query="
url = search_url + search_query

html_weather = requests.get(url).text
soup_weather = BeautifulSoup(html_weather, "lxml")

# 현재 온도
current_temperature = soup_weather.find("div", {"class": "temperature_text"}).text.replace('현재 온도', '')
# current_temperature = soup_weather.select_one('div.temperature_text strong').get_text().replace('현재 온도', '')

# 현재 날씨
current_weather_condition = soup_weather.find("p", {"class": "summary"}).text
# current_weather_condition = soup_weather.select_one('span.weather before_slash').get_text()

# 오전 강수 확률
morning_rainfall = soup_weather.find("span", class_="rainfall").text
# 오후 강수 확률
afternoon_rainfall = soup_weather.find_all("span", class_="rainfall")[1].text

# 현재 시간
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 현재 체감 온도, 습도, 풍속
current_bodily_sensation = soup_weather.select('div.sort dd')
[sensation, humidity, wind_speed] = [x.get_text() for x in current_bodily_sensation]

print("날씨 정보")
print("현재시간:", current_time)
print("지역:", location)
print("현재 온도:", current_temperature)
print("현재 날씨:", current_weather_condition)
print(f"체감 온도: {sensation},\n현재 습도: {humidity},\n풍속     : {wind_speed}")
print(f"오전 강수 확률: {morning_rainfall}")
print(f"오후 강수 확률: {afternoon_rainfall}")

#%%

location = "수원시 장안구 영화동"
search_query = location + " 날씨"
search_url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query="
url = search_url + search_query

html_weather = requests.get(url).text
soup_weather = BeautifulSoup(html_weather, "lxml")

# 현재 온도
current_temperature = soup_weather.find("div", {"class": "temperature_text"}).text.replace('현재 온도', '')
# current_temperature = soup_weather.select_one('div.temperature_text strong').get_text().replace('현재 온도', '')

# 현재 날씨
current_weather_condition = soup_weather.find("p", {"class": "summary"}).text
# current_weather_condition = soup_weather.select_one('span.weather before_slash').get_text()

# 오전 강수 확률
morning_rainfall = soup_weather.find("span", class_="rainfall").text
# 오후 강수 확률
afternoon_rainfall = soup_weather.find_all("span", class_="rainfall")[1].text

# 현재 시간
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 현재 체감 온도, 습도, 풍속
current_bodily_sensation = soup_weather.select('div.sort dd')
[sensation, humidity, wind_speed] = [x.get_text() for x in current_bodily_sensation]

print("날씨 정보")
print("현재시간:", current_time)
print("지역:", location)
print("현재 온도:", current_temperature)
print("현재 날씨:", current_weather_condition)
print(f"체감 온도: {sensation},\n현재 습도: {humidity},\n풍속     : {wind_speed}")
print(f"오전 강수 확률: {morning_rainfall}")
print(f"오후 강수 확률: {afternoon_rainfall}")

#%%

location = "수원시 권선구 고색동"
search_query = location + " 날씨"
search_url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query="
url = search_url + search_query

html_weather = requests.get(url).text
soup_weather = BeautifulSoup(html_weather, "lxml")

# 현재 온도
current_temperature = soup_weather.find("div", {"class": "temperature_text"}).text.replace('현재 온도', '')
# current_temperature = soup_weather.select_one('div.temperature_text strong').get_text().replace('현재 온도', '')

# 현재 날씨
current_weather_condition = soup_weather.find("p", {"class": "summary"}).text
# current_weather_condition = soup_weather.select_one('span.weather before_slash').get_text()

# 오전 강수 확률
morning_rainfall = soup_weather.find("span", class_="rainfall").text
# 오후 강수 확률
afternoon_rainfall = soup_weather.find_all("span", class_="rainfall")[1].text

# 현재 시간
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 현재 체감 온도, 습도, 풍속
current_bodily_sensation = soup_weather.select('div.sort dd')
[sensation, humidity, wind_speed] = [x.get_text() for x in current_bodily_sensation]

print("날씨 정보")
print("현재시간:", current_time)
print("지역:", location)
print("현재 온도:", current_temperature)
print("현재 날씨:", current_weather_condition)
print(f"체감 온도: {sensation},\n현재 습도: {humidity},\n풍속     : {wind_speed}")
print(f"오전 강수 확률: {morning_rainfall}")
print(f"오후 강수 확률: {afternoon_rainfall}")

#%%

location = "수원시 팔달구 매산동"
search_query = location + " 날씨"
search_url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query="
url = search_url + search_query

html_weather = requests.get(url).text
soup_weather = BeautifulSoup(html_weather, "lxml")

# 현재 온도
current_temperature = soup_weather.find("div", {"class": "temperature_text"}).text.replace('현재 온도', '')
# current_temperature = soup_weather.select_one('div.temperature_text strong').get_text().replace('현재 온도', '')

# 현재 날씨
current_weather_condition = soup_weather.find("p", {"class": "summary"}).text
# current_weather_condition = soup_weather.select_one('span.weather before_slash').get_text()

# 오전 강수 확률
morning_rainfall = soup_weather.find("span", class_="rainfall").text
# 오후 강수 확률
afternoon_rainfall = soup_weather.find_all("span", class_="rainfall")[1].text

# 현재 시간
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 현재 체감 온도, 습도, 풍속
current_bodily_sensation = soup_weather.select('div.sort dd')
[sensation, humidity, wind_speed] = [x.get_text() for x in current_bodily_sensation]

print("날씨 정보")
print("현재시간:", current_time)
print("지역:", location)
print("현재 온도:", current_temperature)
print("현재 날씨:", current_weather_condition)
print(f"체감 온도: {sensation},\n현재 습도: {humidity},\n풍속     : {wind_speed}")
print(f"오전 강수 확률: {morning_rainfall}")
print(f"오후 강수 확률: {afternoon_rainfall}")
