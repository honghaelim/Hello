# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:14:14 2024

@author: soyoung
"""
# 현재 시간,지역,온도.날씨,습도, 강수 확률

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime


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

# 현재 강수량
rainfall_element = soup_weather.find_all('div', class_='data_inner')[0].text
print("현재강수량:","시간당",rainfall_element,"mm")


rainfall_element                      #Out[43]: ' 0 '


#%%
#날씨분리
weather=current_weather_condition.split()[-1]

print("네이버 날씨:",weather)



def weather_result(weather):
  
    
    if weather in ['맑음','구름조금','가끔비','가끔 비,눈','가끔 눈','흐린 후 갬','뇌우 후 갬','비 후 갬','눈 후 갬']:
        result='맑음'
    elif weather in ['구름많음','흐림','안개']:
        result='흐림'
    elif weather in ['약한비','비','강한비','소나기','흐려져 비']:
        result='비'
    elif weather in ['약한눈','눈','강한눈','진눈깨비','소낙눈','우박','흐려져 눈']:
        result='눈'
    else:
        result='기타'
        
    return result

weather_result(weather)

"""
네이버 날씨: 맑음
기상상태: 맑음 """
#날씨분리
weather=current_weather_condition.split()[-1]
print("네이버 날씨:",weather)

def weather_result(weather):
    
    if weather in ['맑음','구름조금','가끔비','가끔 비,눈','가끔 눈','흐린 후 갬','뇌우 후 갬','비 후 갬','눈 후 갬']:
        result='맑음'
    elif weather in ['구름많음','흐림','안개']:
        result='흐림'
    elif weather in ['약한비','비','강한비','소나기','흐려져 비']:
        result='비'
    elif weather in ['약한눈','눈','강한눈','진눈깨비','소낙눈','우박','흐려져 눈']:
        result='눈'
    else:
        result='기타'
        
    return result

weather_result(weather)

"""
네이버 날씨: 맑음
기상상태: 맑음 """

#%%

#str ->datetime 형변환
current_time=datetime.strptime(current_time,"%Y-%m-%d %H:%M:%S")
#시간(오전:0~12, 오후:12~24)
hour=current_time.hour


#%%

"""
온도 current_temperature
기상상태 weather_result(weather)
강수확률 am_pm_decision(hour)
시간 hour
강수량 rainfall_element
습도 humidity
"""



#%%
import re

#시간 기준으로 오전강수확률,오후강수확률 사용할지 결정 +(%)없애고 정수형으로 변환

def am_pm_decision(hour):
    if hour>=0 and hour<12:
        rainfall_percent=int(re.sub(r'[^\d.]', '',afternoon_rainfall))
    else:
        rainfall_percent=int(re.sub(r'[^\d.]', '',morning_rainfall))
    return rainfall_percent

print("현재 강수확률(%):",am_pm_decision(hour)  )        #강수확률


#강수량 양쪽 공백 제거후 정수형으로 변환
rainfall=int(rainfall_element.strip())

#현재온도 양쪽 공백, '°' 제거후 실수형으로
current_temperature= float(re.sub(r'[^\d.]', '', current_temperature))

#습도 (%)제거 후 정수형으로 변환
humidity=int(re.sub(r'[^\d.]', '',humidity))

#%%

#노면상태


#비
if weather_result(weather)=='비':
    road_surface='젖음/습기'
#눈
elif weather_result(weather)=='눈':
    if rainfall>50 and current_temperature<0 :
        road_surface='적설'
    elif 1<rainfall<=50 and current_temperature<0:
        road_surface='서리/결빙'
    else:
        road_surface='젖음/습기'
#맑음       
elif weather_result(weather)=='맑음':
    if rainfall>50 and current_temperature<0 :
        road_surface='적설'
    elif 1<=rainfall<=50 and current_temperature<0:
        road_surface='서리/결빙' 
    elif rainfall>=1 and current_temperature>=0:
        road_surface='젖음/습기'
    elif rainfall==0 and current_temperature>0:
        road_surface='건조'
    else:
        road_surface='기타'
#흐림       
elif weather_result(weather)=='흐림':
    if rainfall>=50 and current_temperature<0 and 80<humidity<=100:
        road_surface='적설'
    elif 0<rainfall<50 and current_temperature>=0 and 80<humidity==100:
        road_surface=='젖음/습기'
    elif rainfall==0 and 80<humidity==100:
        road_surface='건조'
#기타       
else:
    road_surface='기타'
    
    
print("노면상태:",road_surface)















