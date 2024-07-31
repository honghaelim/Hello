# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:39:17 2024

@author: ksj
"""

import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

""" 주간/야간 구분 함수
> 주간: 오전 7시부터 오후 8시까지 (13시간)
> 야간: 오후 8시부터 다음 날 오전 7시까지 (11시간) """
def timesplit(current_time):
    if current_time.hour>=7 and current_time.hour<20:
        current="주간"
    else:
        current="야간"
    return current

""" 네이버 날씨(기상청)
- (입력)지역 -> (출력)날씨,온도,습도,강수량

"""
def weather_info(gu) :
    location = "수원시" + gu
    search_query = location + " 날씨"
    search_url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query="
    url = search_url + search_query
    
    html_weather = requests.get(url).text
    soup_weather = BeautifulSoup(html_weather, "lxml")
    
    # 현재 날씨
    current_weather_condition = soup_weather.find("p", {"class": "summary"}).text
    #   : 네이버 날씨(현재 날씨) -> '날씨' 분리
    current_weather=current_weather_condition.split()[-1]
    
    # 현재 온도
    current_temperature = soup_weather.find("div", {"class": "temperature_text"}).text.replace('현재 온도', '')
    #      : 양쪽 공백, '°' 제거후 실수형으로
    current_temperature= float(re.sub(r'[^\d.]', '', current_temperature))
    
    # 현재 체감 온도, 습도, 풍속
    current_bodily_sensation = soup_weather.select('div.sort dd')
    [sensation, humidity, wind_speed] = [x.get_text() for x in current_bodily_sensation]
    #    습도 : (%)제거 후 정수형으로 변환
    humidity=int(re.sub(r'[^\d.]', '',humidity))
    
    # 현재 강수량
    rainfall_element = soup_weather.find_all('div', class_='data_inner')[0].text
    #     : 양쪽 공백 제거후 정수형으로 변환
    rainfall=int(rainfall_element.strip())
    
    return current_weather, current_temperature, humidity, rainfall

# 현재 날씨에 따른 기상상태 가져오기
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

# 기상상태, 현재온도, 강수량, 습도에 따른 노면상태 가져오기 
def road_result(weather, current_temperature, rainfall, humidity) :
    #비
    if weather == '비':
        result='젖음/습기'
    #눈
    elif weather =='눈':
        if rainfall>50 and current_temperature<0 :
            result='적설'
        elif 1<rainfall<=50 and current_temperature<0:
            result='서리/결빙'
        else:
            result='젖음/습기'
    #맑음       
    elif weather=='맑음':
        if rainfall>50 and current_temperature<0 :
            result='적설'
        elif 1<=rainfall<=50 and current_temperature<0:
            result='서리/결빙' 
        elif rainfall>=1 and current_temperature>=0:
            result='젖음/습기'
        elif rainfall==0 and current_temperature>0:
            result='건조'
        else:
            result='기타'
    #흐림       
    elif weather=='흐림':
        if rainfall>=50 and current_temperature<0 :
            result='적설'
        elif 1<rainfall<50 :
            result=='젖음/습기'
        else:
            result='건조'          
    #기타       
    else:
        result='기타'
        
    return result

#%%
###############################################################################
# 조건에 따른 위험지수(eclo_risk_mul) 추출
#   - 위험지수(eclo_risk_mul) = 요인별 (risk(순위 역순)*ECLO)의 합계
###############################################################################

# ## '구' 목록
gu_list = ['권선구', '장안구', '팔달구', '영통구']

for gu in gu_list :
    # ## 주야간
    current_time = timesplit(datetime.now())
    
    # 기상청 데이터
    weather_list = weather_info(gu)
    
    current_weather = weather_list[0]
    current_temperature = weather_list[1]
    humidity = weather_list[2]
    rainfall = weather_list[3] 
    
    # ## 기상상태    
    weather = weather_result(current_weather)
    
    # ## 노면상태    
    road_surface = road_result(weather, current_temperature, rainfall, humidity)    
    
    # 위험지수 데이터 호출
    risk_score = pd.read_csv('요인별 위험지수(ECLO 추가).csv', index_col='Unnamed: 0')
    
    # 조건 설정
    op_area = risk_score['구'] == gu
    op_time = risk_score['주야간'] == current_time
    op_road = risk_score['노면상태'] == road_surface
    op_weather = risk_score['기상상태'] == weather
        
    fil_score = risk_score.loc[op_area & op_time & op_road & op_weather, :]
    
    risk = fil_score.iloc[0, -1]
    
    print(f'조건 : {gu}, {current_time}, 기상({weather}), 노면({road_surface})')
    print(f'-> 위험지수 : {risk}')
"""
조건 : 권선구, 주간, 기상(맑음), 노면(건조)
-> 위험지수 : 71.83567950729916
조건 : 장안구, 주간, 기상(맑음), 노면(건조)
-> 위험지수 : 62.43014086944115
조건 : 팔달구, 주간, 기상(맑음), 노면(건조)
-> 위험지수 : 66.57238466260897
조건 : 영통구, 주간, 기상(맑음), 노면(건조)
-> 위험지수 : 57.974433318178754
"""