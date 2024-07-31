###############################################################################
# 동일 폴더 내 필요 파일
#   - 요인별 위험지수_동(ECLO 추가).csv
#   - 30.수원시_법정경계(읍면동).geojson
###############################################################################
###############################################################################
# 아나콘다 파워셸 프롬프트
#   >>> conda activate YSIT24
#   >>> cd E:/Workspace/!project_team/SeJeong/7.스트림릿_동별_위험지수지도/ (파일저장경로)
#   >>> streamlit run 240517_1530_suwon_dong_accident_forecast.py
###############################################################################
# 라이브러리
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
import json
from streamlit_folium import folium_static
import folium
###############################################################################

#%% 데이터 호출 및 조건 함수
###############################################################################
# 네이버 날씨(기상청)
#   - 입력 : 동
#   - 출력 : 날씨(current_weather), 온도(current_temperature), 습도(humidity), 강수량(rainfall)
def weather_info(dong) :
    location = "수원시" + dong
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

# 주간/야간 구분 함수
#   - 주간, 야간
def timesplit(current_time):
    if current_time.hour>=7 and current_time.hour<20:
        current="주간"
    else:
        current="야간"
    return current

# 현재 날씨에 따른 기상상태 
#   - 맑음, 흐림, 비, 눈, 기타
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

# 기상상태, 현재온도, 강수량, 습도에 따른 노면상태
#   - 건조, 젖음/습기, 서리/결빙, 적설, 기타 
def road_result(weather, current_temperature, rainfall, humidity) :
    if weather == '비':
        result='젖음/습기'
    elif weather =='눈':
        if rainfall>50 and current_temperature<0 :
            result='적설'
        elif 1<rainfall<=50 and current_temperature<0:
            result='서리/결빙'
        else:
            result='젖음/습기'
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
    elif weather=='흐림':
        if rainfall>=50 and current_temperature<0 :
            result='적설'
        elif 1<rainfall<50 :
            result=='젖음/습기'
        else:
            result='건조'  
    else:
        result='기타'
        
    return result

#%% 조건에 따른 위험지수(eclo_risk_mul) 추출
#   - 위험지수(eclo_risk_mul) = 요인별 (risk(순위 역순)*ECLO)의 합계
###############################################################################

# ## '동' 목록(테이블)
risk_score = pd.read_csv('요인별 위험지수_동(ECLO 추가).csv', index_col='Unnamed: 0')

dong_list = risk_score.loc[:, '동'].unique()
dong_risk = pd.DataFrame({'위험지수' : [0.0] * len(dong_list) }, index=dong_list)

# ## '동'별 위험지수, 날씨, 온도, 습도, 강수량, 주야간, 노면상태, 기상상태 테이블 생성
for dong in dong_list :
    # 기상청 데이터 호출
    weather_list = weather_info(dong)
    
    current_weather = weather_list[0]
    current_temperature = weather_list[1]
    humidity = weather_list[2]
    rainfall = weather_list[3] 
    
    # 조건값 생성 및 설정
    # 주야간
    time = datetime.now()
    current_time = timesplit(time)    
    # 기상상태    
    weather = weather_result(current_weather)    
    # 노면상태    
    road_surface = road_result(weather, current_temperature, rainfall, humidity)    
    
    # 위험지수 데이터 호출 및 조건 설정    
    op_area = risk_score['동'] == dong
    op_time = risk_score['주야간'] == current_time
    op_road = risk_score['노면상태'] == road_surface
    op_weather = risk_score['기상상태'] == weather        
    
    # 테이블 생성
    fil_score = risk_score.loc[op_area & op_time & op_road & op_weather, :]
    
    risk = fil_score.iloc[0, -1]
    
    dong_risk.loc[dong,'위험지수'] = round(risk, 2)
    
    dong_risk['날씨'] = current_weather
    dong_risk['온도'] = current_temperature
    dong_risk['습도'] = humidity
    dong_risk['강수량'] = rainfall
    
    dong_risk['주야간'] = current_time
    dong_risk['노면상태'] = road_surface
    dong_risk['기상상태'] = weather

#%% 스트림릿 출력    
###############################################################################
# 조건 일시
st.header('수원시 동별 사고위험지수 및 기상상태')

st.text(f'{time.year}년 {time.month}월 {time.day}일 {time.hour}시 {time.minute}분 기준 ')

# 스트림릿 동별 조건(현재 상태) 테이블
dong_risk = dong_risk.sort_values(by = '위험지수', ascending=False)
dong_risk = dong_risk.reset_index()
# 동 - 구 매칭
area = risk_score.loc[:, ['동','구']].drop_duplicates().set_index('동')
area = area.loc[:, '구'] # DataFrame -> Series (area['오목천동'] 형태로 호출하기 위해)
for x in dong_list :
    dong_risk.loc[dong_risk['index'] == x, '구'] = area[x]

dong_risk.columns = ['동', '위험지수', '날씨', '온도', '습도', '강수량', '주야간', '노면상태', '기상상태', '구']
dong_risk = dong_risk.loc[:, ['구', '동', '위험지수', '날씨', '온도', '습도', '강수량', '주야간', '노면상태', '기상상태']]

st.dataframe(dong_risk)

#%% 수원시 지도 생성
###############################################################################
with open('30.수원시_법정경계(읍면동).geojson', encoding='utf-8') as f:
    data = json.load(f)
    
# json 파일 데이터와 일치
#   - "EMD_KOR_NM": "파장동"
sw_risk = dong_risk.loc[:, ['동','위험지수']]

#%%
name_list = []
# 각 '동'에 대한 팝업 추가
#   -> 오류 발생
for feature in data['features']:
    properties = feature['properties']
    name = properties['EMD_KOR_NM']    
    
    name_list.append(name)
    
d_l = list(dong_list)

for n in name_list :
    if n not in d_l :
        print(n)
## 상광교동 -> 연무동
## 장지동 -> 세류동

sw_plus = pd.DataFrame({'동' : ['상광교동', '장지동'],
                        '위험지수' : [sw_risk.loc[sw_risk['동']=='연무동', '위험지수'].values[0], 
                                  sw_risk.loc[sw_risk['동']=='세류동', '위험지수'].values[0]]})
sw_risk = pd.concat([sw_risk, sw_plus])

# 수원시 중심부의 위도, 경도
center = [37.2636, 127.0286]

# 맵이 center에 위치하고, zoom 레벨은 12로 시작하는 맵 m 생성
m = folium.Map(location=center, zoom_start=12)

# folium.Choropleth 레이어를 만들고, 맵 m에 추가
folium.Choropleth(
    geo_data= data,
    data = sw_risk,
    columns=('동','위험지수'),
    key_on='feature.properties.EMD_KOR_NM',
    fill_color='YlOrRd',
    legend_name='교통사고 발생 가능성',
    ).add_to(m)


#%%  
# 각 '동'에 대한 팝업 추가
#   - 지도에서 '동' 클릭 시 팝업 출력
for feature in data['features']:
    properties = feature['properties']
    name = properties['EMD_KOR_NM']  
    sw_risk['동'] = sw_risk['동'].astype(str)
    sw_risk_rounded = sw_risk.loc[sw_risk['동']==name, '위험지수'].values[0]
    popup_text = f'{name}<br>사고위험지수: {sw_risk_rounded}'
    
    popup = folium.Popup(popup_text, max_width=300)
    
    folium.GeoJson(
        feature,
        name=name,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'black'},
        tooltip=name,
        popup=popup
    ).add_to(m)
    

#%% 스트림릿 출력    
###############################################################################

# 스트림릿 애플리케이션을 구성
st.markdown('<h1 style="text-align: center;">위험지수 지도</h1>', unsafe_allow_html=True)
folium_static(m)

    