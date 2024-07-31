import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, date, time
import re
import streamlit as st
import json
from streamlit_folium import folium_static
import folium

# 현재 날씨 웹 스크래핑
def weather_info(gu):
    location = "수원시 " + gu
    search_query = location + " 날씨"
    search_url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query="
    url = search_url + search_query

    html_weather = requests.get(url).text
    soup_weather = BeautifulSoup(html_weather, "lxml")

    current_weather_condition = soup_weather.find("p", {"class": "summary"})
    if current_weather_condition:
        current_weather = current_weather_condition.text.split()[-1]
    else:
        current_weather = "정보 없음"

    current_temperature_element = soup_weather.find("div", {"class": "temperature_text"})
    if current_temperature_element:
        current_temperature = float(re.sub(r'[^\d.]', '', current_temperature_element.text.replace('현재 온도', '')))
    else:
        current_temperature = None

    current_bodily_sensation = soup_weather.select('div.sort dd')
    if len(current_bodily_sensation) >= 3:
        sensation, humidity, wind_speed = [x.get_text() for x in current_bodily_sensation[:3]]
        try:
            humidity = float(re.sub(r'[^\d.]', '', humidity))
        except ValueError:
            humidity = None
    else:
        humidity = None

    rainfall_element = soup_weather.find_all('div', class_='data_inner')
    if rainfall_element:
        try:
            rainfall = float(re.sub(r'[^\d.]', '', rainfall_element[0].text.strip()))
        except ValueError:
            rainfall = 0.0
    else:
        rainfall = None

    return current_weather, current_temperature, humidity, rainfall

# 미래 날씨 웹 스크래핑 (12시간 후까지만)
def fweather_info(fgu):
    location = "수원시 " + fgu
    search_query = location + " 날씨"
    search_url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query="
    url = search_url + search_query

    html_weather = requests.get(url).text
    soup_weather = BeautifulSoup(html_weather, "lxml")

    fweather_data = []

    li_elements = soup_weather.find_all('li', class_='_li')
    for i, li in enumerate(li_elements):
        if i >= 12:
            break

        time_element = li.find('dt', class_='time')
        if time_element:
            time_text = time_element.text.strip()
        else:
            time_text = "정보 없음"

        fweather_element = li.find('span', class_='blind')
        if fweather_element:
            fweather_text = fweather_element.text.strip()
        else:
            fweather_text = "정보 없음"

        temperature_element = li.find('span', class_='num')
        if temperature_element:
            temperature_text = temperature_element.text.strip()
        else:
            temperature_text = "정보 없음"
        
        rainfall_wrap = soup_weather.find('div', class_='graph_wrap rainfall')
        rainfall_element = "0"
        if rainfall_wrap:
            rainfall_elements = rainfall_wrap.find_all('div', class_='data_inner full')
            if i < len(rainfall_elements):
                rainfall_element = rainfall_elements[i].text.strip()
        
        humidity_wrap = soup_weather.find('div', class_='humidity_graph_box')
        humidity_element = "0"
        if humidity_wrap:
            humidity_elements = humidity_wrap.find_all('div', class_='data_inner')
            if i < len(humidity_elements):
                humidity_span = humidity_elements[i].find('span', class_='num')
                if humidity_span:
                    humidity_element = humidity_span.text.strip()

        fweather_data.append({
            '시간': time_text,
            '기상상태': fweather_text,
            '온도': temperature_text,
            '강수량': rainfall_element,
            '습도': humidity_element
        })

    return fweather_data

# 주야간 구분 함수
def timesplit(current_time):
    if current_time.hour >= 7 and current_time.hour < 20:
        current = "주간"
    else:
        current = "야간"
    return current

# 현재 기상상태
def weather_result(weather):
    if weather in ['맑음', '구름조금', '가끔비', '가끔 비,눈', '가끔 눈', '흐린 후 갬', '뇌우 후 갬', '비 후 갬', '눈 후 갬']:
        result = '맑음'
    elif weather in ['구름많음', '흐림', '안개']:
        result = '흐림'
    elif weather in ['약한비', '비', '강한비', '소나기', '흐려져 비']:
        result = '비'
    elif weather in ['약한눈', '눈', '강한눈', '진눈깨비', '소낙눈', '우박', '흐려져 눈']:
        result = '눈'
    else:
        result = '기타'
    return result
# 미래 기상상태
def fweather_result(fweather):
    if fweather in ['맑음', '구름조금', '가끔비', '가끔 비,눈', '가끔 눈', '흐린 후 갬', '뇌우 후 갬', '비 후 갬', '눈 후 갬']:
        result = '맑음'
    elif fweather in ['구름많음', '흐림', '안개']:
        result = '흐림'
    elif fweather in ['약한비', '비', '강한비', '소나기', '흐려져 비']:
        result = '비'
    elif fweather in ['약한눈', '눈', '강한눈', '진눈깨비', '소낙눈', '우박', '흐려져 눈']:
        result = '눈'
    else:
        result = '기타'
    return result

# 현재 노면상태
def road_result(weather, current_temperature, rainfall, humidity):
    if weather == '비':
        result = '젖음/습기'
    elif weather == '눈':
        if rainfall > 50 and current_temperature < 0:
            result = '적설'
        elif 1 < rainfall <= 50 and current_temperature < 0:
            result = '서리/결빙'
        else:
            result = '젖음/습기'
    elif weather == '맑음':
        if rainfall > 50 and current_temperature < 0:
            result = '적설'
        elif 1 <= rainfall <= 50 and current_temperature < 0:
            result = '서리/결빙'
        elif rainfall >= 1 and current_temperature >= 0:
            result = '젖음/습기'
        elif rainfall == 0 and current_temperature > 0:
            result = '건조'
        else:
            result = '기타'
    elif weather == '흐림':
        if rainfall >= 50 and current_temperature < 0:
            result = '적설'
        elif 1 < rainfall < 50:
            result = '젖음/습기'
        else:
            result = '건조'
    else:
        result = '기타'
    return result

# 미래 노면상태
def froad_result(fweather, ftemperature, frainfall, fhumidity):
    if fweather == '비':
        result = '젖음/습기'
    elif fweather == '눈':
        if frainfall > 50 and ftemperature < 0:
            result = '적설'
        elif 1 < frainfall <= 50 and ftemperature < 0:
            result = '서리/결빙'
        else:
            result = '젖음/습기'
    elif fweather == '맑음':
        if frainfall > 50 and ftemperature < 0:
            result = '적설'
        elif 1 <= frainfall <= 50 and ftemperature < 0:
            result = '서리/결빙'
        elif frainfall >= 1 and ftemperature >= 0:
            result = '젖음/습기'
        elif frainfall == 0 and ftemperature > 0:
            result = '건조'
        else:
            result = '기타'
    elif fweather == '흐림':
        if frainfall >= 50 and ftemperature < 0:
            result = '적설'
        elif 1 < frainfall < 50:
            result = '젖음/습기'
        else:
            result = '건조'
    else:
        result = '기타'
    return result

# 현재 구별 위험지수
def calculate_risk(selected_gu):
    current_weather, current_temperature, humidity, rainfall = weather_info(selected_gu)

    current_time_period = timesplit(datetime.now())
    current_weather_status = weather_result(current_weather)
    current_road_condition = road_result(current_weather_status, current_temperature, rainfall, humidity)

    risk_score = pd.read_csv('요인별 위험지수(ECLO 추가).csv', index_col='Unnamed: 0', encoding='cp949')

    op_area = risk_score['구'] == selected_gu
    op_time = risk_score['주야간'] == current_time_period
    op_road = risk_score['노면상태'] == current_road_condition
    op_weather = risk_score['기상상태'] == current_weather_status

    fil_score = risk_score.loc[op_area & op_time & op_road & op_weather, :]
    
    if fil_score.empty:
        risk = 0  # 기본값 설정
    else:
        risk = fil_score.iloc[0, -1]

    return {
        '위험지수': round(risk, 2),
        '날씨': current_weather,
        '온도': current_temperature,
        '습도': humidity,
        '강수량': rainfall,
        '주야간': current_time_period,
        '노면상태': current_road_condition,
        '기상상태': current_weather_status
    }

# 미래 구별 위험지수
def calculate_future_risk(selected_gu, selected_time):
    fweather_list = fweather_info(selected_gu)
    
    selected_time_str = selected_time.strftime("%H시")
    for forecast in fweather_list:
        if forecast['시간'] == selected_time_str:
            fweather = forecast['기상상태']
            ftemperature = float(forecast['온도'].replace('°', ''))
            try:
                fhumidity = float(forecast['습도'])
            except ValueError:
                fhumidity = None  # 또는 기본값 설정
            try:
                frainfall = float(forecast['강수량'])
            except ValueError:
                frainfall = None  # 또는 기본값 설정

            ftime_period = timesplit(datetime.combine(selected_date, selected_time))
            fweather = fweather_result(fweather)
            frode_condition = froad_result(fweather, ftemperature, frainfall, fhumidity)

            risk_score = pd.read_csv('요인별 위험지수(ECLO 추가).csv', index_col='Unnamed: 0', encoding='cp949')

            op_area = risk_score['구'] == selected_gu
            op_time = risk_score['주야간'] == ftime_period
            op_road = risk_score['노면상태'] == frode_condition
            op_weather = risk_score['기상상태'] == fweather

            fil_score = risk_score.loc[op_area & op_time & op_road & op_weather, :]
            
            if fil_score.empty:
                risk = 0  # 기본값 설정
            else:
                risk = fil_score.iloc[0, -1]

            return {
                '위험지수': round(risk, 2),
                '날씨': fweather,
                '온도': ftemperature,
                '습도': fhumidity,
                '강수량': frainfall,
                '주야간': ftime_period,
                '노면상태': frode_condition,
                '기상상태': fweather
            }
    return None  # 예보 시간에 해당하는 데이터가 없을 경우

st.title('수원시 시간대별 위험지수 및 날씨 정보')

gu_list = ['권선구', '장안구', '팔달구', '영통구']
selected_gu = st.sidebar.selectbox('Select 구', gu_list)

selected_date = st.sidebar.date_input('날짜 선택', date.today())
st.write('Selected date:', selected_date)

time_options = [f"{hour:02d}:00" for hour in range(24)]
#selected_time_str = st.sidebar.selectbox('Select a time', time_options)

#selected_hour, selected_minute = map(int, selected_time_str.split(':'))
#selected_time = time(selected_hour, selected_minute)

# 시간별 날씨 정보를 가져옴
weather_data = fweather_info(selected_gu)
weather_df = pd.DataFrame(weather_data)

# 특정 시간 선택 시 위험지수
def calculate_risk_at_specific_time(time_str):
    specific_time_risk_data = []
    for fgu in gu_list:
        future_risk_data = calculate_future_risk(fgu, datetime.strptime(time_str, '%H시').time())
        if future_risk_data:
            specific_time_risk_data.append([fgu] + list(future_risk_data.values()))
        else:
            specific_time_risk_data.append([fgu, '정보 없음', '정보 없음', '정보 없음', '정보 없음', '정보 없음', '정보 없음', '정보 없음', '정보 없음'])
    specific_time_risk_df = pd.DataFrame(specific_time_risk_data, columns=['구', '위험지수', '날씨', '온도', '습도', '강수량', '주야간', '노면상태', '기상상태'])
    return specific_time_risk_df

selected_time_str_for_risk = st.sidebar.selectbox('위험지수를 확인할 시간대를 선택하세요', weather_df['시간'].unique())
specific_time_risk_df = calculate_risk_at_specific_time(selected_time_str_for_risk)

st.write(f'선택한 시간({selected_time_str_for_risk})에 대한 모든 구의 위험지수 데이터')
st.dataframe(specific_time_risk_df)

# 선택된 구에 대한 현재 날씨 및 위험 지수 정보를 가져옴
current_risk_data = calculate_risk(selected_gu)

st.write(f'{selected_time_str_for_risk} {selected_gu}의 위험지수 데이터')
st.write(current_risk_data)

with open('29.수원시_법정경계(시군구).geojson', encoding='utf-8') as f:
    data = json.load(f)

sw_risk = specific_time_risk_df[['구', '위험지수']].copy()
sw_risk['구'] = '수원시 ' + sw_risk['구']

center = [37.2636, 127.0286]

m = folium.Map(location=center, zoom_start=11)

folium.Choropleth(
    geo_data=data,
    data=sw_risk,
    columns=('구', '위험지수'),
    key_on='feature.properties.SIG_KOR_NM',
    fill_color='YlOrRd',
    legend_name='교통사고 발생 위험도',
).add_to(m)

for feature in data['features']:
    properties = feature['properties']
    name = properties['SIG_KOR_NM']
    sw_risk_rounded = sw_risk.loc[sw_risk['구'] == name, '위험지수'].values[0]
    popup_text = f'{name}<br>사고위험지수: {sw_risk_rounded}'
    popup = folium.Popup(popup_text, max_width=300)
    folium.GeoJson(
        feature,
        name=name,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'black'},
        tooltip=name,
        popup=popup
    ).add_to(m)

st.markdown('<h1 style="text-align: center;">수원시 위험지수 지도</h1>', unsafe_allow_html=True)
folium_static(m)
