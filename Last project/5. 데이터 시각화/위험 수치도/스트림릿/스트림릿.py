# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:59:25 2024

@author: HONGHAE
"""

import streamlit as st
import pandas as pd
import json
from streamlit_folium import folium_static
import folium

# 데이터 불러오기
df = pd.read_csv('./요인별 위험지수(ECLO 추가).csv', encoding='cp949')
df['구'] = '수원시' + ' ' + df['구']

# GeoJSON 파일 불러오기
with open('29.수원시_법정경계(시군구).geojson', encoding='utf-8') as f:
    data = json.load(f)

# 수원시 중심부의 위도, 경도
center = [37.2636, 127.0286]

# 맵이 center에 위치하고, zoom 레벨은 11로 시작하는 맵 m 생성
m = folium.Map(location=center, zoom_start=10)

# Choropleth 레이어를 만들고, 맵 m에 추가
folium.Choropleth(
    geo_data=data,
    data=df,
    columns=('구', 'eclo_risk_mul'),
    key_on='feature.properties.SIG_KOR_NM',
    fill_color='BuPu',
    legend_name='ECLO',
).add_to(m)

# 각 구에 대한 팝업 추가
for feature in data['features']:
    properties = feature['properties']
    name = properties['SIG_KOR_NM']
    eclo_risk_mul = df[df['구'] == name]['eclo_risk_mul'].values[0]
    eclo_risk_mul_rounded = round(eclo_risk_mul, 2)
    popup_text = f'{name}<br>ECLO: {eclo_risk_mul_rounded}'
    popup = folium.Popup(popup_text, max_width=300)
    folium.GeoJson(
        feature,
        name=name,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'black'},
        tooltip=name,
        popup=popup
    ).add_to(m)

# 스트림릿 애플리케이션을 구성
st.markdown('<h1 style="text-align: center;">수원시 ECLO 위험지수 지도</h1>', unsafe_allow_html=True)
folium_static(m)

#%%

## 스트림릿 실행 방법
# 1. 아나콘다 파워셸 프롬프트 실행 후 이 파일을 저장한 경로로 설정
# 2. pip install (streamlit / folium / streamlit-folium) 등 3가지 라이브러리 설치 진행
# 3. 위의 두가지 상황을 충족 시킨 후 프롬프트 콘솔에 "streamlit run 파일명.py" 실행하면 끝 
