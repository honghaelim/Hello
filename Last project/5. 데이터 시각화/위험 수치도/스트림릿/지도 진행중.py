# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:57:03 2024

@author: HONGHAE
"""

import folium
import pandas as pd
import json

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

# 맵 m을 출력
m

# 맵 m을 저장
m.save('map2.html')