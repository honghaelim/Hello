# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:45:27 2024

@author: HONGHAE
"""

import streamlit as st
import folium
import geopandas as gpd
from streamlit.components.v1 import html
from streamlit_folium import folium_static

# Streamlit 설정
st.set_page_config(page_title="수원시 법정경계 지도", layout="wide")

# GeoJSON 파일 로드
geojson_file = "E:/workspace/python/project/Honghae/30.수원시_법정경계(읍면동).geojson"
gdf = gpd.read_file(geojson_file)

# Folium 지도 생성
m = folium.Map(location=[37.2635, 127.0286], zoom_start=12)

# 각 읍면동에 팝업 추가
for _, row in gdf.iterrows():
    # 경계 중심 좌표를 계산
    centroid = row['geometry'].centroid
    # 팝업 내용
    popup_text = f"{row['EMD_KOR_NM']}"
    # 팝업 추가
    folium.Marker([centroid.y, centroid.x], popup=popup_text).add_to(m)

# 지도 출력
folium_static(m)

# Streamlit 실행 지침
st.write("수원시 법정경계(읍면동) 지도입니다.")
