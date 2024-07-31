# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:11:31 2024

@author: HONGHAE
"""

#%%

import streamlit as st
import pandas as pd
import folium
import geopandas as gpd
import json
from streamlit_folium import folium_static

# 수원시 법정경계(읍면동) GeoJSON 파일 경로
geojson_path = "E:/workspace/python/project/Honghae/30.수원시_법정경계(읍면동).geojson"

# Streamlit 앱 제목
st.title("수원시 지도")

# GeoJSON 파일을 GeoDataFrame으로 읽어옴
gdf = gpd.read_file(geojson_path)

# Folium 지도 객체 생성
m = folium.Map(location=[37.2636, 127.0286], zoom_start=11)

# GeoDataFrame의 geometry 데이터를 Folium GeoJson으로 추가
folium.GeoJson(gdf).add_to(m)

# Folium 지도를 Streamlit에 표시
folium_static(m)
