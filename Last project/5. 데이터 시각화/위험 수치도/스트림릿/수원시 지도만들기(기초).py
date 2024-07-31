# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:06:28 2024

@author: HONGHAE
"""

import streamlit as st
import folium
from folium.plugins import MarkerCluster
import pandas as pd

#%%

# 데이터 불러오기
@st.cache
def load_data():
    data = pd.read_csv("E:/workspace(Honghae)/python/project/2024_Accident_Project/SeJeong/2.요인별위험지수계산/요인별 위험지수(가중치부여).csv")
    return data
def main():
    st.title("수원시 사고 관련 지도")
    
    data = load_data()
    
    # 지도 중심 좌표 설정
    map_center = [37.2636, 127.0286]
    
    # 지도 생성
    my_map = folium.Map(location=map_center, zoom_strat=12)
    
    # MarkerCluster 객체 생성
    marker_cluster = MarkerCluster().add_to(my_map)
    
    # 사고 위치를 지도에 추가
    for idx, row in data.iterrows():
        folium.Marker([row['위도'], row['경도']], popup=row['사고내용']).add_to(marker_cluster)
    
    # streamlit에 지도 표시
    folium_static(my_map)
    
if __name__ == "__main__":
    main()

    
    