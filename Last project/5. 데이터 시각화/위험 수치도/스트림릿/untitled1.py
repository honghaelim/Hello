# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:59:34 2024

@author: HONGHAE
"""

pip install geopandas matplotlib

#%%

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 맑은 고딕 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
fontprop = fm.FontProperties(fname=font_path, size=12)

# json 파일 경로 설정
jason_file_path = 'E:/workspace/python/project/Honghae/30.수원시_법정경계(읍면동).geojson'

# geodataframe으로 json 파일 불러오기
gdf = gpd.read_file(jason_file_path)

# 지도 그리기
fig, ax = plt.subplots(1, 1, figsize=(10,10))
gdf.plot(ax=ax, color='lightgrey', edgecolor='black')

# 동 이름 표시
for idx, row in gdf.iterrows():
    centroid = row['geometry'].centroid
    plt.annotate(text=row['EMD_KOR_NM'], xy=(centroid.x, centroid.y),
                 fontproperties=fontprop, ha='center', fontsize=10)
    
# 타이틀 설정
ax.set_title('수원시 법정 경계', fontsize=15, fontproperties=fontprop)

# 축 끄기
ax.set_axis_off()

# 지도 보여주기
plt.show()