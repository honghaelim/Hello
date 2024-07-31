# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:54:24 2024

@author: ksj
"""
import pandas as pd

df_merge = pd.read_csv('./행정구역별_인구수_2019-202404.csv', index_col= '행정구역')

#%%
import matplotlib.pyplot as plt
# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

###############################################################################
# 연도별 '경기도' 전체/ 65세이상 인구수 변화
###############################################################################
op_col = list(filter(lambda x: '전체' in x, df_merge.columns))
op_df = df_merge.loc['경기도', op_col]

gg_people = pd.DataFrame({'연도' : [2019, 2020, 2021, 2022, 2023, 2024],
                          '전체' : [op_df.iloc[0], op_df.iloc[2], op_df.iloc[4], op_df.iloc[6], op_df.iloc[8], op_df.iloc[10]],
                          '65세이상' : [op_df.iloc[1], op_df.iloc[3], op_df.iloc[5], op_df.iloc[7], op_df.iloc[9], op_df.iloc[11]]
                          })
gg_people['65세이상비율'] = (gg_people['65세이상'] / gg_people['전체']) * 100 

#%% 시각화
# 연도별 인구수
plt.plot(gg_people['연도'], gg_people[['전체', '65세이상']], linestyle='-')
plt.title('연도별 인구수')
plt.xlabel('연도')
plt.ylabel('전체인구수')
plt.grid(True)
plt.show()

# 연도별 65세이상 인구비율
plt.plot(gg_people['연도'], gg_people['65세이상비율'], linestyle='-')
plt.title('연도별 65세이상 인구비율')
plt.xlabel('연도')
plt.ylabel('65세이상 인구비율')
plt.grid(True)
plt.show()

# 연도별 전체 인구수 & 65세이상 인구비율
plt.figure(figsize=(10, 6))
ax = plt.axes()  # 하나의 축 공유
ax.bar(gg_people['연도'], gg_people['전체'], alpha=0.7, color='red', label='전체인구수')
ax2 = ax.twinx()  # 오른쪽 축 생성
ax2.plot(gg_people['연도'], gg_people['65세이상비율'], color='blue', label='65세이상 인구비율')
ax.set_xlabel('연도')
ax.set_ylabel('인구수(명)')
ax2.set_ylabel('인구비율(%)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax.set_ylim([0, gg_people['전체'].max() + gg_people['전체'].max() * 0.1])  # 막대와 선이 겹치지 않도록 Y 축 범위 조정
plt.show()

#%%
# 노인 구분에 따른 인구수 비율
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
labels = ['일반', '노인']
tot_driver = gg_people.iloc[5, 1] # 2024년 전체
tot_elderly_driver = gg_people.iloc[5, 2] # 2024년 65세이상
tot_regular_driver = tot_driver - tot_elderly_driver

sizes = [tot_regular_driver, tot_elderly_driver]
colors = ['lightgreen', 'lightsalmon']
explode = (0.1, 0)  # 강조를 위한 explode
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('2024년 4월 인구수 구분 비율(%)')

#%%
###############################################################################
# 연도별 '수원시' 전체/ 65세이상 인구수 변화
###############################################################################
op_col = list(filter(lambda x: '전체' in x, df_merge.columns))
op_df = df_merge.loc['수원시', op_col]

sw_people = pd.DataFrame({'연도' : [2019, 2020, 2021, 2022, 2023, 2024],
                          '전체' : [op_df.iloc[0], op_df.iloc[2], op_df.iloc[4], op_df.iloc[6], op_df.iloc[8], op_df.iloc[10]],
                          '65세이상' : [op_df.iloc[1], op_df.iloc[3], op_df.iloc[5], op_df.iloc[7], op_df.iloc[9], op_df.iloc[11]]
                          })
sw_people['65세이상비율'] = (sw_people['65세이상'] / sw_people['전체']) * 100 


# 연도별 전체 인구수 & 65세이상 인구비율
plt.figure(figsize=(10, 6))
ax = plt.axes()  # 하나의 축 공유
ax.bar(sw_people['연도'], sw_people['전체'], alpha=0.7, color='red', label='전체인구수')
ax2 = ax.twinx()  # 오른쪽 축 생성
ax2.plot(sw_people['연도'], sw_people['65세이상비율'], color='blue', label='65세이상 인구비율')
ax.set_xlabel('연도')
ax.set_ylabel('인구수(백만명)')
ax2.set_ylabel('인구비율(%)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax.set_ylim([0, sw_people['전체'].max() + sw_people['전체'].max() * 0.1])  # 막대와 선이 겹치지 않도록 Y 축 범위 조정
plt.show()

#%%
###############################################################################
# 행정구역별 65세이상 인구비율
###############################################################################
op_col = list(filter(lambda x: '2024' in x, df_merge.columns))
op_df = df_merge.loc[:, op_col]

y24_people = op_df.copy()

y24_people['전체_65세이상비율'] = (y24_people['2024_전체_65세이상'] / y24_people['2024_전체']) * 100 
y24_people['남자_65세이상비율'] = (y24_people['2024_남자_65세이상'] / y24_people['2024_남자']) * 100 
y24_people['여자_65세이상비율'] = (y24_people['2024_여자_65세이상'] / y24_people['2024_여자']) * 100 

# '경기도, 수원시, 장안구, 권선구, 팔달구, 영통구' 추출
gg_sw_df = y24_people.iloc[0:6, :]
# 행정구역별 65세이상 인구비율
plt.figure(figsize=(10, 6))
plt.bar(gg_sw_df.index, gg_sw_df['전체_65세이상비율'], alpha=0.7)
plt.title('행정구역별 65세이상 인구비율')
plt.xlabel('행정구역')
plt.ylabel('인구비율(%)')
plt.show()

# '수원시, 장안구, 권선구, 팔달구, 영통구' 추출
sw_df = y24_people.iloc[1:6, :]
# 행정구역별 전체 인구수 & 65세이상 인구비율
plt.figure(figsize=(10, 6))
ax = plt.axes()  # 하나의 축 공유
ax.bar(sw_df.index, sw_df['2024_전체'], alpha=0.7, color='red', label='인구수')
ax2 = ax.twinx()  # 오른쪽 축 생성
ax2.plot(sw_df.index, sw_df['전체_65세이상비율'], color='blue', label='65세이상 인구비율')
ax.set_xlabel('행정구역')
ax.set_ylabel('인구수(백만명)')
ax2.set_ylabel('인구비율(%)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax.set_ylim([0, sw_df['2024_전체'].max() + sw_df['2024_전체'].max() * 0.1])  # 막대와 선이 겹치지 않도록 Y 축 범위 조정
plt.title('수원시 구별 65세이상 인구비율')
plt.show()

