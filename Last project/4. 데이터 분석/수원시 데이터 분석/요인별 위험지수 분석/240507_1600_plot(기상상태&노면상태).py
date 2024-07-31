# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:00:37 2024

@author: ksj
"""

import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

#%%
df = pd.read_excel('./The elderly driver traffic accidents(suwon).xlsx')

# [사고일시] -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')  ## 2023-01-01 00:00:00

# 날짜(object)                   
df['날짜'] = df['사고일시'].dt.date      
#   일
df['일'] = df['사고일시'].dt.day

#%% 일별 기상상태별 발생 건수 히트맵
pivot_table = df.pivot_table(index='기상상태', columns='날짜', values='사고번호', aggfunc=len)

fig, ax = plt.subplots(figsize=(10, 6))
cmap = 'YlOrBr'  # 원하는 색상 지도 선택 가능 (예: 'YlOrBr', 'RdGy', 'Blues', 'coolwarm' 등)
# 히트맵 그리기
ax.pcolor(pivot_table, cmap=cmap)
# 축 라벨 및 제목 설정
plt.xlabel('날짜')
plt.ylabel('기상상태')
plt.title('일별 기상상태별 발생 건수 히트맵')
# 눈금 표시
plt.xticks(range(len(pivot_table.columns)))
plt.yticks(range(len(pivot_table.index)))
# 그래프 출력
plt.tight_layout()
plt.show()

plt.show()

#%%
accidents = df.groupby('기상상태')['사고번호'].count().reset_index()
accidents.columns = ['기상상태', '사고건수']

plt.bar(accidents['기상상태'], accidents['사고건수'])

#%% 일별 노면상태별 발생 건수 히트맵
pivot_table = df.pivot_table(index='노면상태', columns='날짜', values='사고번호', aggfunc=len)

fig, ax = plt.subplots(figsize=(10, 6))
cmap = 'YlOrBr'  # 원하는 색상 지도 선택 가능 (예: 'YlOrBr', 'RdGy', 'Blues', 'coolwarm' 등)
# 히트맵 그리기
ax.pcolor(pivot_table, cmap=cmap)
# 축 라벨 및 제목 설정
plt.xlabel('날짜')
plt.ylabel('노면상태')
plt.title('일별 노면상태별 발생 건수 히트맵')
# 눈금 표시
plt.xticks(range(len(pivot_table.columns)))
plt.yticks(range(len(pivot_table.index)))
# 그래프 출력
plt.tight_layout()
plt.show()

plt.show()

#%%
accidents = df.groupby('노면상태')['사고번호'].count().reset_index()
accidents.columns = ['노면상태', '사고건수']

plt.bar(accidents['노면상태'], accidents['사고건수'])

#%%
table = df.pivot_table(index = '노면상태', columns = '기상상태', values='사고번호', aggfunc = len)

import seaborn as sns
sns.heatmap(table, annot=True, fmt = 'd', cmap ='YlGnBu', linewidth = .5, cbar = False)
plt.show()

#%% 일별 노면상태별 발생 건수 히트맵
pivot_table = df.pivot_table(index='법규위반', columns='일', values='사고번호', aggfunc=len)

fig, ax = plt.subplots(figsize=(10, 6))
cmap = 'YlOrBr'  # 원하는 색상 지도 선택 가능 (예: 'YlOrBr', 'RdGy', 'Blues', 'coolwarm' 등)
# 히트맵 그리기
ax.pcolor(pivot_table, cmap=cmap)
# 축 라벨 및 제목 설정
plt.xlabel('일')
plt.ylabel('법규위반')
plt.title('일별 법규위반별 발생 건수 히트맵')
# 눈금 표시
plt.xticks(range(len(pivot_table.columns)))
plt.yticks(range(len(pivot_table.index)))
# 그래프 출력
plt.tight_layout()
plt.show()

plt.show()

#%%
accidents = df.groupby('법규위반')['사고번호'].count().reset_index()
accidents.columns = ['법규위반', '사고건수']

plt.bar(accidents['법규위반'], accidents['사고건수'])

df['법규위반'].value_counts()
"""
법규위반
안전운전불이행        948
안전거리미확보        287
신호위반              262
교차로운행방법위반    106
기타                   66
중앙선침범             60
보행자보호의무위반     51
차로위반               43
직진우회전진행방해     35
불법유턴               22
Name: count, dtype: int64
"""
