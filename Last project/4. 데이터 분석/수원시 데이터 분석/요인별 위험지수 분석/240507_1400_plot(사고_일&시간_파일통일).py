# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:35:26 2024

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
df['날짜'] = df['사고일시'].dt.date                                            ## 2023-01-01
# 시간(int) 
# df['시간'] = df['사고일시'].dt.time                                          ## 23:00:00
df['시간'] = df['사고일시'].dt.hour      
#   연
df['연'] = df['사고일시'].dt.year
#   월
df['월'] = df['사고일시'].dt.month
#   일
df['일'] = df['사고일시'].dt.day

#%% 월별 사고 발생 건수
df['연'].unique()
## array([2021, 2022, 2023])

month_accidents = df.groupby(['연', '월'])['사고번호'].count().reset_index()
month_accidents.columns = ['연', '월', '사고건수']

plt.figure(figsize=(12, 6))
plt.plot(month_accidents[month_accidents['연']==2021]['월'], month_accidents[month_accidents['연']==2021]['사고건수'], linestyle='-', label = '2021')
plt.plot(month_accidents[month_accidents['연']==2022]['월'], month_accidents[month_accidents['연']==2022]['사고건수'], linestyle='-', label = '2022')
plt.plot(month_accidents[month_accidents['연']==2023]['월'], month_accidents[month_accidents['연']==2023]['사고건수'], linestyle='-', label = '2023')
plt.title('월별 사고 건수')
plt.xlabel('사고 월')
plt.ylabel('사고 건수')
plt.legend(loc='best')
plt.grid(True)

#%% 날짜별 사고발생 건수
daily_accidents = df.groupby('날짜')['사고번호'].count().reset_index()
daily_accidents.columns = ['사고일자', '사고건수']

plt.figure(figsize=(12, 6))
plt.plot(daily_accidents['사고일자'], daily_accidents['사고건수'], linestyle='-')
plt.title('일별 사고 건수')
plt.xlabel('사고 일자')
plt.ylabel('사고 건수')
plt.grid(True)
plt.show()

#%% 시간대별 사고발생 건수                       
time_accidents = df.groupby('시간')['사고번호'].count().reset_index()
time_accidents.columns = ['시간대', '사고건수']

plt.figure(figsize=(12, 6))
plt.plot(time_accidents['시간대'], time_accidents['사고건수'], linestyle='-')
plt.title('시간대별 사고 건수')
plt.xlabel('시간대')
plt.ylabel('사고 건수')
plt.grid(True)
plt.show()

#%% 월별 일별 사고 발생 건수 히트맵
pivot_table = df.pivot_table(index='월', columns='일', values='사고번호', aggfunc=len)

fig, ax = plt.subplots(figsize=(10, 6))
cmap = 'coolwarm'  # 원하는 색상 지도 선택 가능 (예: 'YlOrBr', 'RdGy', 'Blues', 'coolwarm' 등)
# 히트맵 그리기
ax.pcolor(pivot_table, cmap=cmap)
# 축 라벨 및 제목 설정
plt.xlabel('일')
plt.ylabel('월')
plt.title('월별 일별 사고 발생 건수 히트맵')
# 눈금 표시
plt.xticks(range(len(pivot_table.columns)))
plt.yticks(range(len(pivot_table.index)))
# 그래프 출력
plt.tight_layout()
plt.show()

plt.show()
