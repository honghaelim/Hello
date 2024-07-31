# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:34:43 2024

@author: ksj
"""

import pandas as pd

df = pd.read_csv('./accidentInfoList_TAAS_경기도수원2023_전체.csv', low_memory=False, encoding='CP949')

#%%
import matplotlib.pyplot as plt


#%%
# 사고일시 -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')

#%%
# 날짜 열 만들기
df['날짜'] = df['사고일시'].dt.date

# 날짜별 사고발생 건수
daily_accidents = df.groupby('날짜')['사고번호'].count().reset_index()
daily_accidents.columns = ['사고일자', '사고건수']

plt.figure(figsize=(12, 6))
plt.plot(daily_accidents['사고일자'], daily_accidents['사고건수'], linestyle='-')
plt.title('일별 사고 건수')
plt.xlabel('사고 일자')
plt.ylabel('사고 건수')
plt.grid(True)
plt.show()

#%%
# 시간 열 만들기
# df['시간'] = df['사고일시'].dt.time ## 23:00:00
df['시간'] = df['사고일시'].dt.hour

# 시간대별 사고발생 건수
time_accidents = df.groupby('시간')['사고번호'].count().reset_index()
time_accidents.columns = ['시간대', '사고건수']

plt.figure(figsize=(12, 6))
plt.plot(time_accidents['시간대'], time_accidents['사고건수'], linestyle='-')
plt.title('시간대별 사고 건수')
plt.xlabel('시간대')
plt.ylabel('사고 건수')
plt.grid(True)
plt.show()
