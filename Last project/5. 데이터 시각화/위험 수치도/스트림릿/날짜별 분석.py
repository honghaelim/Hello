# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:12:56 2024

@author: HONGHAE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

df = pd.read_excel('E:/workspace(Honghae)/python/project/노인 운전.xlsx')
print(df.head())

#%%

#   날짜(object) 
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')        
df['날짜'] = df['사고일시'].dt.date                                           
df['시간'] = df['사고일시'].dt.hour
   
#%%

# [시군구] -> 구/ 동
gu = []
dong = []
for i in range(len(df)):
    gu.append(df['시군구'].str.split(' ')[i][2])
    dong.append(df['시군구'].str.split(' ')[i][3])
df['구'] = gu 
df['동'] = dong

#%%

""" df.iloc[:, 18:22].columns 
Index(['피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도'], dtype='object')
"""
df.iloc[:, 18:22] = df.iloc[:, 18:22].fillna(0)


#%%

# [사고유형] '차대사람 - 기타' -> '차대사람', '기타'
dep1 = []
dep2 = []
for i in range(len(df)) :
    dep1.append(df['사고유형'].str.split(' - ')[i][0])
    dep2.append(df['사고유형'].str.split(' - ')[i][1])
df['사고유형1'] = dep1
df['사고유형2'] = dep2

#%%

# [도로형태] '단일로 - 기타' -> '단일로', '기타'
dep1 = []
dep2 = []
for i in range(len(df)) :
    dep1.append(df['도로형태'].str.split(' - ')[i][0])
    dep2.append(df['도로형태'].str.split(' - ')[i][1])
df['도로형태1'] = dep1
df['도로형태2'] = dep2

#%%

# [연령] 00세(object) -> 00(int)
# '가해운전자'
df['가해운전자 연령'].unique()
"""
array(['68세', '66세', '69세', '80세', '75세', '65세', '67세', '79세', '82세',
       '73세', '78세', '70세', '74세', '87세', '72세', '71세', '77세', '76세',
       '83세', '81세', '86세', '85세', '84세', '93세', '88세'], dtype=object)
"""
df['가해운전자 연령'] = df['가해운전자 연령'].str[:-1]

df['가해운전자 연령'] = df['가해운전자 연령'].astype('int64')

#%%

df['피해운전자 연령'].unique()
"""
array(['54세', '68세', '25세', '55세', '72세', '50세', '39세', '53세', '45세',
       '47세', '48세', '59세', '33세', '46세', 0, '38세', '42세', '57세', '75세',
       '21세', '43세', '29세', '35세', '84세', '71세', '31세', '66세', '51세',
       '27세', '56세', '64세', '49세', '78세', '32세', '62세', '23세', '73세',
       '19세', '26세', '60세', '40세', '58세', '77세', '30세', '36세', '37세',
       '52세', '18세', '41세', '13세', '74세', '7세', '61세', '44세', '8세', '22세',
       '20세', '63세', '28세', '85세', '65세', '70세', '34세', '24세', '82세',
       '81세', '67세', '80세', '10세', '3세', '83세', '88세', '15세', '17세',
       '86세', '9세', '14세', '69세', '11세', '87세', '미분류', '79세', '16세',
       '92세', '6세', '95세', '2세', '90세', '76세', '12세'], dtype=object)
"""
df['피해운전자 연령'] = df['피해운전자 연령'].str[:-1]
## -> nan(0->nan), '미분'('미분류')

#%%

# -> '미분류' : 0
df['피해운전자 연령'] = df['피해운전자 연령'].replace('미분', 0)
# -> nan : 0
df['피해운전자 연령'] = df['피해운전자 연령'].fillna(0)

df['피해운전자 연령'] = df['피해운전자 연령'].astype('int64')

#%%

df['총사고수'] = len(df)

#%%

# 사고내용의 각 4가지 유형 별 사고 수 계산
group1 = (df['사고내용'] == '사망사고').sum()
print(group1.head())
group2 = (df['사고내용'] == '중상사고').sum()
print(group2.head())
group3 = (df['사고내용'] == '경상사고').sum()
print(group3.head())
group4 = (df['사고내용'] == '부상신고사고').sum()

#%%

# 총 사고수에서 각 4가지 사고 유형의 비율 계산
tot_pct_group1 = (16 / 1880) * 100  ## 0.85%
tot_pct_group2 = (361 / 1880) * 100 ## 19.20%
tot_pct_group3 = (1390 / 1880) * 100 ## 73.93%
tot_pct_group4 = (113 / 1880) * 100 ## 6.01%

#%%

df_group1 = df.groupby(df['사고내용']=='사망사고')
age_group1 = df_group1['가해운전자 연령']
print(age_group1.head())
df_group2 = df.groupby(df['사고내용']=='중상사고')
age_group2 = df_group2['가해운전자 연령']
print(age_group2.head())
df_group3 = df.groupby(df['사고내용']=='경상사고')
age_group3 = df_group3['가해운전자 연령']
print(age_group3.head())
df_group4 = df.groupby(df['사고내용']=='부상신고사고')
age_group4 = df_group4['가해운전자 연령']
print(age_group1.head())