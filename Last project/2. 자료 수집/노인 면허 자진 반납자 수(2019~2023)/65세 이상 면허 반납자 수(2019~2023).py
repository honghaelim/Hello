# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:54:22 2024

@author: ParkBumCheol
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('경찰청_시도 경찰청별 고령운전자 자진반납 현황_20231231.csv', encoding='cp949')
df
#%%

df = df.dropna(axis=0)
df = df.loc[:, ~df.columns.str.contains('전체')] # '전체' 단어가 포함된 열 삭제
df.rename(columns={'2015 65세 이상': '2015', '2016 65세 이상': '2016', '2017 65세 이상': '2017', '2018 65세 이상': '2018', '2019 65세 이상': '2019', '2020 65세 이상': '2020', '2021 65세 이상': '2021', '2022 65세 이상': '2022', '2023 65세이상': '2023'}, inplace=True)
df
#%%
df = df.drop(['2015','2016','2017','2018'], axis=1)
df

#%%

#df = df.astype({'지방청':'string'})
df.dtypes

#%%
df = df.transpose()

#%%
df = df.reset_index()
#%%
# 열 이름 변경
df.index.name = 'year'
#%%
df.columns = df.iloc[0]
#%%
# drop the first row
df = df.iloc[1:]
#%%
# print the dataframe
print(df)
#%%
df['경기'] = df['경기남부'] + df['경기북부']
df = df.drop(['경기남부', '경기북부'], axis=1)
#%%
df = df [['서울', '경기', '부산', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '대구', '인천', '광주', '대전', '울산', '세종', '제주']]
df.to_excel("65세 이상 면허 반납자 수(2019~2023).xlsx")