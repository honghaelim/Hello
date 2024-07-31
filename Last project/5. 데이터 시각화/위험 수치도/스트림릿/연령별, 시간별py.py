# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:28:56 2024

@author: HONGHAE
"""
# 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# 엑셀 파일 불러와서 df 생성
df = pd.read_excel('E:/workspace(Honghae)/python/project/연령층별 시간대별 교통사고.xlsx')
df.head(5)

#%%
# df를 합계와 65세 이상 자료로만 수정
df.loc[0:6,:].head()
df1 = df.loc[0:6,:]
df.loc[43:,:].head()
df2 = df.loc[43:,:]
df = pd.concat([df1, df2])
df = df.drop(df.index[0])

#%%

year_df3 = year_df3.drop(year_df3.columns[2:28], axis=1)
year_df4 = year_df4.drop(year_df4.columns[2:3], axis=1)
year_df5 = year_df5.drop(year_df5.columns[2:4], axis=1)

#%%

# 년도별로 분리
year_df1 = df.loc[:,'기본계획 연령별1':'2019.12'] # 19년도
year_df2 = df.loc[:,'기본계획 연령별1':'2020.12'] # 20년도
year_df3 = df.loc[:,'기본계획 연령별1':'2021.12'] # 21년도
year_df4 = df.loc[:,'기본계획 연령별1':'2022.12'] # 22년도
year_df5 = df.loc[:,'기본계획 연령별1':'2023.12'] # 23년도

#%%

# 년도별 자료 변수 이름 변경
data_2019 = year_df1
data_2020 = year_df2
data_2021 = year_df3
data_2022 = year_df4
data_2023 = year_df5
data_21 = data_2021.transpose()

#%%

# 열 기준으로 세분화 
dt_21_1 = data_21.loc[:,:1]
dt_21_2 = data_21.loc[:,:2]
dt_21_3 = data_21.loc[:,:3]
dt_21_4 = data_21.loc[:,:4]
dt_21_5 = data_21.loc[:,:5]
dt_21_6 = data_21.loc[:,:6]

dt_21_2 = dt_21_2.drop(columns=[1])
dt_21_3 = dt_21_3.drop(columns=[1,2])
dt_21_4 = dt_21_4.drop(columns=[1,2,3])
dt_21_5 = dt_21_5.drop(columns=[1,2,3,4])
dt_21_6 = dt_21_6.drop(columns=[1,2,3,4,5])

#%%

dt_21_1 = dt_21_1.drop(dt_21_1.index[2])
dt_21_2 = dt_21_2.drop(dt_21_2.index[2])
dt_21_3 = dt_21_3.drop(dt_21_3.index[2])
dt_21_4 = dt_21_4.drop(dt_21_4.index[2])
dt_21_5 = dt_21_5.drop(dt_21_5.index[2])
dt_21_6 = dt_21_6.drop(dt_21_6.index[2])



#%%

# 합계 부분 최빈 값 구하기
dt_21_1 = dt_21_1.sort_values(by=dt_21_1[1],ascending=False) ##  합계 2021.10 -> 18~20시
dt_21_2 = dt_21_2.sort_values(by=dt_21_2[2],ascending=False) ##  사망 2021.10 -> 18~20시
dt_21_3 = dt_21_3.sort_values(by=dt_21_3[3],ascending=False) ##  부상 2021.9 -> 16~18시
dt_21_4 = dt_21_4.sort_values(by=dt_21_4[4],ascending=False) ##  중상자 2021.10 ->18~20시
dt_21_5 = dt_21_5.sort_values(by=dt_21_5[5],ascending=False) ##  경상자 2021.10 ->18~20시
dt_21_6 = dt_21_6.sort_values(by=dt_21_6[6],ascending=False) ##  부상신고자 2021.10 ->18~20시

#%%

dt_21_7 = data_21.loc[:,43:43]
dt_21_8 = data_21.loc[:,44:44]
dt_21_9 = data_21.loc[:,45:45]
dt_21_10 = data_21.loc[:,46:46]
dt_21_11 = data_21.loc[:,47:47]
dt_21_12 = data_21.loc[:,48:48]
dt_21_13 = data_21.loc[:,49:49]
dt_21_14 = data_21.loc[:,50:50]
dt_21_15 = data_21.loc[:,51:51]
dt_21_16 = data_21.loc[:,52:52]
dt_21_17 = data_21.loc[:,53:53]
dt_21_18 = data_21.loc[:,54:54]

#%%

dt_21_7 ## 2021.8(65세~ 합계) 14~16시
dt_21_8 ## 2021.6(65세~ 사망) 10~12시
dt_21_9 ## 2021.8(65세~ 부상) 14~16시
dt_21_10 ## 2021.7(65세~ 중상자) 12~14시
dt_21_11 ## 2021.8(65세~ 경상자) 14~16시
dt_21_12 ## 2021.8(65세~ 부상신고자) 14~16시
dt_21_13 ## 2021.10(연령불명~ 합계) 18~20시
dt_21_14 ## 2021.9(연령불명~ 사망) 16~18시
dt_21_15 ## 2021.10(연령불명~ 부상) 18~20시
dt_21_16 ## 2021.10(연령불명~ 중상자) 18~20시
dt_21_17 ## 2021.10(연령불명~ 경상자) 18~20시
dt_21_18 ## 2021.10(연령불명~ 부상신고자) 18~20시

