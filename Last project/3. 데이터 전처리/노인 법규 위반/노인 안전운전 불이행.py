# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:05:48 2024

@author: soyoung
"""

import pandas as pd

## 2018~2023년 수원 전체 교통사고 현황
df = pd.read_excel(r'D:\Workspace\Python\mini\accidentInfoList_18-23.xlsx')


# 노인 운전자 데이터 기준
old_df = df.loc[df['가해운전자 연령'] >= 65,:]

print(old_df['법규위반'].describe())        
"""
count        3269
unique         11
top       안전운전불이행
freq         1548
Name: 법규위반, dtype: object   """

old_df['법규위반'].unique()
"""
array(['신호위반', '교차로운행방법위반', '기타', '안전운전불이행', '보행자보호의무위반', '안전거리미확보',
       '중앙선침범', '차로위반', '직진우회전진행방해', '불법유턴', '과속'], dtype=object)   """


##노인운전자 사고 중 안전운전 불이행 약 50% 차지
"""
1. 비보호 유턴  2. 빨간불 우회전  3. 고속도로 갓길 주행  4. 민식이법 (스쿨존)  """