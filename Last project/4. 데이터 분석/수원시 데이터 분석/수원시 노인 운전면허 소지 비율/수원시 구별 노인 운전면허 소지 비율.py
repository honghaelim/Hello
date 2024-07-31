# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:38:17 2024

@author: ParkBumCheol
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('./202404월말_경기도운전면허소지현황.xlsx', skiprows=3)

df = df.dropna(axis = 0)
df
  
#%%

# '지역별' = '경기 수원 장안, 권선, 영통, 팔달', '나이' >= 65, '성별' == 남, 여

count_over_65_men_장안 = df[(df['시군구'] == '경기 수원 장안') & (df['나이'] >= 65) & (df['성별'] == '남')] ['계'].sum()
count_over_65_women_장안 = df[(df['시군구'] == '경기 수원 장안') & (df['나이'] >= 65) & (df['성별'] == '여')] ['계'].sum()

count_over_65_men_권선 = df[(df['시군구'] == '경기 수원 권선') & (df['나이'] >= 65) & (df['성별'] == '남')] ['계'].sum()
count_over_65_women_권선 = df[(df['시군구'] == '경기 수원 권선') & (df['나이'] >= 65) & (df['성별'] == '여')] ['계'].sum()

count_over_65_men_영통 = df[(df['시군구'] == '경기 수원 영통') & (df['나이'] >= 65) & (df['성별'] == '남')] ['계'].sum()
count_over_65_women_영통 = df[(df['시군구'] == '경기 수원 영통') & (df['나이'] >= 65) & (df['성별'] == '여')] ['계'].sum()

count_over_65_men_팔달 = df[(df['시군구'] == '경기 수원 팔달') & (df['나이'] >= 65) & (df['성별'] == '남')] ['계'].sum()
count_over_65_women_팔달 = df[(df['시군구'] == '경기 수원 팔달') & (df['나이'] >= 65) & (df['성별'] == '여')] ['계'].sum()

print("남자:", count_over_65_men_장안)
print("여자:", count_over_65_women_장안)

print("남자:", count_over_65_men_권선)
print("여자:", count_over_65_women_권선)

print("남자:", count_over_65_men_영통)
print("여자:", count_over_65_women_영통)

print("남자:", count_over_65_men_팔달)
print("여자:", count_over_65_women_팔달)

"""
남자: 13778.0
여자: 6736.0
남자: 17623.0
여자: 8336.0
남자: 12078.0
여자: 7189.0
남자: 10210.0
여자: 4949.0
"""
#%%

# 원 그래프로 65세 이상 남녀 노인면허 소지비율 표현
region_data = {
    '경기 수원 장안': [count_over_65_men_장안, count_over_65_women_장안],
    '경기 수원 권선': [count_over_65_men_권선, count_over_65_women_권선],
    '경기 수원 영통': [count_over_65_men_영통, count_over_65_women_영통],
    '경기 수원 팔달': [count_over_65_men_팔달, count_over_65_women_팔달]
}

for region, data in region_data.items():
    plt.figure(figsize=(5, 5))
    plt.pie(data, labels=['남자', '여자'], autopct="%1.1f%%", startangle=90)
    plt.title(f'{region} 65세 이상 노인 면허 소지 비율')
    if region == '경기 수원 장안':
      plt.text(-0.67, -0.45, f"{data[0]}", fontsize=9)
      plt.text(0.37, 0.17, f"{data[1]}", fontsize=9)
    if region == '경기 수원 권선':
      plt.text(-0.67, -0.45, f"{data[0]}", fontsize=9)
      plt.text(0.37, 0.17, f"{data[1]}", fontsize=9)
    if region == '경기 수원 영통':
      plt.text(-0.7, -0.36, f"{data[0]}", fontsize=9)
      plt.text(0.4, 0.13, f"{data[1]}", fontsize=9)
    if region == '경기 수원 팔달':
      plt.text(-0.67, -0.45, f"{data[0]}", fontsize=9)
      plt.text(0.37, 0.17, f"{data[1]}", fontsize=9)            
    plt.show()
