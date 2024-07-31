# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:10:02 2024

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
age_accidents = df.groupby('가해운전자 연령')['사고번호'].count().reset_index()
age_accidents.columns = ['연령', '사고건수']

plt.bar(age_accidents['연령'], age_accidents['사고건수'])
plt.title("가해운전자 연령에 따른 사고건수")
plt.xlabel("가해운전자 연령")
plt.ylabel("사고건수")
plt.show()

#%%
sex_accidents = df.groupby('가해운전자 성별')['사고번호'].count().reset_index()
sex_accidents.columns = ['성별', '사고건수']

plt.bar(sex_accidents['성별'], sex_accidents['사고건수'])
plt.title("가해운전자 성별에 따른 사고건수")
plt.xlabel("가해운전자 성별")
plt.ylabel("사고건수")
plt.show()

#%%
age_sex_accidents = df.groupby(['가해운전자 연령','가해운전자 성별'])['사고번호'].count().reset_index()
age_sex_accidents.columns = ['연령', '성별', '사고건수']
plt.bar(age_sex_accidents[age_sex_accidents['성별']=='남']['연령'], age_sex_accidents[age_sex_accidents['성별']=='남']['사고건수'], label = '남')
plt.bar(age_sex_accidents[age_sex_accidents['성별']=='여']['연령'], age_sex_accidents[age_sex_accidents['성별']=='여']['사고건수'], label = '여')
plt.title("가해운전자 연령에 따른 사고건수")
plt.xlabel("가해운전자 연령")
plt.ylabel("사고건수")
plt.legend(loc='best')
plt.show()

#%%

#%%
"""
성별/연령별 운전자 비율 <- 운전면허 발급 현황 데이터?
"""
