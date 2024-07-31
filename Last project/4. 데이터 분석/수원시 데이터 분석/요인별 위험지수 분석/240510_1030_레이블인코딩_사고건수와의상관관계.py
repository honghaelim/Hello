# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:13:12 2024

@author: ksj
"""

import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

df = pd.read_excel('./The elderly driver traffic accidents(suwon).xlsx')

#%% 전처리
# [사고일시] -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')  ## 2023-01-01 00:00:00
#   날짜(object)                   
df['날짜'] = df['사고일시'].dt.date                                            ## 2023-01-01
#   연(int)
df['연'] = df['사고일시'].dt.year                                              ## 2023
#   월(int)
df['월'] = df['사고일시'].dt.month                                             ## 1
#   일(int)
df['일'] = df['사고일시'].dt.day                                               ## 1
#   시간(int)
df['시간'] = df['사고일시'].dt.hour                                            ## 0
# 시간대 -> 주간/야간
""" 기준
주간: 오전 7시부터 오후 8시까지 (13시간)
야간: 오후 8시부터 다음 날 오전 7시까지 (11시간)
"""
df['주야간'] = df['시간'].apply(lambda x: '주간' if 7 <= x <= 20 else '야간')

# [시군구] -> 구/ 동
gu = []
dong = []
for i in range(len(df)) :
    gu.append(df['시군구'].str.split(' ')[i][2])
    dong.append(df['시군구'].str.split(' ')[i][3])
df['구'] = gu 
df['동'] = dong

# [사고유형] '차대사람 - 기타' -> '차대사람', '기타'
dep1 = []
dep2 = []
for i in range(len(df)) :
    dep1.append(df['사고유형'].str.split(' - ')[i][0])
    dep2.append(df['사고유형'].str.split(' - ')[i][1])
df['사고유형1'] = dep1
df['사고유형2'] = dep2

# [도로형태] '단일로 - 기타' -> '단일로', '기타'
dep1 = []
dep2 = []
for i in range(len(df)) :
    dep1.append(df['도로형태'].str.split(' - ')[i][0])
    dep2.append(df['도로형태'].str.split(' - ')[i][1])
df['도로형태1'] = dep1
df['도로형태2'] = dep2

# [피해운전자] nan -> 0
""" df.iloc[:, 18:22].columns 
Index(['피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도'], dtype='object')
"""
df.iloc[:, 18:22] = df.iloc[:, 18:22].fillna(0)

# [연령] 00세(object) -> 00(int)
# '가해운전자'
df['가해운전자 연령'] = df['가해운전자 연령'].str[:-1]
# int 변환
df['가해운전자 연령'] = df['가해운전자 연령'].astype('int64')
#
# '피해운전자'
df['피해운전자 연령'] = df['피해운전자 연령'].str[:-1]
## -> nan(0->nan), '미분'('미분류') 존재
#       -> '미분류' : 0
df['피해운전자 연령'] = df['피해운전자 연령'].replace('미분', 0)
#       -> nan : 0
df['피해운전자 연령'] = df['피해운전자 연령'].fillna(0)
# int 변환
df['피해운전자 연령'] = df['피해운전자 연령'].astype('int64')

#%%
df.columns
"""
Index(['사고번호', '사고일시', '요일', '시군구', '사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
       '사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해운전자 차종', '가해운전자 성별',
       '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령',
       '피해운전자 상해정도', '날짜', '연', '월', '일', '시간', '구', '동', '사고유형1', '사고유형2',
       '도로형태1', '도로형태2', '주야간'],
      dtype='object')
"""

df_table = df.loc[:, ['날짜', '연', '월', '일', '요일', '시간', '주야간',
                      '구', '동', '노면상태', '기상상태', '도로형태1', '도로형태2', 
                      '법규위반', '사고유형1', '사고유형2', '사고내용',
                      '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도', 
                      '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도',
                      '사망자수', '중상자수', '경상자수', '부상신고자수'
                      ]]

df_table['사고건수'] = 1

#%% ECLO 계산 함수
def cal_eclo(df) :
    df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1
    df['ECLO/사고건수'] = df['ECLO']/df['사고건수']
    return df

#%%
accident_cnt = df_table.groupby(['주야간', '구', '노면상태', '기상상태', '도로형태1', 
                      '가해운전자 차종', '가해운전자 성별'])[['사고건수', '사망자수', '중상자수', '경상자수', '부상신고자수']].sum()
accident_cnt = accident_cnt.reset_index()
cal_eclo(accident_cnt) 
print(accident_cnt)
"""
    주야간    구   노면상태 기상상태 도로형태1  ... 중상자수 경상자수  부상신고자수  ECLO  ECLO/사고건수
0    야간  권선구     건조   기타   교차로  ...    0    3       0     9     4.5000
1    야간  권선구     건조   맑음   교차로  ...    0    0       1     1     1.0000
2    야간  권선구     건조   맑음   교차로  ...    5   33       0   134     4.1875
3    야간  권선구     건조   맑음   교차로  ...    1    3       0    14     7.0000
4    야간  권선구     건조   맑음   교차로  ...    3    5       0    30     3.7500
..   ..  ...    ...  ...   ...  ...  ...  ...     ...   ...        ...
285  주간  팔달구  젖음/습기   흐림   교차로  ...    0    3       0     9     4.5000
286  주간  팔달구  젖음/습기   흐림   교차로  ...    0    1       0     3     3.0000
287  주간  팔달구  젖음/습기   흐림    기타  ...    0    1       0     3     3.0000
288  주간  팔달구  젖음/습기   흐림   단일로  ...    0    2       0     6     3.0000
289  주간  팔달구  젖음/습기   흐림   단일로  ...    0    1       0     3     3.0000

[290 rows x 14 columns]
"""

accident_cnt.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 290 entries, 0 to 289
Data columns (total 14 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   주야간        290 non-null    object 
 1   구          290 non-null    object 
 2   노면상태       290 non-null    object 
 3   기상상태       290 non-null    object 
 4   도로형태1      290 non-null    object 
 5   가해운전자 차종   290 non-null    object 
 6   가해운전자 성별   290 non-null    object 
 7   사고건수       290 non-null    int64  
 8   사망자수       290 non-null    int64  
 9   중상자수       290 non-null    int64  
 10  경상자수       290 non-null    int64  
 11  부상신고자수     290 non-null    int64  
 12  ECLO       290 non-null    int64  
 13  ECLO/사고건수  290 non-null    float64
dtypes: float64(1), int64(6), object(7)
memory usage: 31.8+ KB
"""

#%%
###############################################################################
# 요인 - ECLO 상관계수
# 레이블 인코딩
#   - 문자열(object) 데이터 : '주야간', '구', '노면상태', '기상상태', '도로형태1', '가해운전자 차종', '가해운전자 성별'
###############################################################################
accident_label = accident_cnt.copy()

from sklearn.preprocessing import LabelEncoder
for col in ['주야간', '구', '노면상태', '기상상태', '도로형태1', '가해운전자 차종', '가해운전자 성별'] :
    encoder = LabelEncoder()
    accident_label[col] = encoder.fit_transform(accident_label[col])

accident_label.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 290 entries, 0 to 289
Data columns (total 14 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   주야간        290 non-null    int32  
 1   구          290 non-null    int32  
 2   노면상태       290 non-null    int32  
 3   기상상태       290 non-null    int32  
 4   도로형태1      290 non-null    int32  
 5   가해운전자 차종   290 non-null    int32  
 6   가해운전자 성별   290 non-null    int32  
 7   사고건수       290 non-null    int64  
 8   사망자수       290 non-null    int64  
 9   중상자수       290 non-null    int64  
 10  경상자수       290 non-null    int64  
 11  부상신고자수     290 non-null    int64  
 12  ECLO       290 non-null    int64  
 13  ECLO/사고건수  290 non-null    float64
dtypes: float64(1), int32(7), int64(6)
memory usage: 23.9 KB
"""

import seaborn as sns
corr_table = accident_label.corr()
sns.heatmap(corr_table, annot = True, cmap='YlGnBu', linewidth=.5, cbar=False)
plt.show()
