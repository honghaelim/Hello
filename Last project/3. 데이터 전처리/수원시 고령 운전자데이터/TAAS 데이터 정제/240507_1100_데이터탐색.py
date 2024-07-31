# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:20:02 2024

@author: ksj
"""

import pandas as pd

df = pd.read_csv('./accidentInfoList_TAAS_경기도수원2023_전체.csv', low_memory=False, encoding='CP949')

df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4744 entries, 0 to 4743
Data columns (total 22 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   사고번호        4744 non-null   int64 
 1   사고일시        4744 non-null   object
 2   요일          4744 non-null   object
 3   시군구         4744 non-null   object
 4   사고내용        4744 non-null   object
 5   사망자수        4744 non-null   int64 
 6   중상자수        4744 non-null   int64 
 7   경상자수        4744 non-null   int64 
 8   부상신고자수      4744 non-null   int64 
 9   사고유형        4744 non-null   object
 10  법규위반        4744 non-null   object
 11  노면상태        4744 non-null   object
 12  기상상태        4744 non-null   object
 13  도로형태        4744 non-null   object
 14  가해운전자 차종    4744 non-null   object
 15  가해운전자 성별    4744 non-null   object
 16  가해운전자 연령    4744 non-null   object
 17  가해운전자 상해정도  4744 non-null   object
 18  피해운전자 차종    4599 non-null   object
 19  피해운전자 성별    4599 non-null   object
 20  피해운전자 연령    4599 non-null   object
 21  피해운전자 상해정도  4599 non-null   object
dtypes: int64(5), object(17)
memory usage: 815.5+ KB
"""

df.columns
"""
Index(['사고번호', '사고일시', '요일', '시군구', '사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
       '사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해운전자 차종', '가해운전자 성별',
       '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령',
       '피해운전자 상해정도'],
      dtype='object')
"""

#%%
# 사고일시
df['사고일시'] = pd.to_datetime(df['사고일시'])
## DateParseError: Unknown datetime string format, unable to parse: 2023년 1월 1일 00시, at position 0
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')

print(df['사고일시'].dtype)
## datetime64[ns]

import matplotlib.pyplot as plt
df.plot(x='사고일시', y='사고일시', kind='line')
plt.show()

#%%
# unique()
df['사고내용'].unique()
"""
array(['경상사고', '중상사고', '부상신고사고', '사망사고'], dtype=object)
"""

df['사고유형'].unique()
"""
array(['차대사람 - 기타', '차대차 - 측면충돌', '차대차 - 기타', '차대사람 - 횡단중', '차량단독 - 기타',
       '차대차 - 추돌', '차량단독 - 공작물충돌', '차대사람 - 차도통행중', '차대차 - 정면충돌',
       '차대사람 - 길가장자리구역통행중', '차대차 - 후진중충돌', '차대사람 - 보도통행중',
       '차량단독 - 전도전복 - 전도', '차량단독 - 전도전복 - 전복'], dtype=object)
"""

df['법규위반'].unique()
"""
array(['안전운전불이행', '신호위반', '직진우회전진행방해', '안전거리미확보', '보행자보호의무위반',
       '교차로운행방법위반', '중앙선침범', '차로위반', '기타', '불법유턴'], dtype=object)
"""

df['노면상태'].unique()
"""
array(['건조', '서리/결빙', '젖음/습기', '적설', '기타'], dtype=object)
"""

df['기상상태'].unique()
"""
array(['맑음', '흐림', '비', '기타', '눈'], dtype=object)
"""

df['도로형태'].unique()
"""
array(['단일로 - 기타', '기타 - 기타', '교차로 - 교차로안', '교차로 - 교차로부근',
       '교차로 - 교차로횡단보도내', '단일로 - 고가도로위', '단일로 - 지하차도(도로)내', '단일로 - 교량',
       '주차장 - 주차장', '단일로 - 터널', '미분류 - 미분류'], dtype=object)
"""

df['가해운전자 차종'].unique()
"""
array(['승용', '이륜', '승합', '화물', '특수', '건설기계', '원동기', '기타불명', '개인형이동수단(PM)',
       '자전거', '농기계', '사륜오토바이(ATV)'], dtype=object)
"""

df['피해운전자 차종'].unique()
"""
array(['보행자', '자전거', '이륜', '개인형이동수단(PM)', '승용', nan, '승합', '화물', '특수',
       '원동기', '건설기계', '기타불명'], dtype=object)
"""

df['가해운전자 상해정도'].unique()
"""
array(['상해없음', '중상', '부상신고', '경상', '기타불명', '사망'], dtype=object)
"""

df['피해운전자 상해정도'].unique()
"""
array(['경상', '중상', '부상신고', nan, '상해없음', '기타불명', '사망'], dtype=object)
"""

