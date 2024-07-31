# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:53:52 2024

@author: ksj
"""

import pandas as pd

#%% 2018~2023년 수원 전체 교통사고 현황
df = pd.read_excel('E:/Workspace/!project_team/4.18-23수원교통사고/accidentInfoList_18-23.xlsx')

# 인덱스 데이터 제거
df = df.iloc[:, 2:]

# ECLO, CNT 추가
df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1
df['cnt'] = 1

#%%
df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 28460 entries, 0 to 28459
Data columns (total 30 columns):
 #   Column      Non-Null Count  Dtype         
---  ------      --------------  -----         
 0   사고번호        28460 non-null  int64         
 1   사고일시        28460 non-null  datetime64[ns]
 2   요일          28460 non-null  object        
 3   시군구         28460 non-null  object        
 4   사고내용        28460 non-null  object        
 5   사망자수        28460 non-null  int64         
 6   중상자수        28460 non-null  int64         
 7   경상자수        28460 non-null  int64         
 8   부상신고자수      28460 non-null  int64         
 9   사고유형        28460 non-null  object        
 10  법규위반        28460 non-null  object        
 11  노면상태        28460 non-null  object        
 12  기상상태        28460 non-null  object        
 13  도로형태        28460 non-null  object        
 14  가해운전자 차종    28460 non-null  object        
 15  가해운전자 성별    28460 non-null  object        
 16  가해운전자 연령    28460 non-null  int64         
 17  가해운전자 상해정도  28460 non-null  object        
 18  피해운전자 차종    27614 non-null  object        
 19  피해운전자 성별    27614 non-null  object        
 20  피해운전자 연령    27614 non-null  object        
 21  피해운전자 상해정도  27614 non-null  object        
 22  연           28460 non-null  int64         
 23  월           28460 non-null  int64         
 24  일           28460 non-null  int64         
 25  시간          28460 non-null  int64         
 26  구           28460 non-null  object        
 27  동           28460 non-null  object        
 28  ECLO        28460 non-null  int64         
 29  cnt         28460 non-null  int64         
dtypes: datetime64[ns](1), int64(12), object(17)
memory usage: 6.5+ MB
"""

#%% 데이터 전처리
# 피해운전자 연령 -> object
p_age = r'(\d{1,2})세'  ## 하나 또는 두 자리 숫자로 표현된 나이 일치

df["피해운전자 연령"] = df['피해운전자 연령'].str.extract(p_age)
df["피해운전자 연령"] = df["피해운전자 연령"].apply(pd.to_numeric)

# 가해/피해 운전자 연령 -> 연령대 구분(20미만, 20, 30, 40, 50, 60, 70, 80, 90이상)
df['가해운전자 연령대'] = pd.cut(df['가해운전자 연령'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['20미만', '20대', '30대', '40대', '50대', '60대','70대', '80대', '90이상'], include_lowest=True)
df['피해운전자 연령대'] = pd.cut(df['피해운전자 연령'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['20미만', '20대', '30대', '40대', '50대', '60대','70대', '80대', '90이상'], include_lowest=True)

# 피해운전자 연령대 결측값(nan) -> '미분류'
df["가해운전자 연령대"].isna().sum() # 0
df["피해운전자 연령대"].isna().sum() # 881

import numpy as np
df['피해운전자 연령대'] = np.where(df['피해운전자 연령대'].notna(), df['피해운전자 연령대'], '미분류')
"""
notna() : NaN 값이 아닌 값을 True 반환
>> "피해운전자 연령대"의 값이 NaN이 아니면 원래 값을 그대로 유지하고, NaN이면 '미분류' 문자열 할당
"""

## 사고유형 -> 사고유형, 사고유형 - 세부분류 분리 & 추가
str_pattern = r'(.+) - (.+)'
acc_type = df['사고유형'].str.extract(str_pattern)

df['사고유형 - 세부분류'] = df['사고유형']
df['사고유형'] = acc_type[0]

#%% 평균ECLO(심각도), 총사고건수(발생도) 계산 함수
def cal_fmea(df, col) :
    # col_data = list(df[column].unique())
    
    # 컬럼별 평균 ECLO(col+"_dangerous") 계산
    column_dangerous = df[[col, "ECLO"]].groupby(col).mean()

    # 컬럼별 사고건수(cnt) 계산
    column_count = df[[col, "cnt"]].groupby(col).sum()
    
    # 컬럼별 평균 ECLO + 동별 사고건수 / 데이터프레임 병합
    temp = pd.merge(column_dangerous, column_count, how="left", on=[col])  
    temp.reset_index(inplace = True)
    temp['column'] = col
    temp.columns = ['data', "평균ECLO", '총사고건수', 'column']
    temp = temp.loc[:, ['column', 'data', "평균ECLO", '총사고건수']]
    
    return temp

#%% 컬럼별 평균ECLO, 총사고건수 계산 및 엑셀 저장
df = df.loc[:, ['연', '월', '일', '시간', '요일', '구', '시군구', '사고내용', 
                '사고유형', '사고유형 - 세부분류', '법규위반', '노면상태', '기상상태', 
                '도로형태', '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도', 
                '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도']]

concat_df = cal_fmea(df, '연')

for col in df.columns[1:] :
    temp = cal_fmea(df, col)
    
    concat_df = pd.concat([concat_df, temp])
    
concat_df.to_excel('FMEA_계산.xlsx')
