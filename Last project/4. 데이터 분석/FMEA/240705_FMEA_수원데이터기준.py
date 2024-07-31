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
df.isna().sum()
"""
사고번호            0
사고일시            0
요일              0
시군구             0
사고내용            0
사망자수            0
중상자수            0
경상자수            0
부상신고자수          0
사고유형            0
법규위반            0
노면상태            0
기상상태            0
도로형태            0
가해운전자 차종        0
가해운전자 성별        0
가해운전자 연령        0
가해운전자 상해정도      0
피해운전자 차종      846
피해운전자 성별      846
피해운전자 연령      846
피해운전자 상해정도    846
연               0
월               0
일               0
시간              0
구               0
동               0
ECLO            0
cnt             0
dtype: int64
"""

#%% 피해운전자 결측값
## 피해운전자 연령대 결측값(nan) -> '피해자없음'
import numpy as np
df['피해운전자 차종'] = np.where(df['피해운전자 차종'].notna(), df['피해운전자 차종'], '피해자없음')
df['피해운전자 성별'] = np.where(df['피해운전자 성별'].notna(), df['피해운전자 성별'], '피해자없음')
df['피해운전자 연령'] = np.where(df['피해운전자 연령'].notna(), df['피해운전자 연령'], '피해자없음')
df['피해운전자 상해정도'] = np.where(df['피해운전자 상해정도'].notna(), df['피해운전자 상해정도'], '피해자없음')
"""
notna() : NaN 값이 아닌 값을 True 반환
>> "피해운전자"의 값이 NaN이 아니면 원래 값을 그대로 유지하고, NaN이면 '피해자없음' 문자열 할당
"""

#%% 데이터 전처리
## 사고유형 -> 사고유형, 사고유형 - 세부분류 분리 & 추가
str_pattern = r'(.+) - (.+)'
acc_type = df['사고유형'].str.extract(str_pattern)

df['사고유형 - 세부분류'] = df['사고유형']
df['사고유형'] = acc_type[0]

df['사고유형'].unique()
## array(['차대차', '차대사람', '차량단독', '차량단독 - 전도전복', '차량단독 - 도로외이탈'], dtype=object)

df["사고유형"] = df["사고유형"].replace('차량단독 - 전도전복', '차량단독')
df["사고유형"] = df["사고유형"].replace('차량단독 - 도로외이탈', '차량단독')

#%%
# 가해 운전자 연령 -> 연령대 구분(미분류(0), 20미만, 20, 30, 40, 50, 60, 70, 80, 90이상)
df["가해운전자 연령"].describe() # min 0 ~ max 96
df['가해운전자 연령대'] = pd.cut(df['가해운전자 연령'], bins=[0, 1, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['미분류', '20미만', '20대', '30대', '40대', '50대', '60대','70대', '80대', '90이상'], include_lowest=True)

#%%
# 피해 운전자 연령 -> 연령대 구분(피해자없음, 미분류, 20미만, 20, 30, 40, 50, 60, 70, 80, 90이상)
def age_category(age):
  if age == '피해자없음':
    return '피해자없음'
  elif age == '미분류':
    return '미분류'
  else:
    # 나이 추출
    age_num = int(age[:-1])  # '세' 제외하고 숫자만 추출
    # 10대 단위로 나이 그룹 설정
    if age_num < 20:
      return '20미만'
    elif age_num < 30:
      return '20대'
    elif age_num < 40:
      return '30대'
    elif age_num < 50:
      return '40대'
    elif age_num < 60:
      return '50대'
    elif age_num < 70:
      return '60대'
    elif age_num < 80:
      return '70대'
    elif age_num < 90:
      return '80대'
    else:
      return '90이상'
    
# 피해운전자 연령 -> object
df["피해운전자 연령"].unique()
"""
array(['25세', '30세', '71세', '20세', '73세', '40세', '58세', '50세', '41세',
       '52세', '83세', '7세', '48세', '46세', '17세', '65세', '33세', '78세',
       '56세', '23세', '53세', '60세', '61세', '27세', '24세', '18세', '54세',
       '32세', '51세', '37세', '49세', '57세', '80세', '26세', '29세', '36세',
       '31세', '70세', '42세', '피해자없음', '28세', '19세', '59세', '55세', '68세',
       '43세', '64세', '45세', '63세', '34세', '47세', '39세', '87세', '35세',
       '74세', '62세', '4세', '75세', '21세', '22세', '15세', '85세', '66세',
       '38세', '44세', '77세', '81세', '88세', '72세', '76세', '79세', '67세',
       '13세', '16세', '10세', '14세', '69세', '8세', '미분류', '11세', '86세',
       '82세', '90세 이상', '9세', '6세', '84세', '5세', '12세', '1세', '3세', '89세',
       '2세', '95세', '94세', '91세', '98세 이상', '90세', '92세', '97세', '93세'],
      dtype=object)
"""
df["피해운전자 연령"] = df["피해운전자 연령"].replace('90세 이상', '90세')
df["피해운전자 연령"] = df["피해운전자 연령"].replace('98세 이상', '98세')

for i in range(len(df)) :
    df.loc[i, "피해운전자 연령대"] = age_category(df.loc[i, "피해운전자 연령"])

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
df_copy = df.loc[:, ['연', '월', '일', '시간', '요일', '구', '시군구', '사고내용', 
                    '사고유형', '사고유형 - 세부분류', '법규위반', '노면상태', '기상상태', 
                    '도로형태', '가해운전자 차종', '가해운전자 성별', '가해운전자 연령대', '가해운전자 상해정도', 
                    '피해운전자 차종', '피해운전자 성별', '피해운전자 연령대', '피해운전자 상해정도',
                    'ECLO', 'cnt']]

concat_df = cal_fmea(df_copy, df_copy.columns[0])

for col in df_copy.columns[1:-2] :
    temp = cal_fmea(df_copy, col)
    
    concat_df = pd.concat([concat_df, temp])
    

#%%
tp = concat_df.copy()

## 분산(var)이란 변수의 흩어진 정도를 계산하는 지표
# 기댓값이 확률변수에서 어떤 값이 나올지를 예측한 것이라면 분산은 그 예측의 정확도 혹은 신뢰성을 표현한 것
# 분산이 크면 예측의 정확도가 떨어진다(기댓값 = 총사고건수)
tp_var = tp[['column', '총사고건수']].groupby('column').var()

tp_var.reset_index(inplace=True)
tp_var.columns = ['column', '분산_사고건수']

concat_df = pd.merge(concat_df, tp_var, on='column')

#%%
concat_df.to_excel('240705_FMEA_계산.xlsx')

#%%
#%%
tp = concat_df.copy()

## 표준편차 : 분산의 제곱근
# 데이터의 분산이나 표준편차가 높을수록 데이터의 변동성이 크고, 
# 따라서 미래 값을 예측하기 어려울 가능성이 높다는 점에서 간접적인 관련성이 있다고 볼 수 있다
tp_std = tp[['column', '총사고건수']].groupby('column').std()

tp_std.reset_index(inplace=True)
tp_std.columns = ['column', '표준편차_사고건수']

concat_df = pd.merge(concat_df, tp_std, on='column')

#%%
concat_df.to_excel('240705_FMEA_계산.xlsx')