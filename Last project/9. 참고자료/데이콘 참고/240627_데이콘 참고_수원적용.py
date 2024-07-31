# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:39:12 2024

@author: ksj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

#%% 2018~2023년 수원 전체 교통사고 현황
df = pd.read_excel('E:/Workspace/!project_team/4.18-23수원교통사고/accidentInfoList_18-23.xlsx')

# 인덱스 데이터 제거
df = df.iloc[:, 2:]

df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1

len(df)
#%% 
df.columns
"""
Index(['사고번호', '사고일시', '요일', '시군구', '사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
       '사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해운전자 차종', '가해운전자 성별',
       '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령',
       '피해운전자 상해정도', '연', '월', '일', '시간', '구', '동', 'ECLO'],
      dtype='object')
"""

#%%
from pandas import Timestamp
from workalendar.asia import SouthKorea

# int(year, month, day) 입력해서 공휴일인지 판단
# 공휴일이면 True
def is_holiday(year, month, day):
    cal = SouthKorea()
    return cal.is_holiday(Timestamp(year, month, day))

# int(year, month, day) 입력해서 공휴일&주말인지 판단
# 공휴일, 주말이면 1, 아니면 0
def classify_day(year, month, day):
    date = Timestamp(year, month, day)
    if date.dayofweek < 5 and not is_holiday(year, month, day):
        return 0
    else:
        return 1
    
# == 공휴일 : 추가 ==
df['Holiday'] = df.apply(lambda row: classify_day(row['연'], row['월'], row['일']), axis=1)

#%%
# == 도로형태 ==
## -> 도로형태1, 도로형태2 분리 & 추가
road_pattern = r'(.+) - (.+)'
df[['도로형태1', '도로형태2']] = df['도로형태'].str.extract(road_pattern)

#%%
# == 연령 ==
## -> 수치화
p_age = r'(\d{1,2})세'  ## 하나 또는 두 자리 숫자로 표현된 나이 일치

df["피해운전자 연령"] = df['피해운전자 연령'].str.extract(p_age)
df["피해운전자 연령"] = df["피해운전자 연령"].apply(pd.to_numeric)

#%%
df['사고유형'].unique()
"""
array(['차대차 - 기타', '차대사람 - 기타', '차대차 - 추돌', '차대차 - 측면충돌', '차대사람 - 횡단중',
       '차대사람 - 길가장자리구역통행중', '차대차 - 정면충돌', '차대사람 - 보도통행중', '차대사람 - 차도통행중',
       '차량단독 - 기타', '차량단독 - 공작물충돌', '차대차 - 후진중충돌', '차량단독 - 전도전복 - 전도',
       '차량단독 - 전도전복 - 전복', '차량단독 - 도로외이탈 - 기타', '차량단독 - 도로외이탈 - 추락',
       '차량단독 - 주/정차차량 충돌'], dtype=object)
"""

# [사고유형] '차대사람 - 기타' -> '차대사람', '기타'
dep1 = []
dep2 = []
for i in range(len(df)) :
    dep1.append(df['사고유형'].str.split(' - ')[i][0])
    dep2.append(df['사고유형'].str.split(' - ')[i][1])
df['사고유형'] = dep1
df['사고유형 - 세부분류'] = dep2

#%% 
df['사고유형'].unique()
"""
array(['차대차', '차대사람', '차량단독'], dtype=object)
"""

df.to_excel('사고유형분리_18-23.xlsx')

#%%
# == 차대차, 차대사람의 데이터 ==
# 차량단독이 아닌 경우에 대해서는 결측치 모두 drop
not_car = df[df['사고유형'] != '차량단독'].copy()
not_car.reset_index(inplace=True, drop=True)

not_car.isna().sum()
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
피해운전자 차종        0
피해운전자 성별        0
피해운전자 연령       27
피해운전자 상해정도      0
연               0
월               0
일               0
시간              0
구               0
동               0
ECLO            0
Holiday         0
도로형태1           0
도로형태2           0
사고유형 - 세부분류     0
dtype: int64
"""
"""
not_car[not_car['피해운전자 연령'].isna()].to_excel("차량단독이 아닌 경우 결측치.xlsx")
#>> 피해운전자 정보 "기타 불명"
"""

not_car = not_car.dropna()
not_car.reset_index(inplace=True, drop=True)

# == 차량단독의 데이터 ==
# 차량단독에서 ('노면상태', '가해운전자 연령', '시군구', '구', '동') column에 대한 결측치 drop
car = df[df['사고유형'] == '차량단독'].copy()
car.reset_index(inplace=True, drop=True)

car.isna().sum()
"""
사고번호             0
사고일시             0
요일               0
시군구              0
사고내용             0
사망자수             0
중상자수             0
경상자수             0
부상신고자수           0
사고유형             0
법규위반             0
노면상태             0
기상상태             0
도로형태             0
가해운전자 차종         0
가해운전자 성별         0
가해운전자 연령         0
가해운전자 상해정도       0
피해운전자 차종       846
피해운전자 성별       846
피해운전자 연령       854
피해운전자 상해정도     846
연                0
월                0
일                0
시간               0
구                0
동                0
ECLO             0
Holiday          0
도로형태1            0
도로형태2            0
사고유형 - 세부분류      0
dtype: int64
"""
#>> ('노면상태', '가해운전자 연령', '시군구', '구', '동') 결측치 없음

# 차량단독의 경우 피해운전자 차종은 미분류나 결측치임. 따라서 없음으로 대치
car['피해운전자 차종'].unique()
## array([nan, '미분류'], dtype=object)
car['피해운전자 차종'] = '없음'

# 차량단독과 차량단독이 아닌 경우를 합쳐서 전체 데이터 형성
df_all = pd.concat([car, not_car], axis=0)
df_all.reset_index(inplace=True, drop=True)

#%%
df_all.isna().sum()
"""
사고번호             0
사고일시             0
요일               0
시군구              0
사고내용             0
사망자수             0
중상자수             0
경상자수             0
부상신고자수           0
사고유형             0
법규위반             0
노면상태             0
기상상태             0
도로형태             0
가해운전자 차종         0
가해운전자 성별         0
가해운전자 연령         0
가해운전자 상해정도       0
피해운전자 차종         0
피해운전자 성별       846
피해운전자 연령       854
피해운전자 상해정도     846
연                0
월                0
일                0
시간               0
구                0
동                0
ECLO             0
Holiday          0
도로형태1            0
도로형태2            0
사고유형 - 세부분류      0
dtype: int64
"""

# 2023년 데이터와 그외 데이터 분리
train_df = df.loc[df['연']!=2023, :]
test_df = df.loc[df['연']==2023, :]

# 사고유형에 따라 데이터 분리
train1=train_df[train_df["사고유형"]=="차량단독"]
train2=train_df[train_df["사고유형"]=="차대차"]
train3=train_df[train_df["사고유형"]=="차대사람"]

test1=test_df[test_df["사고유형"]=="차량단독"]
test2=test_df[test_df["사고유형"]=="차대차"]
test3=test_df[test_df["사고유형"]=="차대사람"]

print(train1.shape)
print(train2.shape)
print(train3.shape)
"""
(709, 33)
(18320, 33)
(4687, 33)
"""

print(test1.shape)
print(test2.shape)
print(test3.shape)
"""
(145, 33)       # 145/709    = 0.20451339915373765
(3710, 33)      # 3710/18320 = 0.20251091703056767
(889, 33)       # 889/4687   = 0.18967356518028589
>> 2018~2022년까지의 데이터 대비 2023년의 데이터가 약 20% 비율로 적정하다고 판단됨
"""

#%%
# 컬럼 별 위험도(ECLO)를 측정 후 "동" 별로 컬럼별 발생 비율을 반영한 가중치 계산
def danger(df, column):
    col = str(column)
    
    # 컬럼별 평균 ECLO(col+"_dangerous") 계산
    column_dangerous = df[[col, "ECLO"]].groupby(col).mean()
    column_dangerous.columns = [col+"_dangerous"]
    
    # 컬럼별 동별 사고건수(cnt) 계산
    column_count = df[["구", "동", col]]
    column_count["cnt"] = 1
    column_count = column_count.groupby(["구", "동", col]).count()
    column_count.reset_index(inplace=True)
    
    # 컬럼별 평균 ECLO + 동별 사고건수 / 데이터프레임 병합
    temp = pd.merge(column_count, column_dangerous, how="left", on=[col])    
    # multiply 계산 > 사고건수 * 평균 ECLO
    temp['multiply'] = temp['cnt'] * temp[col+'_dangerous']    
    # 동별 그룹 > multiply, cnt의 sum 계산/ col+'_dangerous' drop
    temp = temp.groupby(["구", "동"]).sum().reset_index().drop([col+"_dangerous"], axis=1)    
    # col+'_dangerous' 재계산 > (사고건수*평균ECLO)의 합 / 사고건수 합 
    temp[col+'_dangerous'] = temp['multiply'] / temp['cnt']
    # multiply, cnt drop
    temp.drop([column,'multiply','cnt'], axis=1, inplace=True)
    
    # 동별 col+"_dangerous"[(동별 컬럼 사고건수* 컬럼 평균ECLO)의 합 / 동별 사고건수] return
    return temp

# 특정 백분위수 계산
def set_limit(column):
    # np.quantile 함수 : NumPy에서 제공하는 함수, 데이터 배열의 특정 백분위수(quantile) 계산
    return np.quantile(column, 0.999)

# 남 1, 여 0, 그외 nan
def sex_transform(x):
    if x=='남':
        return 1
    elif x=='여':
        return 0
    else:
        return np.NaN
    
# 65세 이상 1, 그외 0    
def sepa(x):
    if x>=65 :
        return 1
    else :
        return 0
    
# == 이상치 1% 제거 ==
def del_ECLO(df) :
    sns.boxplot(x="ECLO", data=df)
    plt.show()

    df_copy = df.copy()

    df_copy = df_copy[['노면상태', '사고유형', '사고유형 - 세부분류', '도로형태', 'ECLO']]
    df_copy.reset_index(inplace=True, drop=True)
    df_copy.index.to_list()

    outlier_idxs = []
    for col in df_copy.columns[:-1]: # ECLO 제외 컬럼명 순차 추출
        # 그룹별 ECLO 0.999인 백분위수
        temp_df = pd.DataFrame(df_copy.groupby(col)['ECLO'].agg(set_limit)).reset_index()
        
        for j in range(len(temp_df)):
            # (컬럼 == col) & (ECLO > 0.999인 백분위수(이상치 1%))의 인덱스 리스트 추출
            s_idxs = df[(df[col] == temp_df.loc[j, col]) & (df['ECLO'] > temp_df.loc[j, 'ECLO'])].index.to_list()
            outlier_idxs = outlier_idxs + s_idxs
            
    # set로 변환해서 중복값 제거
    outliers = list(set(outlier_idxs))
    
    print('outlier 수 : ', len(outliers))      ## 10
    
    df = df.drop(outliers, axis=0)
    df.reset_index(inplace=True, drop=True)
    
    print("train_df 데이터 수 : ", len(df))  ## 699
    
    # 이상치 제거 후 그래프 확인
    sns.boxplot(x="ECLO", data=df)
    plt.show()
    
    return df

#%%
train1 = del_ECLO(train1)
"""
outlier 수 :  10
train_df 데이터 수 :  699
"""
train2 = del_ECLO(train2)
"""
outlier 수 :  33
train_df 데이터 수 :  18287
"""
train3 = del_ECLO(train3)
"""
outlier 수 :  10
train_df 데이터 수 :  4677
"""

print(train1.shape)
print(train2.shape)
print(train3.shape)
"""
(699, 33)
(18287, 33)
(4677, 33)
"""
    
#%%
# 차량 단독 df(train1), test1
def feat_eng1(df) :
    global test1    
    
    # == 사고유형 -> drop ==
    # 사고유형 == 차량단독
    df = df.drop(columns=["사고유형"])
    
    test_eng = test1.copy()    
    test_eng = test_eng.drop(columns=["사고유형"])

    ##############################################################################################
    # == 지역(동)별 가해운전자 평균 연령 추출 ==
    age_mean = df[['구','동','가해운전자 연령']].groupby(['구','동']).mean()
    age_mean.columns = ['가해운전자 평균연령']
    
    df = pd.merge(df, age_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 연령 추가 : 전체 평균으로 nan에 값을 채움
    # == 사고유형 -> drop ==
    test_eng = pd.merge(test_eng, age_mean, how="left", on=["구", "동"])
    test_eng[['가해운전자 평균연령']] = test_eng[['가해운전자 평균연령']].fillna(test_eng[['가해운전자 평균연령']].mean())

    ##############################################################################################
    # == 지역별 가해운전자 평균 성별 추출
    # 가해운전자 성별 : 남 1, 여 0, else nan
    df['가해운전자 성별'] = df['가해운전자 성별'].apply(lambda x:sex_transform(x))
    
    sex_mean = df[['구','동','가해운전자 성별']].groupby(['구','동']).mean()    
    sex_mean.columns = ['가해운전자 평균성별']
    
    df = pd.merge(df, sex_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 성별 추가 : 전체 평균으로 nan에 값을 채움
    test_eng = pd.merge(test_eng, sex_mean, how="left", on=["구", "동"])
    test_eng[['가해운전자 평균성별']] = test_eng[['가해운전자 평균성별']].fillna(test_eng[['가해운전자 평균성별']].mean())
    
    ##############################################################################################
    # == 가해노인운전자 위험도 ==
    # 가해운전자 연령 : 65세 이상 1, 그외 0
    df["가해운전자 연령"] = df["가해운전자 연령"].apply(sepa)
    
    old_count = df[["구", "동", "가해운전자 연령"]]    
    # cnt 변수
    old_count["cnt"] = 1    
    # 동별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령=1) 합계 
    old_count = old_count.groupby(["구", "동"]).sum().reset_index()    
    # 노인 사고 / 전체 사고
    old_count["ratio"] = old_count["가해운전자 연령"] / old_count["cnt"]
    
    # 동별 노인 사고 평균 ECLO 
    old_eclo = df[df["가해운전자 연령"]==1][["구", "동", "ECLO"]]
    old_eclo = old_eclo.groupby(["구", "동"]).mean().reset_index()
    
    # 동별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령), 노인 평균 ECLO(ECLO) 병합
    temp = pd.merge(old_count, old_eclo, how="left", on=["구", "동"])
    # nan -> 0
    temp.fillna(0, inplace=True)
    
    # (노인 사고건수 / 전체 사고건수) * 노인 사고 평균 ECLO 
    temp["가해노인운전자 위험도"] = temp["ratio"] * temp["ECLO"]
    
    temp.drop(["가해운전자 연령", "cnt", "ratio", "ECLO"], axis=1, inplace=True)

    df = pd.merge(df, temp, how="left", on=["구", "동"])
    test_eng = pd.merge(test_eng, temp, how="left", on=["구", "동"])

    ##############################################################################################
    # == 가해운전자 차종 별 위험도 ==
    # danger : 컬럼 별 위험도(ECLO)를 측정 후 "동" 별로 컬럼별 발생 비율을 반영한 가중치 계산
    danger1 = danger(df, "가해운전자 차종")

    df = pd.merge(df, danger1, how="left", on=["구", "동"])
    test_eng = pd.merge(test_eng, danger1, how="left", on=["구", "동"])

    ##############################################################################################
    # == 칼럼 동기화 ==
    ytest = test_eng['ECLO']
    test_eng = test_eng[['요일', '도로형태', '노면상태', '연', '월', '일', '시간', 'Holiday', '구', '동', '도로형태1',
                         '가해운전자 평균연령', '가해운전자 평균성별', "가해노인운전자 위험도", "가해운전자 차종_dangerous"]]
    
    ytrain = df["ECLO"]    
    df = df[test_eng.columns]

    ##############################################################################################
    # == 원핫 인코딩 ==
    ## '노면상태',"도로형태1"
    one_hot_features=['노면상태',"도로형태1"]

    train_oh = pd.get_dummies(df[one_hot_features])
    test_oh = pd.get_dummies(test_eng[one_hot_features])

    for i in train_oh.columns:
        if i not in test_oh.columns:
            test_oh[i] = 0
    for i in test_oh.columns:
        if i not in train_oh.columns:
            train_oh[i] = 0

    print("[원핫인코딩] test_oh 컬럼수 : ", len(test_oh.columns), "/ train_oh 컬럼수 : ", len(train_oh.columns))

    # 원데이터 드롭
    df.drop(one_hot_features,axis=1,inplace=True)
    test_eng.drop(one_hot_features,axis=1,inplace=True)

    # 원핫인코딩 결과 병합
    df = pd.concat([df,train_oh],axis=1)
    test_eng = pd.concat([test_eng,test_oh],axis=1)

    ##############################################################################################
    # == 일 드롭 ==
    df = df.drop(columns=["일"])
    
    ##############################################################################################
    # == 레이블 인코딩
    ## "요일", "구", "동", "도로형태"
    label_features = ["요일", "구", "동", "도로형태"]

    for i in label_features:
        print("[레이블인코딩] ", i)
        le = LabelEncoder()
        le = le.fit(df[i])
        df[i] = le.transform(df[i])

        for case in np.unique(test_eng[i]):
            if case not in le.classes_:
                print(case, ' : test case is not in classes')
                le.classes_ = np.append(le.classes_, case)
        test_eng[i] = le.transform(test_eng[i])

    # == 타겟 인코딩
    return df, ytrain, test_eng, ytest

#%%
train1.columns
"""
Index(['사고번호', '사고일시', '요일', '시군구', '사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
       '사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해운전자 차종', '가해운전자 성별',
       '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령',
       '피해운전자 상해정도', '연', '월', '일', '시간', '구', '동', 'ECLO', 'Holiday', '도로형태1',
       '도로형태2', '사고유형 - 세부분류'],
      dtype='object')
"""
test1.columns
"""
Index(['사고번호', '사고일시', '요일', '시군구', '사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
       '사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해운전자 차종', '가해운전자 성별',
       '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령',
       '피해운전자 상해정도', '연', '월', '일', '시간', '구', '동', 'ECLO', 'Holiday', '도로형태1',
       '도로형태2', '사고유형 - 세부분류'],
      dtype='object')
"""

X_train_eng1, y_train1, X_test_eng1, y_test_eng1 = feat_eng1(train1)
"""
[원핫인코딩] test_oh 컬럼수 :  9 / train_oh 컬럼수 :  9
[레이블인코딩]  요일
[레이블인코딩]  구
[레이블인코딩]  동
상광교동  : test case is not in classes
[레이블인코딩]  도로형태
"""

#%%
# 차대차 df(train2), test2
def feat_eng2(df) :
    global test2    
    
    # == 사고유형 -> drop ==
    # 사고유형 == 차대차
    df = df.drop(columns=["사고유형"])
    
    test_eng = test2.copy()    
    test_eng = test_eng.drop(columns=["사고유형"])

    ##############################################################################################
    # == 지역(동)별 가해운전자 평균 연령 추출 ==
    age_mean = df[['구','동','가해운전자 연령']].groupby(['구','동']).mean()
    age_mean.columns = ['가해운전자 평균연령']
    
    df = pd.merge(df, age_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 연령 추가 : 전체 평균으로 nan에 값을 채움
    # == 사고유형 -> drop ==
    test_eng = pd.merge(test_eng, age_mean, how="left", on=["구", "동"])
    test_eng[['가해운전자 평균연령']] = test_eng[['가해운전자 평균연령']].fillna(test_eng[['가해운전자 평균연령']].mean())
    
    # == 지역(동)별 피해운전자 평균 연령 추출 ==
    age_mean = df[['구','동','피해운전자 연령']].groupby(['구','동']).mean()
    age_mean.columns = ['피해운전자 평균연령']
    
    df = pd.merge(df, age_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 연령 추가 : 전체 평균으로 nan에 값을 채움
    # == 사고유형 -> drop ==
    test_eng = pd.merge(test_eng, age_mean, how="left", on=["구", "동"])
    test_eng[['피해운전자 평균연령']] = test_eng[['피해운전자 평균연령']].fillna(test_eng[['피해운전자 평균연령']].mean())

    ##############################################################################################
    # == 지역별 가해운전자 평균 성별 추출
    # 가해운전자 성별 : 남 1, 여 0, else nan
    df['가해운전자 성별'] = df['가해운전자 성별'].apply(lambda x:sex_transform(x))
    
    sex_mean = df[['구','동','가해운전자 성별']].groupby(['구','동']).mean()    
    sex_mean.columns = ['가해운전자 평균성별']
    
    df = pd.merge(df, sex_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 성별 추가 : 전체 평균으로 nan에 값을 채움
    test_eng = pd.merge(test_eng, sex_mean, how="left", on=["구", "동"])
    test_eng[['가해운전자 평균성별']] = test_eng[['가해운전자 평균성별']].fillna(test_eng[['가해운전자 평균성별']].mean())
    
    # == 지역별 피해운전자 평균 성별 추출
    # 피해운전자 성별 : 남 1, 여 0, else nan
    df['피해운전자 성별'] = df['피해운전자 성별'].apply(lambda x:sex_transform(x))
    
    sex_mean = df[['구','동','피해운전자 성별']].groupby(['구','동']).mean()    
    sex_mean.columns = ['피해운전자 평균성별']
    
    df = pd.merge(df, sex_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 성별 추가 : 전체 평균으로 nan에 값을 채움
    test_eng = pd.merge(test_eng, sex_mean, how="left", on=["구", "동"])
    test_eng[['피해운전자 평균성별']] = test_eng[['피해운전자 평균성별']].fillna(test_eng[['피해운전자 평균성별']].mean())
    
    ##############################################################################################
    # == 가해노인운전자 위험도 ==
    # 가해운전자 연령 : 65세 이상 1, 그외 0
    df["가해운전자 연령"] = df["가해운전자 연령"].apply(sepa)
    
    old_count = df[["구", "동", "가해운전자 연령"]]    
    # cnt 변수
    old_count["cnt"] = 1    
    # 동별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령=1) 합계 
    old_count = old_count.groupby(["구", "동"]).sum().reset_index()    
    # 노인 사고 / 전체 사고
    old_count["ratio"] = old_count["가해운전자 연령"] / old_count["cnt"]
    
    # 동별 노인 사고 평균 ECLO 
    old_eclo = df[df["가해운전자 연령"]==1][["구", "동", "ECLO"]]
    old_eclo = old_eclo.groupby(["구", "동"]).mean().reset_index()
    
    # 동별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령), 노인 평균 ECLO(ECLO) 병합
    temp = pd.merge(old_count, old_eclo, how="left", on=["구", "동"])
    # nan -> 0
    temp.fillna(0, inplace=True)
    
    # (노인 사고건수 / 전체 사고건수) * 노인 사고 평균 ECLO 
    temp["가해노인운전자 위험도"] = temp["ratio"] * temp["ECLO"]
    
    temp.drop(["가해운전자 연령", "cnt", "ratio", "ECLO"], axis=1, inplace=True)

    df = pd.merge(df, temp, how="left", on=["구", "동"])
    test_eng = pd.merge(test_eng, temp, how="left", on=["구", "동"])

    # == 피해노인운전자 위험도 ==
    # 피해운전자 연령 : 65세 이상 1, 그외 0
    df["피해운전자 연령"] = df["피해운전자 연령"].apply(sepa)
    
    old_count = df[["구", "동", "피해운전자 연령"]]    
    # cnt 변수
    old_count["cnt"] = 1    
    # 동별 전체 사고건수(cnt), 노인 사고건수(피해운전자 연령=1) 합계 
    old_count = old_count.groupby(["구", "동"]).sum().reset_index()    
    # 노인 사고 / 전체 사고
    old_count["ratio"] = old_count["피해운전자 연령"] / old_count["cnt"]
    
    # 동별 노인 사고 평균 ECLO 
    old_eclo = df[df["피해운전자 연령"]==1][["구", "동", "ECLO"]]
    old_eclo = old_eclo.groupby(["구", "동"]).mean().reset_index()
    
    # 동별 전체 사고건수(cnt), 노인 사고건수(피해운전자 연령), 노인 평균 ECLO(ECLO) 병합
    temp = pd.merge(old_count, old_eclo, how="left", on=["구", "동"])
    # nan -> 0
    temp.fillna(0, inplace=True)
    
    # (노인 사고건수 / 전체 사고건수) * 노인 사고 평균 ECLO 
    temp["피해노인운전자 위험도"] = temp["ratio"] * temp["ECLO"]
    
    temp.drop(["피해운전자 연령", "cnt", "ratio", "ECLO"], axis=1, inplace=True)
    
    df = pd.merge(df, temp, how="left", on=["구", "동"])
    test_eng = pd.merge(test_eng, temp, how="left", on=["구", "동"])
    
    ##############################################################################################
    # == 가해운전자 차종 별 위험도 ==
    # danger : 컬럼 별 위험도(ECLO)를 측정 후 "동" 별로 컬럼별 발생 비율을 반영한 가중치 계산
    danger1 = danger(df, "가해운전자 차종")

    df = pd.merge(df, danger1, how="left", on=["구", "동"])
    test_eng = pd.merge(test_eng, danger1, how="left", on=["구", "동"])
    
    # == 피해운전자 차종 별 위험도 ==
    # danger : 컬럼 별 위험도(ECLO)를 측정 후 "동" 별로 컬럼별 발생 비율을 반영한 가중치 계산
    danger1 = danger(df, "피해운전자 차종")

    df = pd.merge(df, danger1, how="left", on=["구", "동"])
    test_eng = pd.merge(test_eng, danger1, how="left", on=["구", "동"])

    ##############################################################################################
    # == 칼럼 동기화 ==
    ytest = test_eng['ECLO']
    test_eng = test_eng[['요일', '도로형태', '노면상태', '연', '월', '일', '시간', 'Holiday', '구', '동', '도로형태1',
                         '가해운전자 평균연령', '가해운전자 평균성별', "가해노인운전자 위험도", "가해운전자 차종_dangerous",                         
                         '피해운전자 평균연령', '피해운전자 평균성별', "피해노인운전자 위험도", "피해운전자 차종_dangerous"]]
    
    ytrain = df["ECLO"]    
    df = df[test_eng.columns]

    ##############################################################################################
    # == 원핫 인코딩 ==
    ## '노면상태',"도로형태1"
    one_hot_features=['노면상태',"도로형태1"]

    train_oh = pd.get_dummies(df[one_hot_features])
    test_oh = pd.get_dummies(test_eng[one_hot_features])

    for i in train_oh.columns:
        if i not in test_oh.columns:
            test_oh[i] = 0
    for i in test_oh.columns:
        if i not in train_oh.columns:
            train_oh[i] = 0

    print("[원핫인코딩] test_oh 컬럼수 : ", len(test_oh.columns), "/ train_oh 컬럼수 : ", len(train_oh.columns))

    # 원데이터 드롭
    df.drop(one_hot_features,axis=1,inplace=True)
    test_eng.drop(one_hot_features,axis=1,inplace=True)

    # 원핫인코딩 결과 병합
    df = pd.concat([df,train_oh],axis=1)
    test_eng = pd.concat([test_eng,test_oh],axis=1)

    ##############################################################################################
    # == 일 드롭 ==
    df = df.drop(columns=["일"])
    
    ##############################################################################################
    # == 레이블 인코딩
    ## "요일", "구", "동", "도로형태"
    label_features = ["요일", "구", "동", "도로형태"]

    for i in label_features:
        print("[레이블인코딩] ", i)
        le = LabelEncoder()
        le = le.fit(df[i])
        df[i] = le.transform(df[i])

        for case in np.unique(test_eng[i]):
            if case not in le.classes_:
                print(case, ' : test case is not in classes')
                le.classes_ = np.append(le.classes_, case)
        test_eng[i] = le.transform(test_eng[i])

    # == 타겟 인코딩
    return df, ytrain, test_eng, ytest

#%%
X_train_eng2, y_train2, X_test_eng2, y_test_eng2 = feat_eng2(train2)
"""
[원핫인코딩] test_oh 컬럼수 :  11 / train_oh 컬럼수 :  11
[레이블인코딩]  요일
[레이블인코딩]  구
[레이블인코딩]  동
[레이블인코딩]  도로형태
"""

#%%
# 차대사람 df(train3), test3
def feat_eng3(df) :
    global test3    
    
    # == 사고유형 -> drop ==
    # 사고유형 == 차대사람
    df = df.drop(columns=["사고유형"])
    
    test_eng = test3.copy()    
    test_eng = test_eng.drop(columns=["사고유형"])

    ##############################################################################################
    # == 지역(동)별 가해운전자 평균 연령 추출 ==
    age_mean = df[['구','동','가해운전자 연령']].groupby(['구','동']).mean()
    age_mean.columns = ['가해운전자 평균연령']
    
    df = pd.merge(df, age_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 연령 추가 : 전체 평균으로 nan에 값을 채움
    # == 사고유형 -> drop ==
    test_eng = pd.merge(test_eng, age_mean, how="left", on=["구", "동"])
    test_eng[['가해운전자 평균연령']] = test_eng[['가해운전자 평균연령']].fillna(test_eng[['가해운전자 평균연령']].mean())
    
    # == 지역(동)별 피해운전자 평균 연령 추출 ==
    age_mean = df[['구','동','피해운전자 연령']].groupby(['구','동']).mean()
    age_mean.columns = ['피해운전자 평균연령']
    
    df = pd.merge(df, age_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 연령 추가 : 전체 평균으로 nan에 값을 채움
    # == 사고유형 -> drop ==
    test_eng = pd.merge(test_eng, age_mean, how="left", on=["구", "동"])
    test_eng[['피해운전자 평균연령']] = test_eng[['피해운전자 평균연령']].fillna(test_eng[['피해운전자 평균연령']].mean())

    ##############################################################################################
    # == 지역별 가해운전자 평균 성별 추출
    # 가해운전자 성별 : 남 1, 여 0, else nan
    df['가해운전자 성별'] = df['가해운전자 성별'].apply(lambda x:sex_transform(x))
    
    sex_mean = df[['구','동','가해운전자 성별']].groupby(['구','동']).mean()    
    sex_mean.columns = ['가해운전자 평균성별']
    
    df = pd.merge(df, sex_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 성별 추가 : 전체 평균으로 nan에 값을 채움
    test_eng = pd.merge(test_eng, sex_mean, how="left", on=["구", "동"])
    test_eng[['가해운전자 평균성별']] = test_eng[['가해운전자 평균성별']].fillna(test_eng[['가해운전자 평균성별']].mean())
    
    # == 지역별 피해운전자 평균 성별 추출
    # 피해운전자 성별 : 남 1, 여 0, else nan
    df['피해운전자 성별'] = df['피해운전자 성별'].apply(lambda x:sex_transform(x))
    
    sex_mean = df[['구','동','피해운전자 성별']].groupby(['구','동']).mean()    
    sex_mean.columns = ['피해운전자 평균성별']
    
    df = pd.merge(df, sex_mean, how="left", on=["구", "동"])
    
    # test_eng에 평균 성별 추가 : 전체 평균으로 nan에 값을 채움
    test_eng = pd.merge(test_eng, sex_mean, how="left", on=["구", "동"])
    test_eng[['피해운전자 평균성별']] = test_eng[['피해운전자 평균성별']].fillna(test_eng[['피해운전자 평균성별']].mean())
    
    ##############################################################################################
    # == 가해노인운전자 위험도 ==
    # 가해운전자 연령 : 65세 이상 1, 그외 0
    df["가해운전자 연령"] = df["가해운전자 연령"].apply(sepa)
    
    old_count = df[["구", "동", "가해운전자 연령"]]    
    # cnt 변수
    old_count["cnt"] = 1    
    # 동별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령=1) 합계 
    old_count = old_count.groupby(["구", "동"]).sum().reset_index()    
    # 노인 사고 / 전체 사고
    old_count["ratio"] = old_count["가해운전자 연령"] / old_count["cnt"]
    
    # 동별 노인 사고 평균 ECLO 
    old_eclo = df[df["가해운전자 연령"]==1][["구", "동", "ECLO"]]
    old_eclo = old_eclo.groupby(["구", "동"]).mean().reset_index()
    
    # 동별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령), 노인 평균 ECLO(ECLO) 병합
    temp = pd.merge(old_count, old_eclo, how="left", on=["구", "동"])
    # nan -> 0
    temp.fillna(0, inplace=True)
    
    # (노인 사고건수 / 전체 사고건수) * 노인 사고 평균 ECLO 
    temp["가해노인운전자 위험도"] = temp["ratio"] * temp["ECLO"]
    
    temp.drop(["가해운전자 연령", "cnt", "ratio", "ECLO"], axis=1, inplace=True)

    df = pd.merge(df, temp, how="left", on=["구", "동"])
    test_eng = pd.merge(test_eng, temp, how="left", on=["구", "동"])

    ##############################################################################################
    # == 가해운전자 차종 별 위험도 ==
    # danger : 컬럼 별 위험도(ECLO)를 측정 후 "동" 별로 컬럼별 발생 비율을 반영한 가중치 계산
    danger1 = danger(df, "가해운전자 차종")

    df = pd.merge(df, danger1, how="left", on=["구", "동"])
    test_eng = pd.merge(test_eng, danger1, how="left", on=["구", "동"])
    
    ##############################################################################################
    # == 칼럼 동기화 ==
    ytest = test_eng['ECLO']
    test_eng = test_eng[['요일', '도로형태', '노면상태', '연', '월', '일', '시간', 'Holiday', '구', '동', '도로형태1',
                         '가해운전자 평균연령', '가해운전자 평균성별', "가해노인운전자 위험도", "가해운전자 차종_dangerous",                         
                         '피해운전자 평균연령', '피해운전자 평균성별']]
    
    ytrain = df["ECLO"]    
    df = df[test_eng.columns]

    ##############################################################################################
    # == 원핫 인코딩 ==
    ## '노면상태',"도로형태1"
    one_hot_features=['노면상태',"도로형태1"]

    train_oh = pd.get_dummies(df[one_hot_features])
    test_oh = pd.get_dummies(test_eng[one_hot_features])

    for i in train_oh.columns:
        if i not in test_oh.columns:
            test_oh[i] = 0
    for i in test_oh.columns:
        if i not in train_oh.columns:
            train_oh[i] = 0

    print("[원핫인코딩] test_oh 컬럼수 : ", len(test_oh.columns), "/ train_oh 컬럼수 : ", len(train_oh.columns))

    # 원데이터 드롭
    df.drop(one_hot_features,axis=1,inplace=True)
    test_eng.drop(one_hot_features,axis=1,inplace=True)

    # 원핫인코딩 결과 병합
    df = pd.concat([df,train_oh],axis=1)
    test_eng = pd.concat([test_eng,test_oh],axis=1)

    ##############################################################################################
    # == 일 드롭 ==
    df = df.drop(columns=["일"])
    
    ##############################################################################################
    # == 레이블 인코딩
    ## "요일", "구", "동", "도로형태"
    label_features = ["요일", "구", "동", "도로형태"]

    for i in label_features:
        print("[레이블인코딩] ", i)
        le = LabelEncoder()
        le = le.fit(df[i])
        df[i] = le.transform(df[i])

        for case in np.unique(test_eng[i]):
            if case not in le.classes_:
                print(case, ' : test case is not in classes')
                le.classes_ = np.append(le.classes_, case)
        test_eng[i] = le.transform(test_eng[i])

    # == 타겟 인코딩
    return df, ytrain, test_eng, ytest

#%%
X_train_eng3, y_train3, X_test_eng3, y_test_eng3 = feat_eng3(train3)
"""
[원핫인코딩] test_oh 컬럼수 :  11 / train_oh 컬럼수 :  11
[레이블인코딩]  요일
[레이블인코딩]  구
[레이블인코딩]  동
[레이블인코딩]  도로형태
"""

#%% 결측치 확인
print(X_train_eng1.isnull().sum().sum())
print(X_test_eng1.isnull().sum().sum())  ## 2
X_test_eng1[X_test_eng1["가해노인운전자 위험도"].isna()]
"""
    요일  도로형태     연  월   일  ...  도로형태1_교차로  도로형태1_기타  도로형태1_단일로  노면상태_적설  도로형태1_주차장
52   1     6  2023  5  18  ...      False     False       True        0          0
"""
X_test_eng1[X_test_eng1["가해운전자 차종_dangerous"].isna()]
"""
    요일  도로형태     연  월   일  ...  도로형태1_교차로  도로형태1_기타  도로형태1_단일로  노면상태_적설  도로형태1_주차장
52   1     6  2023  5  18  ...      False     False       True        0          0
"""
#>> 장안구 상광교동 교통사고 데이터가 2018~2022년에는 없으나 2023년에는 있어서 null
#>> nan -> 0
X_test_eng1.fillna(0, inplace=True)

print(y_train1.isnull().sum().sum())
print(y_test_eng1.isnull().sum().sum())
print(X_train_eng2.isnull().sum().sum())
print(X_test_eng2.isnull().sum().sum())
print(y_train2.isnull().sum().sum())
print(y_test_eng2.isnull().sum().sum())
print(X_train_eng3.isnull().sum().sum())
print(X_test_eng3.isnull().sum().sum())
print(y_train3.isnull().sum().sum())
print(y_test_eng3.isnull().sum().sum())

#%% 하이퍼 파라미터 튜닝
# Log transformation of target variable : 로그 변환
y_train_log_total1 = np.log1p(y_train1)
y_train_log_total2 = np.log1p(y_train2)
y_train_log_total3 = np.log1p(y_train3)

X_train1, X_valid1, y_train_log1, y_valid_log1 = train_test_split(X_train_eng1, y_train_log_total1, test_size=0.2, random_state=42, shuffle=True)
X_train2, X_valid2, y_train_log2, y_valid_log2 = train_test_split(X_train_eng2, y_train_log_total2, test_size=0.2, random_state=42, shuffle=True)
X_train3, X_valid3, y_train_log3, y_valid_log3 = train_test_split(X_train_eng3, y_train_log_total3, test_size=0.2, random_state=42, shuffle=True)

#%% LGBM 
from lightgbm import LGBMRegressor, early_stopping
import optuna
from sklearn.metrics import mean_squared_log_error as msle

def lgbm_modeling(X_train, y_train, X_valid, y_valid):
  # [목표 함수 정의] 
  def objective(trial):
    # param이라는 딕셔너리 정의 : LGBM 모델의 하이퍼파라미터  
    param = {
        'objective': 'regression',                                             # 회귀
        'verbose': -1,                                                         # 로깅 자세성 수준(조용한 모드는 -1로 설정)
        'metric': 'rmse',                                                      # 평가 지표('rmse'를 루트 평균 제곱 오류로 설정)
        # trial.suggest_int : 지정된 범위에서 정수 값을 제안
        'num_leaves': trial.suggest_int('num_leaves', 2, 16),                  # 각 트리의 잎 수
        # trial.suggest_uniform : 지정된 범위에서 랜덤 부동소수점 값을 제안
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 1.0), # 각 트리를 구성할 때 열의 서브샘플 비율
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0.0, 1.0),             # L1 정규화 항
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0.0, 10.0),          # L2 정규화 항
        'max_depth': trial.suggest_int('max_depth', 3, 8),                     # 각 트리의 최대 깊이
        # trial.suggest_loguniform : 로그 일관성 분포에서 랜덤 값을 제안
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-2, 0.1), # 모델 업데이트를 위한 학습률
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),          # 포레스트에서 자라는 트리의 수
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 40),   # 자식 노드에 필요한 최소 샘플 수
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),             # 학습 인스턴스의 서브샘플 비율
    }

    # [모델 학습 및 조기 중단]
    model = LGBMRegressor(**param, random_state=42, n_jobs=-1)
    # random_state=42 : 무작위 시드를 42로 설정하여 결과 재현성을 보장
    # n_jobs=-1 : 가능한 한 많은 CPU 코어를 사용하여 학습 속도를 향상
    
    bst_lgbm = model.fit(X_train, y_train, eval_set = [(X_valid,y_valid)], 
                         eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=100)])
    # eval_set = [(X_valid,y_valid)] : 검증 데이터를 사용하여 모델 성능을 평가하도록 지정
    # eval_metric='rmse' : 평가 지표로 루트 평균 제곱 오류(RMSE)를 사용하도록 지정
    # callbacks=[early_stopping(stopping_rounds=100)] : early_stopping 콜백 함수를 사용하여 과적합을 방지
    #       > early_stopping(stopping_rounds=100) : 검증 손실이 100번 연속으로 개선되지 않으면 학습을 중단
    
    # [예측 및 손실 계산]
    preds = bst_lgbm.predict(X_valid)
    
    # 예측값 중 음수가 있으면 0으로 바꿈
    if (preds<0).sum() > 0:
      print('negative')
      preds = np.where(preds>0,preds,0)
     
    # 평균 제곱 로그 오류(MSLE)를 사용하여 검증 데이터의 실제 값 y_valid과 예측값 preds 간의 손실을 계산  
    loss = msle(y_valid,preds)
    
    # [SQRT RMSE 반환]
    # 계산된 MSLE 손실의 제곱근 반환
    return np.sqrt(loss)


  # [Optuna 최적화 및 최상의 모델 반환]  
  study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=100))
  # optuna.create_study 함수를 사용하여 새로운 Optuna 연구 객체
  # direction='minimize': 최적화 방향을 '최소화'(목표 함수(objective)가 반환하는 값을 최소화하기 위해)
  # sampler=optuna.samplers.TPESampler(seed=100) : 하이퍼파라미터 샘플러로 TPESampler를 사용하며 시드 값을 100으로 설정하여 재현성 보장
  #     > TPESampler : 과거 시도 결과를 기반으로 하이퍼파라미터 조합을 제안하는 알고리즘
  
  study_lgbm.optimize(objective, n_trials=30, show_progress_bar=False)
  # optimize 메서드를 사용하여 하이퍼파라미터 최적화를 수행
  # n_trials=30: 시도할 하이퍼파라미터 조합의 수 설정
  # show_progress_bar=True: 최적화 진행 상황을 표시하는 진행률 표시줄 출력

  # [최상의 모델 사용] 새로운 LGBM 회귀 모델 생성
  # **study_lgbm.best_params : 최적화 과정에서 발견된 최상의 하이퍼파라미터 세트를 study_lgbm 객체에서 가져옴
  lgbm_reg = LGBMRegressor(**study_lgbm.best_params, random_state=42, n_jobs=-1)  
  lgbm_reg.fit(X_train, y_train, eval_set = [(X_valid,y_valid)], eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=100)])

  # [학습된 최적 모델 lgbm_reg과 Optuna 연구 객체 study_lgbm을 함께 반환]  
  return lgbm_reg,study_lgbm

#%%
# lgbm1, study_lgbm1 - 사고유형 : 차량단독
lgbm1, study_lgbm1 = lgbm_modeling(X_train1, y_train_log1, X_valid1, y_valid_log1)
# lgbm2, study_lgbm2 - 사고유형 : 차대차
lgbm2, study_lgbm2 = lgbm_modeling(X_train2, y_train_log2, X_valid2, y_valid_log2)
# lgbm3, study_lgbm3 - 사고유형 : 차대사람
lgbm3, study_lgbm3 = lgbm_modeling(X_train3, y_train_log3, X_valid3, y_valid_log3)

# Optuna 연구 객체 study_lgbm1
print(study_lgbm1.best_params)
"""
{'num_leaves': 3, 'colsample_bytree': 0.9505778710449351, 'reg_alpha': 0.3379624888729935, 
 'reg_lambda': 8.425455367797097, 'max_depth': 3, 'learning_rate': 0.04437368353073049, 
 'n_estimators': 626, 'min_child_samples': 26, 'subsample': 0.9992613153173495}
"""
#  > 최적화 프로세스에서 발견된 목표 함수의 최상의 값
# = objective에서 계산된 MSLE 손실의 제곱근의 최저값
print(study_lgbm1.best_value)
### 0.19136748569556997

# Optuna 연구 객체 study_lgbm2
print(study_lgbm2.best_params)
"""
{'num_leaves': 3, 'colsample_bytree': 0.8584105669954142, 'reg_alpha': 0.9921580365105283, 
 'reg_lambda': 3.950359317582296, 'max_depth': 5, 'learning_rate': 0.06389259646429983, 
 'n_estimators': 779, 'min_child_samples': 26, 'subsample': 0.85361467318491}
"""
print(study_lgbm2.best_value)
### 0.17028704985460946

# Optuna 연구 객체 study_lgbm3
print(study_lgbm3.best_params)
"""
{'num_leaves': 10, 'colsample_bytree': 0.9307345513316955, 'reg_alpha': 0.2506952291383959, 
 'reg_lambda': 2.8589569040686467, 'max_depth': 8, 'learning_rate': 0.0944074992066832, 
 'n_estimators': 897, 'min_child_samples': 27, 'subsample': 0.8395435783502989}
"""
print(study_lgbm3.best_value)
### 0.12376566288298953

#%% 피처중요도
import lightgbm as lgb

## 사고유형 : 차량단독
lgb.plot_importance(lgbm1, height=0.8, figsize=(10, 8), title="Feature Importance1")
## 사고유형 : 차대차
lgb.plot_importance(lgbm2, height=0.8, figsize=(10, 8), title="Feature Importance2")
## 사고유형 : 차대사람
lgb.plot_importance(lgbm3, height=0.8, figsize=(10, 8), title="Feature Importance3")
plt.show()

#%% Catboost
from catboost import CatBoostRegressor

def catboost_modeling(X_train, y_train, X_valid, y_valid):
    # [목표 함수 정의] 
    def objective(trial):
        # param이라는 딕셔너리 정의 : Catboost 모델의 하이퍼파라미터  
        param = {
            'iterations': trial.suggest_int("iterations", 1000, 8000),
            'od_wait': trial.suggest_int('od_wait', 500, 1500),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.2),
            'reg_lambda': trial.suggest_uniform('reg_lambda', 1e-5, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1),
            'random_strength': trial.suggest_uniform('random_strength', 10, 30),
            'depth': trial.suggest_int('depth', 5, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
            'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 10.00),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        }
        
        # [모델 학습]
        model = CatBoostRegressor(**param, random_seed=42, thread_count=-1)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, verbose_eval=False)

        # [예측 및 손실 계산]
        preds = model.predict(X_valid)
        
        # 예측값 중 음수가 있으면 0으로 바꿈
        if (preds < 0).sum() > 0:
            print('Negative predictions found. Adjusting...')
            preds = np.where(preds > 0, preds, 0)

        # 평균 제곱 로그 오류(MSLE)를 사용하여 검증 데이터의 실제 값 y_valid과 예측값 preds 간의 손실을 계산  
        loss = msle(y_valid, preds)
        
        # [SQRT RMSE 반환]
        # 계산된 MSLE 손실의 제곱근 반환
        return np.sqrt(loss)

    # [Optuna 최적화 및 최상의 모델 반환]  
    study_catboost = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=100))
    study_catboost.optimize(objective, n_trials=30, show_progress_bar=True)

    # [최상의 모델 사용] 새로운 LGBM 회귀 모델 생성
    # **study_catboost.best_params : 최적화 과정에서 발견된 최상의 하이퍼파라미터 세트를 study_lgbm 객체에서 가져옴
    catboost_reg = CatBoostRegressor(**study_catboost.best_params, random_seed=42, thread_count=-1)
    catboost_reg.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, verbose_eval=False)

    # [학습된 최적 모델 catboost_reg Optuna 연구 객체 study_catboost 함께 반환]
    return catboost_reg, study_catboost

#%%
# catboost1, study_catboost1 - 사고유형 : 차량단독
catboost1, study_catboost1 = catboost_modeling(X_train1, y_train_log1, X_valid1, y_valid_log1)
# catboost2, study_catboost2 - 사고유형 : 차대차
catboost2, study_catboost2 = catboost_modeling(X_train2, y_train_log2, X_valid2, y_valid_log2)
# catboost3, study_catboost3 - 사고유형 : 차대사람
catboost3, study_catboost3 = catboost_modeling(X_train3, y_train_log3, X_valid3, y_valid_log3)

# Optuna 연구 객체 study_catboost1
print(study_catboost1.best_params)
"""
{'iterations': 7240, 'od_wait': 1374, 'learning_rate': 0.1291721563482153, 'reg_lambda': 7.539777138628888, 
 'subsample': 0.7251666564158438, 'random_strength': 21.432431757677623, 'depth': 9, 'min_data_in_leaf': 4, 
 'leaf_estimation_iterations': 8, 'bagging_temperature': 3.408605043231599, 'colsample_bylevel': 0.5721287697234007}
"""
print(study_catboost1.best_value)
### 0.18760916804109704

# Optuna 연구 객체 study_catboost2
print(study_catboost2.best_params)
"""
{'iterations': 7829, 'od_wait': 1313, 'learning_rate': 0.05390953359974773, 'reg_lambda': 6.566459608417512, 
 'subsample': 0.8548562473805941, 'random_strength': 19.63943045266951, 'depth': 7, 'min_data_in_leaf': 4, 
 'leaf_estimation_iterations': 8, 'bagging_temperature': 0.10721637564744814, 'colsample_bylevel': 0.6285653931833259}
"""
print(study_catboost2.best_value)
### 0.17008743822867786

# Optuna 연구 객체 study_catboost3
print(study_catboost3.best_params)
"""
{'iterations': 6571, 'od_wait': 1338, 'learning_rate': 0.12427333108659572, 'reg_lambda': 6.740203081958165, 
 'subsample': 0.7727142520850205, 'random_strength': 19.907406131595998, 'depth': 7, 'min_data_in_leaf': 5, 
 'leaf_estimation_iterations': 8, 'bagging_temperature': 0.01041409449605258, 'colsample_bylevel': 0.6096439296896454}
"""
print(study_catboost3.best_value)
### 0.12382627936607764

#%% 피처중요도
from catboost import Pool
import matplotlib.pyplot as plt
import pandas as pd

## 사고유형 : 차량단독
# X_train1 데이터프레임의 열 이름을 추출하여 feature_names_list 리스트에 저장
feature_names_list = list(X_train1.columns)

# CatBoost Pool 객체를 생성
catboost_pool = Pool(data=X_train1, label=y_train_log1, feature_names=feature_names_list)
# get_feature_importance 메서드를 사용하여 특징 중요도 점수 추출
# > 모델의 각 특징에 대한 중요도 점수 리스트를 반환
feature_importance = catboost1.get_feature_importance(
    data=catboost_pool,
    type='PredictionValuesChange'
    # 계산할 특징 중요도의 유형 정의
    # "PredictionValuesChange" : 특정 특징 값을 변경했을 때 모델 예측의 평균 변화
)

# 시각화를 위한 데이터 구성
feature_importance_df = pd.DataFrame({'Feature': feature_names_list, 'Importance': feature_importance})
# 'Importance' 열을 기준으로 feature_importance_df DataFrame을 오름차순으로 정렬
feature_importance_df = feature_importance_df.sort_values(by='Importance')

# Plot the feature importance
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance (PredictionValuesChange)')
plt.ylabel('Features')
plt.title('CatBoost Feature Importance1 (Ascending Order)')
plt.show()

## 사고유형 : 차대차
feature_names_list = list(X_train2.columns)
catboost_pool = Pool(data=X_train2, label=y_train_log2, feature_names=feature_names_list)

feature_importance = catboost2.get_feature_importance(
    data=catboost_pool,
    type='PredictionValuesChange'
)

feature_importance_df = pd.DataFrame({'Feature': feature_names_list, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance')

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance (PredictionValuesChange)')
plt.ylabel('Features')
plt.title('CatBoost Feature Importance2 (Ascending Order)')
plt.show()


## 사고유형 : 차대사람
feature_names_list = list(X_train3.columns)
catboost_pool = Pool(data=X_train3, label=y_train_log3, feature_names=feature_names_list)

feature_importance = catboost3.get_feature_importance(
    data=catboost_pool,
    type='PredictionValuesChange'
)

feature_importance_df = pd.DataFrame({'Feature': feature_names_list, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance')

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance (PredictionValuesChange)')
plt.ylabel('Features')
plt.title('CatBoost Feature Importance3 (Ascending Order)')
plt.show()

#%% 모델 검증(예측)
print(X_train1.columns) 
print(X_test_eng1.columns)
"""
Index(['요일', '도로형태', '연', '월', '시간', 'Holiday', '구', '동', '가해운전자 평균연령',
       '가해운전자 평균성별', '가해노인운전자 위험도', '가해운전자 차종_dangerous', '노면상태_건조', '노면상태_기타',
       '노면상태_서리/결빙', '노면상태_적설', '노면상태_젖음/습기', '도로형태1_교차로', '도로형태1_기타',
       '도로형태1_단일로', '도로형태1_주차장'],
      dtype='object')

Index(['요일', '도로형태', '연', '월', ['일'], '시간', 'Holiday', '구', '동', '가해운전자 평균연령',
       '가해운전자 평균성별', '가해노인운전자 위험도', '가해운전자 차종_dangerous', '노면상태_건조', '노면상태_기타',
       '노면상태_서리/결빙', '노면상태_젖음/습기', '도로형태1_교차로', '도로형태1_기타', '도로형태1_단일로',
       '노면상태_적설', '도로형태1_주차장'],
      dtype='object')
"""
X_test_eng1=X_test_eng1.drop(columns=["일"])
X_test_eng2=X_test_eng2.drop(columns=["일"])
X_test_eng3=X_test_eng3.drop(columns=["일"])

lgbm_prediction1 = np.expm1(lgbm1.predict(X_test_eng1))
lgbm_prediction2 = np.expm1(lgbm2.predict(X_test_eng2))
lgbm_prediction3 = np.expm1(lgbm3.predict(X_test_eng3))

catboost_prediction1 = np.expm1(catboost1.predict(X_test_eng1))
catboost_prediction2 = np.expm1(catboost2.predict(X_test_eng2))
catboost_prediction3 = np.expm1(catboost3.predict(X_test_eng3))

#%% 앙상블 : lgbm_prediction*0.2 + catboost_prediction*0.8
test1["predict"]=lgbm_prediction1*0.2+catboost_prediction1*0.8
test2["predict"]=lgbm_prediction2*0.2+catboost_prediction2*0.8
test3["predict"]=lgbm_prediction3*0.2+catboost_prediction3*0.8

test1_result = test1[['사고번호', 'ECLO', 'predict']]
test2_result = test2[['사고번호', 'ECLO', 'predict']]
test3_result = test3[['사고번호', 'ECLO', 'predict']]

test_result = pd.concat([test1_result, test2_result, test3_result])

#%%
plt.figure(figsize = (10,5))
ax1 = sns.kdeplot(test_result['ECLO'], label = 'y_test')
ax2 = sns.kdeplot(test_result['predict'], label = 'y_hat', ax=ax1)
plt.legend()
plt.show()