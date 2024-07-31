# -*- coding: utf-8 -*-
"""
https://velog.io/@car052000/%EB%8D%B0%EC%9D%B4%EC%BD%98-%ED%9B%84%EA%B8%B0%EB%8C%80%EA%B5%AC-%EA%B5%90%ED%86%B5%EC%82%AC%EA%B3%A0-%ED%94%BC%ED%95%B4-%EC%98%88%EC%B8%A1-AI-%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C

https://github.com/ParkSeokwoo/dacon_first/blob/main/%EB%B8%94%EB%A1%9C%EA%B7%B8_%EC%9E%91%EC%84%B1%EC%9A%A9.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import Timestamp
from workalendar.asia import SouthKorea

import warnings
warnings.filterwarnings("ignore")

## 한글 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

from sklearn.preprocessing import LabelEncoder

#%% 데이터 불러오기
train_df=pd.read_csv("./train.csv")
test_df=pd.read_csv("./test.csv")
ss=pd.read_csv("./sample_submission.csv")
total_df=pd.read_csv("./countrywide_accident.csv")

# 대구 데이터 + 전국 데이터
country_df=pd.concat([train_df, total_df])

#%% 전체 데이터 파생변수
country_df.columns
"""
Index(['ID', '사고일시', '요일', '기상상태', '시군구', '도로형태', '노면상태', '사고유형',
       '사고유형 - 세부분류', '법규위반', '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도',
       '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도', '사망자수', '중상자수',
       '경상자수', '부상자수', 'ECLO'],
      dtype='object')
"""

# == ID : 패스 ==

#%%
# == 사고일시 ==
country_df['사고일시'][0] 
## 0    2019-01-01 00

## -> 연, 월, 일, 시간 분리 & 추가
time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2})'

country_df[['연', '월', '일', '시간']] = country_df['사고일시'].str.extract(time_pattern)
country_df[['연', '월', '일', '시간']] = country_df[['연', '월', '일', '시간']].apply(pd.to_numeric) # 추출된 문자열을 수치화해줍니다

## -> 사고일시 삭제
country_df.drop(columns=['사고일시'], inplace = True) 

#%%
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
country_df['Holiday'] = country_df.apply(lambda row: classify_day(row['연'], row['월'], row['일']), axis=1)

#%%
# == 시군구 ==
country_df['시군구'][0] 
## 0     대구광역시 중구 대신동

## -> 도시, 구, 동 분리 & 추가
location_pattern = r'(\S+) (\S+) (\S+)'

country_df[['도시', '구', '동']] = country_df['시군구'].str.extract(location_pattern)

## -> 시군구 삭제
country_df.drop(columns=['시군구'], inplace = True)

#%%
# == 도로형태 ==
country_df['도로형태'][0] 
## 0          단일로 - 기타

## -> 도로형태1, 도로형태2 분리 & 추가
road_pattern = r'(.+) - (.+)'

country_df[['도로형태1', '도로형태2']] = country_df['도로형태'].str.extract(road_pattern)

## -> 도로형태2 삭제
country_df = country_df.drop(columns=['도로형태2'])

#%%
# == 기상상태 : 삭제 ==
country_df.drop(["기상상태"], axis=1, inplace=True)

#%%
# == 연령 ==
country_df['가해운전자 연령'][0] 
## 0    51세
country_df['피해운전자 연령'][0] 
## 0    70세

## -> 수치화
g_age = r'(\d{2})세'    ## 두 자리 숫자로 표현된 나이만 일치
p_age = r'(\d{1,2})세'  ## 하나 또는 두 자리 숫자로 표현된 나이 일치

country_df["가해운전자 연령"] = country_df['가해운전자 연령'].str.extract(g_age)
country_df["피해운전자 연령"] = country_df['피해운전자 연령'].str.extract(p_age)

country_df["가해운전자 연령"]=country_df["가해운전자 연령"].apply(pd.to_numeric)
country_df["피해운전자 연령"]=country_df["피해운전자 연령"].apply(pd.to_numeric)

#%%
test_df.columns
"""
Index(['ID', '사고일시', '요일', '기상상태', '시군구', '도로형태', '노면상태', '사고유형'], dtype='object')
"""
# == ID 패스 ==

# == 사고일시 ==
test_df[['연', '월', '일', '시간']] = test_df['사고일시'].str.extract(time_pattern)
test_df[['연', '월', '일', '시간']] = test_df[['연', '월', '일', '시간']].apply(pd.to_numeric) 

test_df.drop(columns=['사고일시'], inplace = True)

# = 공휴일 =
test_df['Holiday'] = test_df.apply(lambda row: classify_day(row['연'], row['월'], row['일']), axis=1) 

# == 시군구 ==
test_df[['도시', '구', '동']] = test_df['시군구'].str.extract(location_pattern)

test_df.drop(columns=['시군구'], inplace = True)

# == 도로형태 ==
test_df[['도로형태1', '도로형태2']] = test_df['도로형태'].str.extract(road_pattern)

test_df = test_df.drop(columns=['도로형태2'])

# == 기상상태
test_df.drop(["기상상태"], axis=1, inplace=True)

#%%
# == 차대차, 차대사람의 데이터 ==
not_car = country_df[country_df['사고유형'] != '차량단독'].copy()
not_car.reset_index(inplace=True, drop=True)

# 차량단독이 아닌 경우에 대해서는 결측치 모두 drop
not_car.isna().sum()
"""
ID                        0
요일                      0
도로형태                  0
노면상태                  0
사고유형                  0
사고유형 - 세부분류       0
법규위반                  0
가해운전자 차종           0
가해운전자 성별           0
가해운전자 연령       10901
가해운전자 상해정도       0
피해운전자 차종           0
피해운전자 성별           0
피해운전자 연령         942
피해운전자 상해정도       0
사망자수                  0
중상자수                  0
경상자수                  0
부상자수                  0
ECLO                      0
연                        0
월                        0
일                        0
시간                      0
Holiday                   0
도시                    764
구                      764
동                      764
도로형태1                 0
dtype: int64
"""

not_car = not_car.dropna()
not_car.reset_index(inplace=True, drop=True)

# == 차량단독의 데이터 ==
car = country_df[country_df['사고유형'] == '차량단독'].copy()
car.reset_index(inplace=True, drop=True)

# 차량단독에서 ('노면상태', '가해운전자 연령', '도시', '구', '동') column에 대한 결측치 drop
car.isna().sum()
"""
ID                        0
요일                      0
도로형태                  0
노면상태                  1
사고유형                  0
사고유형 - 세부분류       0
법규위반                  0
가해운전자 차종           0
가해운전자 성별           0
가해운전자 연령          84
가해운전자 상해정도       0
피해운전자 차종       26818
피해운전자 성별       26820
피해운전자 연령       26819
피해운전자 상해정도   26820
사망자수                  0
중상자수                  0
경상자수                  0
부상자수                  0
ECLO                      0
연                        0
월                        0
일                        0
시간                      0
Holiday                   0
도시                     25
구                       25
동                       25
도로형태1                 0
dtype: int64
"""

car = car.dropna(subset=['노면상태', '가해운전자 연령', '도시', '구', '동'])
car.reset_index(inplace=True, drop=True)

# 차량단독의 경우 피해운전자 차종은 미분류나 결측치임. 따라서 없음으로 대치
car['피해운전자 차종'].unique()
## array([nan, '미분류'], dtype=object)

car['피해운전자 차종'] = '없음'

#%%
# 차량단독과 차량단독이 아닌 경우를 합쳐서 전체 데이터 형성
country_df = pd.concat([car, not_car], axis=0)
country_df.reset_index(inplace=True, drop=True)

train_df=country_df[country_df["도시"]=="대구광역시"]
country_df=country_df[country_df["도시"]!="대구광역시"]

# train_df(대구 데이터)와 country_df(전체 도시 데이터)의 object형 column 확인
temp_obj_col = []
con_obj_col = []

for col in train_df.columns:
    if train_df[col].dtype == object:
        temp_obj_col.append(col)
        
for col in country_df.columns:
    if country_df[col].dtype == object:
        con_obj_col.append(col)
        
print(temp_obj_col)
"""
['ID', '요일', '도로형태', '노면상태', '사고유형', '사고유형 - 세부분류', '법규위반', 
 '가해운전자 차종', '가해운전자 성별', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 상해정도', 
 '도시', '구', '동', '도로형태1']
"""
print(con_obj_col)
"""
['ID', '요일', '도로형태', '노면상태', '사고유형', '사고유형 - 세부분류', '법규위반', 
 '가해운전자 차종', '가해운전자 성별', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 상해정도', 
 '도시', '구', '동', '도로형태1']
"""
        
#%%
#대구 데이터에 없는 object형 변수 값을 갖고 있는 전체 도시 데이터의 인덱스를 파악
not_using_cols = ['ID', '도시', '구', '동', "도로형태1"]

idxs = []
for col in con_obj_col: ## 전체 도시 데이터 컬럼
    if col not in not_using_cols: 
        temp_values = list(train_df[col].unique())
        total_values = list(country_df[col].unique())

        # 컬럼에서 전체 도시 데이터에는 있으나 대구 데이터에는 없는 값 리스트 
        unique = []
        for tvalue in total_values: ## 전체 도시 데이터 값(unique)
            if tvalue not in temp_values: ## 대구 데이터 값(unique)
                unique.append(tvalue)
            
        # unique 리스트 값의 인덱스 리스트
        for u in unique:
            idx = country_df[country_df[col]==u].index.to_list()
            idxs = idxs + idx

idxs = list(set(idxs))

# 대구 데이터에 없는 값을 가지는 전체 도시 데이터 드롭
country_df = country_df.drop(idxs, axis=0)

country_df.reset_index(inplace=True, drop=True)

# 대구 데이터 + 전체 도시 데이터
country_df=pd.concat([train_df, country_df])

#%% 사고유형에 따라 데이터 분리
train1=country_df[country_df["사고유형"]=="차량단독"]
train2=country_df[country_df["사고유형"]=="차대차"]
train3=country_df[country_df["사고유형"]=="차대사람"]

test1=test_df[test_df["사고유형"]=="차량단독"]
test2=test_df[test_df["사고유형"]=="차대차"]
test3=test_df[test_df["사고유형"]=="차대사람"]

print(train1.shape)
print(train2.shape)
print(train3.shape)
"""
(26620, 29)
(489871, 29)
(113105, 29)
"""

#%%
# 컬럼 별 위험도(ECLO)를 측정 후 "구" 별로 컬럼별 발생 비율을 반영한 가중치 계산
def danger(df, column):
    col=str(column)
    
    # 컬럼별 평균 ECLO 계산
    column_dangerous=df[[col, "ECLO"]].groupby(col).mean()
    column_dangerous.columns=[col+"_dangerous"]
    
    # 컬럼별 구별 사고건수 계산
    column_count=df[["도시", "구", col]]
    column_count["cnt"]=1
    column_count=column_count.groupby(["도시", "구", col]).count()
    column_count.reset_index(inplace=True)
    
    # 컬럼별 평균 ECLO + 구별 사고건수
    temp=pd.merge(column_count, column_dangerous, how="left", on=[col])
    
    # 사고건수 * 평균 ECLO
    temp['multiply'] = temp['cnt']*temp[col+'_dangerous']
    
    # 구별 그룹
    temp = temp.groupby(['도시','구']).sum().reset_index().drop([col+"_dangerous"],axis=1)
    
    # (사고건수*평균ECLO)의 합 / 사고건수 합 
    temp[col+'_dangerous'] = temp['multiply']/temp['cnt']
    
    temp.drop([column,'multiply','cnt'],axis=1,inplace=True)
    
    return temp

#%%
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
    
#%%
#차량 단독 test1
def feat_eng1(df) :
    global test1    
    """
    df(train1) 
        - 'ECLO' 이상치 1% 제거
            : '노면상태', '사고유형', '사고유형 - 세부분류', '도로형태' 기준
        - 사고유형 drop 
    
    test1 -> test_eng
        - 동별 가해운전자 평균 연령
        - 동별 가해운전자 평균 성별 : 남 1, 여 0
        - 구별 가해"노인"운전자 위험도 : (노인사고건수/전체사고건수) * 노인 사고 평균 ECLO 
        - 구별 가해운전자 차종 별 위험도
    """
    
    # == 이상치 1% 제거 ==
    sns.boxplot(x="ECLO", data=df)
    plt.show()

    df_copy = df.copy()
    
    df_copy = df_copy[['노면상태', '사고유형', '사고유형 - 세부분류', '도로형태', 'ECLO']]
    df_copy.reset_index(inplace=True, drop=True)

    outlier_idxs = []
    for col in df_copy.columns[:-1]: # ECLO 제외 컬럼명 순차 추출
        # 그룹별 ECLO 0.999인 백분위수
        temp_df = pd.DataFrame(df_copy.groupby(col)['ECLO'].agg(set_limit)).reset_index()
        
        for j in range(len(temp_df)):
            # 컬럼 == col & ECLO > 0.999인 백분위수(이상치 1%)의 인덱스 리스트 추출
            s_idxs = df[(df[col] == temp_df.loc[j, col]) & (df['ECLO'] > temp_df.loc[j, 'ECLO'])].index.to_list()
            outlier_idxs = outlier_idxs + s_idxs
            
    # set로 변환해서 중복값 제거
    outliers = list(set(outlier_idxs))

    print('outlier 수 : ', len(outliers))

    df = df.drop(outliers, axis=0)
    df.reset_index(inplace=True, drop=True)

    print("train_df 데이터 수 : ", len(df))
    
    # 이상치 제거 후 그래프 확인
    sns.boxplot(x="ECLO", data=df)
    plt.show()

    ##############################################################################################
    # == 사고유형 -> drop ==
    # 사고유형 == 차량단독
    df.drop(["사고유형"], axis=1, inplace=True)

    ##############################################################################################
    # == 지역(동)별 가해운전자 평균 연령 추출 ==
    age_mean = df[['도시','구','동','가해운전자 연령']].groupby(['도시', '구','동']).mean()
    age_mean.columns = ['가해운전자 평균연령']
    
    df = pd.merge(df, age_mean, how="left", on=["도시", "구", "동"])
    
    # test_eng에 평균 연령 추가 : 전체 평균으로 nan에 값을 채움
    # == 사고유형 -> drop ==
    test_eng = pd.merge(test1, age_mean, how="left", on=["도시", "구", "동"]).drop(columns=["사고유형"])
    test_eng[['가해운전자 평균연령']]=test_eng[['가해운전자 평균연령']].fillna(test_eng[['가해운전자 평균연령']].mean())

    ##############################################################################################
    # == 지역별 가해운전자 평균 성별 추출
    # 가해운전자 성별 : 남 1, 여 0, else nan
    df['가해운전자 성별'] = df['가해운전자 성별'].apply(lambda x:sex_transform(x))
    
    sex_mean = df[['도시','구','동','가해운전자 성별']].groupby(['도시', '구','동']).mean()    
    sex_mean.columns = ['가해운전자 평균성별']
    
    df = pd.merge(df, sex_mean, how="left", on=["도시", "구", "동"])
    
    # test_eng에 평균 성별 추가 : 전체 평균으로 nan에 값을 채움
    test_eng = pd.merge(test_eng, sex_mean, how="left", on=["도시", "구", "동"])
    test_eng[['가해운전자 평균성별']] = test_eng[['가해운전자 평균성별']].fillna(test_eng[['가해운전자 평균성별']].mean())
    
    ##############################################################################################
    # == 가해노인운전자 위험도 ==
    # 가해운전자 연령 : 65세 이상 1, 그외 0
    df["가해운전자 연령"]=df["가해운전자 연령"].apply(sepa)
    
    old_count = df[["도시", "구", "가해운전자 연령"]]
    
    # cnt 변수
    old_count["cnt"]=1
    
    # 구별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령=1) 합계 
    old_count = old_count.groupby(["도시", "구"]).sum().reset_index()
    
    # 노인 사고 / 전체 사고
    old_count["ratio"]=old_count["가해운전자 연령"]/old_count["cnt"]
    
    # 노인 사고 평균 ECLO 
    old_eclo=df[df["가해운전자 연령"]==1][["도시", "구", "ECLO"]]
    old_eclo = old_eclo.groupby(["도시", "구"]).mean().reset_index()
    
    # 구별 전체 사고건수, 노인 사고건수, 노인 평균 ECLO 병합
    temp=pd.merge(old_count, old_eclo, how="left", on=["도시", "구"])
    # nan -> 0
    temp.fillna(0, inplace=True)
    
    # (노인 사고 / 전체 사고)*노인 사고 평균 ECLO 
    temp["가해노인운전자 위험도"]=temp["ratio"]*temp["ECLO"]
    
    temp.drop(["가해운전자 연령", "cnt", "ratio", "ECLO"], axis=1, inplace=True)

    df = pd.merge(df, temp, how="left", on=["도시", "구"])
    test_eng = pd.merge(test_eng, temp, how="left", on=["도시", "구"])

    ##############################################################################################
    # == 가해운전자 차종 별 위험도 ==
    # danger : 컬럼 별 위험도(ECLO)를 측정 후 "구" 별로 컬럼별 발생 비율을 반영한 가중치 계산
    danger1=danger(df, "가해운전자 차종")

    df = pd.merge(df, danger1, how="left", on=["도시", "구"])
    test_eng = pd.merge(test_eng, danger1, how="left", on=["도시", "구"])

    ##############################################################################################
    # == 칼럼 동기화 ==
    test_eng = test_eng.drop(columns=["ID"])
    
    ytrain = df["ECLO"]
    df = df[test_eng.columns]
    """ test_eng.columns
    '요일', '도로형태', '노면상태', '연', '월', '일', '시간', 'Holiday', '도시', '구', '동', '도로형태1',
    '가해운전자 평균연령', '가해운전자 평균성별', "가해노인운전자 위험도", "가해운전자 차종_dangerous"
    """

    ##############################################################################################
    # == 원핫 인코딩 ==
    ## '노면상태',"도로형태1"
    one_hot_features=['노면상태',"도로형태1"]

    train_oh=pd.get_dummies(df[one_hot_features])
    test_oh=pd.get_dummies(test_eng[one_hot_features])

    for i in train_oh.columns:
        if i not in test_oh.columns:
            test_oh[i]=0
    for i in test_oh.columns:
        if i not in train_oh.columns:
            train_oh[i]=0

    print("[원핫인코딩] test_oh 컬럼수 : ", len(test_oh.columns), "/ train_oh 컬럼수 : ", len(train_oh.columns))

    # 원데이터 드롭
    df.drop(one_hot_features,axis=1,inplace=True)
    test_eng.drop(one_hot_features,axis=1,inplace=True)

    # 원핫인코딩 결과 병합
    df=pd.concat([df,train_oh],axis=1)
    test_eng=pd.concat([test_eng,test_oh],axis=1)

    ##############################################################################################
    # == 일 드롭 ==
    df=df.drop(columns=["일"])
    
    ##############################################################################################
    # == 레이블 인코딩
    ## "요일", "도시", "구", "동", "도로형태"
    label_features = ["요일", "도시", "구", "동", "도로형태"]

    for i in label_features:
        print("[레이블인코딩] ", i)
        le = LabelEncoder()
        le=le.fit(df[i])
        df[i]=le.transform(df[i])

        for case in np.unique(test_eng[i]):
            if case not in le.classes_:
                print(case, ' : test case is not in classes')
                le.classes_ = np.append(le.classes_, case)
        test_eng[i]=le.transform(test_eng[i])

    # == 타겟 인코딩
    return df, ytrain ,test_eng

#%%
train1.columns
"""
Index(['ID', '요일', '도로형태', '노면상태', '사고유형', '사고유형 - 세부분류', '법규위반', '가해운전자 차종',
       '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별',
       '피해운전자 연령', '피해운전자 상해정도', '사망자수', '중상자수', '경상자수', '부상자수', 'ECLO', '연',
       '월', '일', '시간', 'Holiday', '도시', '구', '동', '도로형태1'],
      dtype='object')
"""
test1.columns
"""
Index(['ID', '요일', '도로형태', '노면상태', '사고유형', '연', '월', '일', '시간', 'Holiday',
       '도시', '구', '동', '도로형태1'],
      dtype='object')
"""


X_train_eng1, y_train1, X_test_eng1 = feat_eng1(train1)
"""
outlier 수 :  42
train_df 데이터 수 :  26575
[원핫인코딩] test_oh 컬럼수 :  11 / train_oh 컬럼수 :  11
[레이블인코딩]  요일
[레이블인코딩]  도시
[레이블인코딩]  구
[레이블인코딩]  동
남일동  : test case is not in classes
노곡동  : test case is not in classes
노변동  : test case is not in classes
동인동3가  : test case is not in classes
매여동  : test case is not in classes
수성동4가  : test case is not in classes
태평로3가  : test case is not in classes
[레이블인코딩]  도로형태
"""

np.unique(train1["동"])
np.unique(test1["동"])
"""
array(['가창면', '가천동', '각산동', '감삼동', '검단동', '고성동3가', '관음동', '구암동', '구지면',
       '국우동', '금강동', '금호동', '남산동', '남일동', '내당동', '노곡동', '노변동', '노원동3가',
       '논공읍', '다사읍', '대곡동', '대명동', '대봉동', '대신동', '대현동', '덕산동', '도동',
       '도원동', '도학동', '동산동', '동인동1가', '동인동3가', '동인동4가', '동천동', '동호동',
       '두류동', '두산동', '만촌동', '매여동', '매천동', '매호동', '문화동', '방촌동', '범물동',
       '범어동', '복현동', '본동', '본리동', '봉덕동', '봉무동', '불로동', '비산동', '산격동',
       '삼덕동', '삼덕동1가', '삼덕동2가', '상동', '상리동', '상서동', '상인동', '서변동', '서호동',
       '성당동', '성동', '송정동', '송현동', '수성동1가', '수성동4가', '숙천동', '시지동', '신기동',
       '신당동', '신매동', '신암동', '신천동', '연호동', '옥포읍', '용계동', '용산동', '용수동',
       '월성동', '유가읍', '유천동', '율하동', '읍내동', '이곡동', '이천동', '이현동', '인교동',
       '입석동', '장기동', '조야동', '중대동', '중동', '중리동', '지묘동', '지산동', '진천동',
       '칠성동1가', '칠성동2가', '침산동', '태전동', '태평로1가', '태평로2가', '태평로3가', '파동',
       '파호동', '팔달동', '평리동', '현풍읍', '화원읍', '황금동', '효목동'], dtype=object)
"""


#%%
# 차대차
def feat_eng2(df) :
    global test2
    
    # == 이상치 1% 제거
    sns.boxplot(x="ECLO", data=df)
    plt.show()
    
    df_copy = df.copy()
    df_copy = df_copy[[ '노면상태', '사고유형', '사고유형 - 세부분류', '도로형태', 'ECLO']]
    df_copy.reset_index(inplace=True, drop=True)

    outlier_idxs = []
    for col in df_copy.columns[:-1]:
        temp_df = pd.DataFrame(df_copy.groupby(col)['ECLO'].agg(set_limit)).reset_index()
        for j in range(len(temp_df)):
            s_idxs = df[(df[col] == temp_df.loc[j, col]) & (df['ECLO'] > temp_df.loc[j, 'ECLO'])].index.to_list()
            outlier_idxs = outlier_idxs + s_idxs
    outliers = list(set(outlier_idxs))

    print('outlier 수 : ', len(outliers))

    df = df.drop(outliers, axis=0)
    df.reset_index(inplace=True, drop=True)

    print("train_df 데이터 수 : ", len(df))
    
    sns.boxplot(x="ECLO", data=df)
    plt.show()

    # == 사고유형 -> drop
    df.drop(["사고유형"], axis=1, inplace=True)

    # == 지역별 가해운전자 & 피해운전자 평균 연령 추출
    age_mean = df[['도시','구','동','가해운전자 연령', "피해운전자 연령"]].groupby(['도시', '구','동']).mean()
    age_mean.columns = ['가해운전자 평균연령', "피해운전자 평균연령"]
    
    df = pd.merge(df, age_mean, how="left", on=["도시", "구", "동"])
    
    test_eng = pd.merge(test2, age_mean, how="left", on=["도시", "구", "동"]).drop(columns=["사고유형"])
    test_eng[['가해운전자 평균연령','피해운전자 평균연령']]=test_eng[['가해운전자 평균연령','피해운전자 평균연령']].fillna(test_eng[['가해운전자 평균연령','피해운전자 평균연령']].mean())

    # == 지역별 가해운전자 & 피해운전자 평균 성별 추출
    df['가해운전자 성별'] = df['가해운전자 성별'].apply(lambda x:sex_transform(x))
    df['피해운전자 성별'] = df['피해운전자 성별'].apply(lambda x:sex_transform(x))
    
    sex_mean = df[['도시','구','동','가해운전자 성별','피해운전자 성별']].groupby(['도시', '구','동']).mean()
    sex_mean.columns = ['가해운전자 평균성별','피해운전자 평균성별']
    
    df = pd.merge(df, sex_mean, how="left", on=["도시", "구", "동"])
    
    test_eng = pd.merge(test_eng, sex_mean, how="left", on=["도시", "구", "동"])
    test_eng[['가해운전자 평균성별','피해운전자 평균성별']]=test_eng[['가해운전자 평균성별','피해운전자 평균성별']].fillna(test_eng[['가해운전자 평균성별','피해운전자 평균성별']].mean())

    ##############################################################################################
    # == 가해노인운전자 위험도 ==
    # 가해운전자 연령 : 65세 이상 1, 그외 0
    df["가해운전자 연령"]=df["가해운전자 연령"].apply(sepa)
    
    old_count = df[["도시", "구", "가해운전자 연령"]]
    
    # cnt 변수
    old_count["cnt"]=1
    
    # 구별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령=1) 합계 
    old_count = old_count.groupby(["도시", "구"]).sum().reset_index()
    
    # 노인 사고 / 전체 사고
    old_count["ratio"]=old_count["가해운전자 연령"]/old_count["cnt"]
    
    # 노인 사고 평균 ECLO 
    old_eclo=df[df["가해운전자 연령"]==1][["도시", "구", "ECLO"]]
    old_eclo = old_eclo.groupby(["도시", "구"]).mean().reset_index()
    
    # 구별 전체 사고건수, 노인 사고건수, 노인 평균 ECLO 병합
    temp=pd.merge(old_count, old_eclo, how="left", on=["도시", "구"])
    # nan -> 0
    temp.fillna(0, inplace=True)
    
    # (노인 사고 / 전체 사고)*노인 사고 평균 ECLO 
    temp["가해노인운전자 위험도"]=temp["ratio"]*temp["ECLO"]
    
    temp.drop(["가해운전자 연령", "cnt", "ratio", "ECLO"], axis=1, inplace=True)
    
    df = pd.merge(df, temp, how="left", on=["도시", "구"])
    test_eng = pd.merge(test_eng, temp, how="left", on=["도시", "구"])
    
    ##############################################################################################
    # == 피해노인운전자 위험도 ==
    # 피해운전자 연령 : 65세 이상 1, 그외 0
    df["피해운전자 연령"]=df["피해운전자 연령"].apply(sepa)
    
    old_count1 = df[["도시", "구", "피해운전자 연령"]]
    
    # cnt 변수
    old_count1["cnt"]=1
    
    # 구별 전체 사고건수(cnt), 노인 사고건수(피해운전자 연령=1) 합계 
    old_count1 = old_count1.groupby(["도시", "구"]).sum().reset_index()
    
    # 노인 사고 / 전체 사고
    old_count1["ratio"]=old_count1["피해운전자 연령"]/old_count1["cnt"]
    
    # 노인 사고 평균 ECLO 
    old_eclo1=df[df["피해운전자 연령"]==1][["도시", "구", "ECLO"]]
    old_eclo1 = old_eclo1.groupby(["도시", "구"]).mean().reset_index()
    
    # 구별 전체 사고건수, 노인 사고건수, 노인 평균 ECLO 병합
    temp1 = pd.merge(old_count1, old_eclo1, how="left", on=["도시", "구"])
    # nan -> 0
    temp1.fillna(0, inplace=True)
    
    # (노인 사고 / 전체 사고)*노인 사고 평균 ECLO 
    temp1["피해노인운전자 위험도"]=temp1["ratio"]*temp1["ECLO"]
    
    temp1.drop(["피해운전자 연령", "cnt", "ratio", "ECLO"], axis=1, inplace=True)
    
    df = pd.merge(df, temp1, how="left", on=["도시", "구"])
    test_eng = pd.merge(test_eng, temp1, how="left", on=["도시", "구"])
    
    ##############################################################################################
    # 가해운전자 차종 별 위험도
    danger21=danger(df, "가해운전자 차종")

    df = pd.merge(df, danger21, how="left", on=["도시", "구"])
    test_eng = pd.merge(test_eng, danger21, how="left", on=["도시", "구"])
    
    # 피해운전자 차종 별 위험도
    danger22=danger(df, "피해운전자 차종")

    df = pd.merge(df, danger22, how="left", on=["도시", "구"])
    test_eng = pd.merge(test_eng, danger22, how="left", on=["도시", "구"])

    # == 칼럼 동기화
    test_eng = test_eng.drop(columns=["ID"])
    ytrain = df["ECLO"]
    df = df[test_eng.columns]

    # == 원핫 인코딩
    one_hot_features=['노면상태', "도로형태1"]

    train_oh=pd.get_dummies(df[one_hot_features])
    test_oh=pd.get_dummies(test_eng[one_hot_features])

    for i in train_oh.columns:
        if i not in test_oh.columns:
            test_oh[i]=0
    for i in test_oh.columns:
        if i not in train_oh.columns:
            train_oh[i]=0

    print("[원핫인코딩] test_oh 컬럼수 : ", len(test_oh.columns), "/ train_oh 컬럼수 : ", len(train_oh.columns))

    df.drop(one_hot_features,axis=1,inplace=True)
    test_eng.drop(one_hot_features,axis=1,inplace=True)

    df=pd.concat([df,train_oh],axis=1)
    test_eng=pd.concat([test_eng,test_oh],axis=1)

    df=df.drop(columns=["일"])
    
    # == 레이블 인코딩
    label_features = ["요일", "도시", "구", "동", "도로형태"]

    for i in label_features:
        print("[레이블인코딩] ", i)
        le = LabelEncoder()
        le=le.fit(df[i])
        df[i]=le.transform(df[i])

        for case in np.unique(test_eng[i]):
            if case not in le.classes_:
                print(case, ' : test case is not in classes')
                le.classes_ = np.append(le.classes_, case)
        test_eng[i]=le.transform(test_eng[i])

    # == 타겟 인코딩
    return df, ytrain, test_eng

#%%
X_train_eng2, y_train2, X_test_eng2=feat_eng2(train2)
"""
outlier 수 :  634
train_df 데이터 수 :  489180
[원핫인코딩] test_oh 컬럼수 :  11 / train_oh 컬럼수 :  11
[레이블인코딩]  요일
[레이블인코딩]  도시
[레이블인코딩]  구
[레이블인코딩]  동
[레이블인코딩]  도로형태
"""

#%%
# 차대 사람
def feat_eng3(df) :
    global test3
    
    # == 이상치 1% 제거
    sns.boxplot(x="ECLO", data=df)
    plt.show()

    df_copy = df.copy()
    df_copy = df_copy[[ '노면상태', '사고유형', '사고유형 - 세부분류', '도로형태', 'ECLO']]
    df_copy.reset_index(inplace=True, drop=True)

    outlier_idxs = []
    for col in df_copy.columns[:-1]:
        temp_df = pd.DataFrame(df_copy.groupby(col)['ECLO'].agg(set_limit)).reset_index()
        for j in range(len(temp_df)):
            s_idxs = df[(df[col] == temp_df.loc[j, col]) & (df['ECLO'] > temp_df.loc[j, 'ECLO'])].index.to_list()
            outlier_idxs = outlier_idxs + s_idxs
    outliers = list(set(outlier_idxs))

    print('outlier 수 : ', len(outliers))

    df = df.drop(outliers, axis=0)
    df.reset_index(inplace=True, drop=True)

    print("train_df 데이터 수 : ", len(df))
    
    sns.boxplot(x="ECLO", data=df)
    plt.show()

    # == 사고유형 -> drop
    df.drop(["사고유형"], axis=1, inplace=True)

    # == 지역별 가해운전자 & 피해운전자 평균 연령 추출
    age_mean = df[['도시','구','동','가해운전자 연령', "피해운전자 연령"]].groupby(['도시', '구','동']).mean()
    age_mean.columns = ['가해운전자 평균연령', "피해운전자 평균연령"]
    
    df = pd.merge(df, age_mean, how="left", on=["도시", "구", "동"])
    
    test_eng = pd.merge(test3, age_mean, how="left", on=["도시", "구", "동"]).drop(columns=["사고유형"])
    test_eng[['가해운전자 평균연령','피해운전자 평균연령']]=test_eng[['가해운전자 평균연령','피해운전자 평균연령']].fillna(test_eng[['가해운전자 평균연령','피해운전자 평균연령']].mean())

    # == 지역별 가해운전자 & 피해운전자 평균 성별 추출
    df['가해운전자 성별'] = df['가해운전자 성별'].apply(lambda x:sex_transform(x))
    df['피해운전자 성별'] = df['피해운전자 성별'].apply(lambda x:sex_transform(x))
    
    sex_mean = df[['도시','구','동','가해운전자 성별','피해운전자 성별']].groupby(['도시', '구','동']).mean()
    sex_mean.columns = ['가해운전자 평균성별','피해운전자 평균성별']
    
    df = pd.merge(df, sex_mean, how="left", on=["도시", "구", "동"])
    
    test_eng = pd.merge(test_eng, sex_mean, how="left", on=["도시", "구", "동"])
    test_eng[['가해운전자 평균성별','피해운전자 평균성별']]=test_eng[['가해운전자 평균성별','피해운전자 평균성별']].fillna(test_eng[['가해운전자 평균성별','피해운전자 평균성별']].mean())

    
    ##############################################################################################
    # == 가해노인운전자 위험도 ==
    # 가해운전자 연령 : 65세 이상 1, 그외 0
    df["가해운전자 연령"]=df["가해운전자 연령"].apply(sepa)
    
    old_count = df[["도시", "구", "가해운전자 연령"]]
    
    # cnt 변수
    old_count["cnt"]=1
    
    # 구별 전체 사고건수(cnt), 노인 사고건수(가해운전자 연령=1) 합계 
    old_count = old_count.groupby(["도시", "구"]).sum().reset_index()
    
    # 노인 사고 / 전체 사고
    old_count["ratio"]=old_count["가해운전자 연령"]/old_count["cnt"]
    
    # 노인 사고 평균 ECLO 
    old_eclo=df[df["가해운전자 연령"]==1][["도시", "구", "ECLO"]]
    old_eclo = old_eclo.groupby(["도시", "구"]).mean().reset_index()
    
    # 구별 전체 사고건수, 노인 사고건수, 노인 평균 ECLO 병합
    temp=pd.merge(old_count, old_eclo, how="left", on=["도시", "구"])
    # nan -> 0
    temp.fillna(0, inplace=True)
    
    # (노인 사고 / 전체 사고)*노인 사고 평균 ECLO 
    temp["가해노인운전자 위험도"]=temp["ratio"]*temp["ECLO"]
    
    temp.drop(["가해운전자 연령", "cnt", "ratio", "ECLO"], axis=1, inplace=True)
    
    df = pd.merge(df, temp, how="left", on=["도시", "구"])
    test_eng = pd.merge(test_eng, temp, how="left", on=["도시", "구"])
    
    ##############################################################################################
    # 가해운전자 차종 별 위험도
    danger31=danger(df, "가해운전자 차종")

    df = pd.merge(df, danger31, how="left", on=["도시", "구"])
    test_eng = pd.merge(test_eng, danger31, how="left", on=["도시", "구"])

    # == 칼럼 동기화
    test_eng = test_eng.drop(columns=["ID"])
    ytrain = df["ECLO"]
    df = df[test_eng.columns]

    # == 원핫 인코딩
    one_hot_features=['노면상태', "도로형태1"]

    train_oh=pd.get_dummies(df[one_hot_features])
    test_oh=pd.get_dummies(test_eng[one_hot_features])

    for i in train_oh.columns:
        if i not in test_oh.columns:
            test_oh[i]=0
    for i in test_oh.columns:
        if i not in train_oh.columns:
            train_oh[i]=0

    print("[원핫인코딩] test_oh 컬럼수 : ", len(test_oh.columns), "/ train_oh 컬럼수 : ", len(train_oh.columns))

    df.drop(one_hot_features,axis=1,inplace=True)
    test_eng.drop(one_hot_features,axis=1,inplace=True)

    df=pd.concat([df,train_oh],axis=1)
    test_eng=pd.concat([test_eng,test_oh],axis=1)

    df=df.drop(columns=["일"])
    
    # == 레이블 인코딩
    label_features = ["요일", "도시", "구", "동", "도로형태"]

    for i in label_features:
        print("[레이블인코딩] ", i)
        le = LabelEncoder()
        le=le.fit(df[i])
        df[i]=le.transform(df[i])

        for case in np.unique(test_eng[i]):
            if case not in le.classes_:
                print(case, ' : test case is not in classes')
                le.classes_ = np.append(le.classes_, case)
        test_eng[i]=le.transform(test_eng[i])

    # == 타겟 인코딩
    return df, ytrain, test_eng

#%%
X_train_eng3, y_train3, X_test_eng3=feat_eng3(train3)
"""
outlier 수 :  125
train_df 데이터 수 :  112976
[원핫인코딩] test_oh 컬럼수 :  11 / train_oh 컬럼수 :  11
[레이블인코딩]  요일
[레이블인코딩]  도시
[레이블인코딩]  구
[레이블인코딩]  동
능성동  : test case is not in classes
매여동  : test case is not in classes
완전동  : test case is not in classes
하서동  : test case is not in classes
[레이블인코딩]  도로형태
"""

#%% 결측치 확인
print(X_train_eng1.isnull().sum().sum())
print(X_test_eng1.isnull().sum().sum())
print(y_train1.isnull().sum().sum())
print(X_train_eng2.isnull().sum().sum())
print(X_test_eng2.isnull().sum().sum())
print(y_train2.isnull().sum().sum())
print(X_train_eng3.isnull().sum().sum())
print(X_test_eng3.isnull().sum().sum())
print(y_train3.isnull().sum().sum())

#%%
print(X_train_eng1.columns)
"""
Index(['요일', '도로형태', '연', '월', '시간', 'Holiday', '도시', '구', '동', '가해운전자 평균연령',
       '가해운전자 평균성별', '가해노인운전자 위험도', '가해운전자 차종_dangerous', '노면상태_건조', '노면상태_기타',
       '노면상태_서리/결빙', '노면상태_적설', '노면상태_젖음/습기', '노면상태_침수', '도로형태1_교차로',
       '도로형태1_기타', '도로형태1_단일로', '도로형태1_미분류', '도로형태1_주차장'],
      dtype='object')
"""
print(X_train_eng2.columns)
"""
Index(['요일', '도로형태', '연', '월', '시간', 'Holiday', '도시', '구', '동', '가해운전자 평균연령',
       '피해운전자 평균연령', '가해운전자 평균성별', '피해운전자 평균성별', '가해노인운전자 위험도', '피해노인운전자 위험도',
       '가해운전자 차종_dangerous', '피해운전자 차종_dangerous', '노면상태_건조', '노면상태_기타',
       '노면상태_서리/결빙', '노면상태_적설', '노면상태_젖음/습기', '노면상태_침수', '도로형태1_교차로',
       '도로형태1_기타', '도로형태1_단일로', '도로형태1_미분류', '도로형태1_주차장'],
      dtype='object')
"""
print(X_train_eng3.columns)
"""
Index(['요일', '도로형태', '연', '월', '시간', 'Holiday', '도시', '구', '동', '가해운전자 평균연령',
       '피해운전자 평균연령', '가해운전자 평균성별', '피해운전자 평균성별', '가해노인운전자 위험도',
       '가해운전자 차종_dangerous', '노면상태_건조', '노면상태_기타', '노면상태_서리/결빙', '노면상태_적설',
       '노면상태_젖음/습기', '노면상태_침수', '도로형태1_교차로', '도로형태1_기타', '도로형태1_단일로',
       '도로형태1_미분류', '도로형태1_주차장'],
      dtype='object')
"""

#%% 하이퍼 파라미터 튜닝
from sklearn.model_selection import train_test_split

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
{'num_leaves': 8, 'colsample_bytree': 0.7056196173607632, 'reg_alpha': 0.30507303158163174, 
 'reg_lambda': 5.270276286326108, 'max_depth': 3, 'learning_rate': 0.026633708346045667, 
 'n_estimators': 846, 'min_child_samples': 30, 'subsample': 0.6403247567928743}
"""
#  > 최적화 프로세스에서 발견된 목표 함수의 최상의 값
# = objective에서 계산된 MSLE 손실의 제곱근의 최저값
print(study_lgbm1.best_value)
### 0.19968321719767126

# Optuna 연구 객체 study_lgbm2
print(study_lgbm2.best_params)
"""
{'num_leaves': 15, 'colsample_bytree': 0.9541571786528705, 'reg_alpha': 0.6628645292437817, 
 'reg_lambda': 0.20954147350634456, 'max_depth': 7, 'learning_rate': 0.07908017015330027, 
 'n_estimators': 754, 'min_child_samples': 35, 'subsample': 0.7878281692508873}
"""
print(study_lgbm2.best_value)
### 0.16526418737304982

# Optuna 연구 객체 study_lgbm3
print(study_lgbm3.best_params)
"""
{'num_leaves': 7, 'colsample_bytree': 0.8405004430911097, 'reg_alpha': 0.3487963064160773, 
 'reg_lambda': 0.256191395874264, 'max_depth': 4, 'learning_rate': 0.055470645378579674, 
 'n_estimators': 895, 'min_child_samples': 35, 'subsample': 0.903963628661378}
"""
print(study_lgbm3.best_value)
### 0.12744852427534145

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
{'iterations': 7622, 'od_wait': 1175, 'learning_rate': 0.010789422579476505, 'reg_lambda': 0.023179090122379398, 
 'subsample': 0.8449323656584777, 'random_strength': 23.226909367725796, 'depth': 7, 'min_data_in_leaf': 10, 
 'leaf_estimation_iterations': 8, 'bagging_temperature': 0.016159916374179493, 'colsample_bylevel': 0.6834512059856851}
"""
print(study_catboost1.best_value)
### 0.19964526114075148

# Optuna 연구 객체 study_catboost2
print(study_catboost2.best_params)
"""
{'iterations': 5145, 'od_wait': 853, 'learning_rate': 0.03628064334357001, 'reg_lambda': 0.8946686553083643, 
 'subsample': 0.7115981309278958, 'random_strength': 21.429619597681395, 'depth': 6, 'min_data_in_leaf': 15, 
 'leaf_estimation_iterations': 5, 'bagging_temperature': 0.3424148436951217, 'colsample_bylevel': 0.6905553068132618}
"""
print(study_catboost2.best_value)
### 0.16525750833626787

# Optuna 연구 객체 study_catboost3
print(study_catboost3.best_params)
"""
{'iterations': 1943, 'od_wait': 1083, 'learning_rate': 0.03742147642190215, 'reg_lambda': 4.3575812498465885, 
 'subsample': 0.7993142769258825, 'random_strength': 28.07481879970023, 'depth': 6, 'min_data_in_leaf': 1, 
 'leaf_estimation_iterations': 2, 'bagging_temperature': 1.809216772005867, 'colsample_bylevel': 0.6546465429000649}
"""
print(study_catboost3.best_value)
### 0.1274512406313261

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
Index(['요일', '도로형태', '연', '월', '시간', 'Holiday', '도시', '구', '동', '가해운전자 평균연령',
       '가해운전자 평균성별', '가해노인운전자 위험도', '가해운전자 차종_dangerous', '노면상태_건조', '노면상태_기타',
       '노면상태_서리/결빙', '노면상태_적설', '노면상태_젖음/습기', '노면상태_침수', '도로형태1_교차로',
       '도로형태1_기타', '도로형태1_단일로', '도로형태1_미분류', '도로형태1_주차장'],
      dtype='object')

Index(['요일', '도로형태', '연', '월', '일', '시간', 'Holiday', '도시', '구', '동',
       '가해운전자 평균연령', '가해운전자 평균성별', '가해노인운전자 위험도', '가해운전자 차종_dangerous',
       '노면상태_건조', '노면상태_기타', '노면상태_서리/결빙', '노면상태_젖음/습기', '노면상태_침수',
       '도로형태1_교차로', '도로형태1_기타', '도로형태1_단일로', '노면상태_적설', '도로형태1_미분류',
       '도로형태1_주차장'],
      dtype='object')
"""
X_test_eng1=X_test_eng1.drop(columns=["일"])
X_test_eng2=X_test_eng2.drop(columns=["일"])
X_test_eng3=X_test_eng3.drop(columns=["일"])

lgbm_prediction1 = np.expm1(lgbm1.predict(X_test_eng1))
lgbm_prediction2 = np.expm1(lgbm2.predict(X_test_eng2))
lgbm_prediction3 = np.expm1(lgbm3.predict(X_test_eng3))
"""
LGBM 모델은 특징 순서에 민감하며 불일치로 인해 이 오류가 발생할 수 있습니다.
>> ValueError: Number of features of the model must match the input. Model n_features_ is 26 and input n_features is 27
"""

catboost_prediction1 = np.expm1(catboost1.predict(X_test_eng1))
catboost_prediction2 = np.expm1(catboost2.predict(X_test_eng2))
catboost_prediction3 = np.expm1(catboost3.predict(X_test_eng3))

#%% 앙상블 : lgbm_prediction*0.2 + catboost_prediction*0.8
test1["predict"]=lgbm_prediction1*0.2+catboost_prediction1*0.8
test2["predict"]=lgbm_prediction2*0.2+catboost_prediction2*0.8
test3["predict"]=lgbm_prediction3*0.2+catboost_prediction3*0.8

#%%
# 원데이터(test_df)에 id를 기준으로 차량단독 예측값 병합
test_f1=pd.merge(test_df, test1[["ID","predict"]], how="left", on="ID" )
test_f1["predict"]=test_f1["predict"].fillna(0)

# test_f1에 id를 기준으로 차대차 예측값 병합
test_f2=pd.merge(test_f1, test2[["ID","predict"]], how="left", on="ID" )
# > 차량단독 예측값 컬럼명 predict_x, 차대차 예측값 컬럼명 predict_y로 자동 변환
test_f2["predict_y"]=test_f2["predict_y"].fillna(0)

# test_f2에 id를 기준으로 차대사람 예측값 병합
test_f3=pd.merge(test_f2, test3[["ID","predict"]], how="left", on="ID" )
test_f3["predict"]=test_f3["predict"].fillna(0)

# 각 예측값 더함 / 없는 값은 0으로 바꿨으므로 합계가 각각의 예측값과 동일
test_f3["predict"]= test_f3["predict_x"]+test_f3["predict_y"]+test_f3["predict"]

# 최종 제출결과
ss['ECLO'] = test_f3["predict"]
"""
                   ID      ECLO
0      ACCIDENT_39609  3.719355
1      ACCIDENT_39610  3.740126
2      ACCIDENT_39611  5.045868
3      ACCIDENT_39612  4.434415
4      ACCIDENT_39613  4.664393
              ...       ...
10958  ACCIDENT_50567  5.455761
10959  ACCIDENT_50568  4.421505
10960  ACCIDENT_50569  4.465885
10961  ACCIDENT_50570  4.461500
10962  ACCIDENT_50571  4.794541

[10963 rows x 2 columns]
"""