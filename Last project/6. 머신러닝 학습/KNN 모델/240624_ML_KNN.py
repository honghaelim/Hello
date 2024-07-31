# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:51:19 2024

@author: ksj
"""

import pandas as pd

## 2018~2023년 수원 전체 교통사고 현황
df = pd.read_excel('E:/Workspace/!project_team/4.18-23수원교통사고/accidentInfoList_18-23.xlsx')

df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 28460 entries, 0 to 28459
Data columns (total 30 columns):
 #   Column      Non-Null Count  Dtype         
---  ------      --------------  -----         
 0   Unnamed: 0  28460 non-null  int64         
 1   index       28460 non-null  int64         
 2   사고번호        28460 non-null  int64         
 3   사고일시        28460 non-null  datetime64[ns]
- 4   요일          28460 non-null  object        
- 5   시군구         28460 non-null  object               
 6   사고내용        28460 non-null  object        
* 7   사망자수        28460 non-null  int64         
* 8   중상자수        28460 non-null  int64         
* 9   경상자수        28460 non-null  int64         
* 10  부상신고자수      28460 non-null  int64         
 11  사고유형        28460 non-null  object        
 12  법규위반        28460 non-null  object        
- 13  노면상태        28460 non-null  object        
- 14  기상상태        28460 non-null  object        
- 15  도로형태        28460 non-null  object        
- 16  가해운전자 차종    28460 non-null  object        
- 17  가해운전자 성별    28460 non-null  object        
- 18  가해운전자 연령    28460 non-null  int64         
 19  가해운전자 상해정도  28460 non-null  object        
 20  피해운전자 차종    27614 non-null  object        
 21  피해운전자 성별    27614 non-null  object        
 22  피해운전자 연령    27614 non-null  object        
 23  피해운전자 상해정도  27614 non-null  object        
 24  연           28460 non-null  int64         
 25  월           28460 non-null  int64         
 26  일           28460 non-null  int64         
- 27  시간          28460 non-null  int64         
 28  구           28460 non-null  object        
 29  동           28460 non-null  object        
dtypes: datetime64[ns](1), int64(12), object(17)
memory usage: 6.5+ MB
"""

#%%
## 라벨인코딩
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

onehot_location = label_encoder.fit_transform(df['시군구'])
onehot_weathre = label_encoder.fit_transform(df['기상상태'])
onehot_surface = label_encoder.fit_transform(df['노면상태'])
onehot_road = label_encoder.fit_transform(df['도로형태'])
onehot_car = label_encoder.fit_transform(df['가해운전자 차종'])
onehot_sex = label_encoder.fit_transform(df['가해운전자 성별'])
onehot_week = label_encoder.fit_transform(df['요일'])

df['location'] = onehot_location
df['weather'] = onehot_weathre
df['surface'] = onehot_surface
df['road'] = onehot_road
df['car'] = onehot_car
df['sex'] = onehot_sex
df['week'] = onehot_week

## ECLO 계산
df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1

#%%
from sklearn.model_selection import train_test_split

# 독립변수 선택
X = df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령', 'week', '시간']]

# 종속변수 ECLO
y = df['ECLO']

# 설명변수 데이터를 정규화
## 설명(독립)변수 열들이 갖는 데이터의 상대적 크기 차이 없애기
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data와 test data 구분(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

print('훈련 데이터 : ', X_train.shape)
print('검증 데이터 : ', X_test.shape)
"""
훈련 데이터 :  (22768, 9)
검증 데이터 :  (5692, 9)
"""

#%%
from sklearn.neighbors import KNeighborsClassifier

# 모형 객체 생성(K = 5로 설정)
knn = KNeighborsClassifier(n_neighbors=5)

# train data를 가지고 모형 학습
knn.fit(X_train, y_train)

# test data를 가지고 y_hat 예측(분류)
y_hat = knn.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])
"""
[5 3 1 3 3 5 3 3 3 3]
[3 6 1 8 3 6 3 6 1 3]
"""

#%%
# 모델 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix)
"""
[[  33    1  236    1   18    6    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   2    0   11    0    2    2    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [ 121    6 2479    7  202  127    0    8   10    2    1    1    0    0
     2    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   8    0   74    0    5    5    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [  48    3  792    3   76   36    1    2    2    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [  21    1  529    2   57   27    0    1    3    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   1    0   26    0    1    2    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   9    0  113    0   20    7    0    1    1    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   3    1  168    0   16    8    0    2    2    1    0    0    1    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   3    0   49    0    5    3    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   3    0   36    0    7    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   1    0   76    0    5    2    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   2    0   18    0    3    2    0    0    1    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   1    0   12    0    2    3    0    0    1    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   4    0   27    0    3    4    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0   12    0    1    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    8    0    1    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   1    0   14    0    1    1    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    2    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    6    0    0    0    0    1    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    3    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    2    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    1    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    3    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   1    0    1    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    1    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    1    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]
 [   0    0    1    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0]]
"""

# 모형 성능 평가 - 평가 지표 계산
knn_report = metrics.classification_report(y_test, y_hat)
print(knn_report)
"""
              precision    recall  f1-score   support

           1       0.13      0.11      0.12       295
           2       0.00      0.00      0.00        17
           3       0.53      0.84      0.65      2966
           4       0.00      0.00      0.00        92
           5       0.18      0.08      0.11       963
           6       0.11      0.04      0.06       641
           7       0.00      0.00      0.00        30
           8       0.07      0.01      0.01       151
           9       0.10      0.01      0.02       202
          10       0.00      0.00      0.00        60
          11       0.00      0.00      0.00        46
          12       0.00      0.00      0.00        84
          13       0.00      0.00      0.00        26
          14       0.00      0.00      0.00        19
          15       0.00      0.00      0.00        38
          16       0.00      0.00      0.00        13
          17       0.00      0.00      0.00         9
          18       0.00      0.00      0.00        17
          19       0.00      0.00      0.00         2
          20       0.00      0.00      0.00         7
          21       0.00      0.00      0.00         3
          23       0.00      0.00      0.00         2
          27       0.00      0.00      0.00         1
          28       0.00      0.00      0.00         3
          29       0.00      0.00      0.00         2
          30       0.00      0.00      0.00         1
          35       0.00      0.00      0.00         1
          70       0.00      0.00      0.00         1

    accuracy                           0.46      5692
   macro avg       0.04      0.04      0.03      5692
weighted avg       0.33      0.46      0.37      5692
"""
## ECLO 값이 1~70까지여서 각각에 해당하는 정확도 등 계산

#%%
# 함수화
def ml_knn(df, col_ls) :
    X = df[col_ls]
    y = df['ECLO']

    X = preprocessing.StandardScaler().fit(X).transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    print('훈련 데이터 : ', X_train.shape)
    print('검증 데이터 : ', X_test.shape)
    
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)

    y_hat = knn.predict(X_test)

    knn_report = metrics.classification_report(y_test, y_hat)
    print(knn_report)
    
#%%
# ECLO 8 이하인 데이터 기준
ndf = df.loc[df['ECLO'] <= 8, :] 

ml_knn(ndf, ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령', '시간'])    
"""
              precision    recall  f1-score   support

           1       0.17      0.10      0.12       316
           2       0.00      0.00      0.00        20
           3       0.58      0.86      0.69      2947
           4       0.00      0.00      0.00        91
           5       0.19      0.08      0.11       926
           6       0.13      0.03      0.05       659
           7       0.00      0.00      0.00        32
           8       0.12      0.01      0.01       148

    accuracy                           0.52      5139
   macro avg       0.15      0.13      0.12      5139
weighted avg       0.40      0.52      0.43      5139
"""

#%%
# 노인 운전자 데이터 기준
old_df = df.loc[df['가해운전자 연령'] >= 65,:]

ml_knn(old_df, ['location', 'weather', 'road', 'car', 'sex', '가해운전자 연령', '시간', 'week'])
"""
              precision    recall  f1-score   support

           1       0.12      0.10      0.11        39
           2       0.00      0.00      0.00         1
           3       0.54      0.87      0.67       352
           4       0.00      0.00      0.00         7
           5       0.22      0.06      0.09       108
           6       0.12      0.04      0.07        68
           8       0.50      0.06      0.10        18
           9       0.00      0.00      0.00        22
          10       0.00      0.00      0.00        10
          11       0.00      0.00      0.00         2
          12       0.00      0.00      0.00        10
          13       0.00      0.00      0.00         6
          14       0.00      0.00      0.00         1
          15       0.00      0.00      0.00         4
          18       0.00      0.00      0.00         1
          19       0.00      0.00      0.00         1
          20       0.00      0.00      0.00         1
          23       0.00      0.00      0.00         1
          27       0.00      0.00      0.00         1
          30       0.00      0.00      0.00         1

    accuracy                           0.49       654
   macro avg       0.08      0.06      0.05       654
weighted avg       0.36      0.49      0.39       654
"""
