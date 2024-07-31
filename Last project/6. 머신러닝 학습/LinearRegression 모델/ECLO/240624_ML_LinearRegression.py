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
- 3   사고일시        28460 non-null  datetime64[ns]
 4   요일          28460 non-null  object        
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
 27  시간          28460 non-null  int64         
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

df['location'] = onehot_location
df['weather'] = onehot_weathre
df['surface'] = onehot_surface
df['road'] = onehot_road
df['car'] = onehot_car
df['sex'] = onehot_sex

## ECLO 계산
df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1

#%%
from sklearn.model_selection import train_test_split

# 독립변수 선택
X = df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']]
"""
X = df[['사고일시', 'location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']]

lr.fit(X_train, y_train)
>> 에러발생
DTypePromotionError: The DType <class 'numpy.dtypes.DateTime64DType'> could not be promoted by <class 'numpy.dtypes.Int64DType'>. 
This means that no common DType exists for the given inputs. 
For example they cannot be stored in a single array unless the dtype is `object`. 
The full list of DTypes is: (<class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int32DType'>, <class 'numpy.dtypes.Int32DType'>, 
                             <class 'numpy.dtypes.Int32DType'>, <class 'numpy.dtypes.Int32DType'>, <class 'numpy.dtypes.Int32DType'>, 
                             <class 'numpy.dtypes.Int32DType'>, <class 'numpy.dtypes.Int64DType'>)
"""
# 종속변수 ECLO
y = df['ECLO']

# train data와 test data 구분(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

print('훈련 데이터 : ', X_train.shape)
print('검증 데이터 : ', X_test.shape)
"""
훈련 데이터 :  (22768, 7)
검증 데이터 :  (5692, 7)
"""

#%%
from sklearn.linear_model import LinearRegression

# train data를 가지고 모형 학습
lr = LinearRegression()

lr.fit(X_train, y_train)

# 학습을 마친 모형에 test data 적용하여 결정계수(R-제곱) 계산
## 결정계수 값이 클수록 모형의 예측 능력이 좋다고 판단
r_square = lr.score(X_test, y_test)
print(r_square)
## 0.0024016319984759837

# 회귀식의 기울기 및 절편 확인
print('X 변수의 계수 a :', lr.coef_)
## X 변수의 계수 a : [-0.00605218 -0.02745732  0.03010239 -0.01821668 -0.02235062 -0.01140147  0.00280166]
print('상수항 b : ', lr.intercept_)
## 상수항 b :  4.8578515731893575

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

# 모형에 전체 데이터 X를 입력하여 예측한 값 y_hat을 실제 값 y와 비교
y_hat = lr.predict(X)

plt.figure(figsize = (10,5))
ax1 = sns.kdeplot(y_test, label = 'y_test')
ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
plt.legend()
plt.show()

#%%
## ECLO 범위가 생각보다 넓음
df['ECLO'].unique()
"""
array([ 5,  3,  1, 12, 13,  6, 10,  4, 11, 15,  8,  9,  7, 14,  2, 16, 18,
       17, 52, 31, 24, 23, 19, 21, 28, 20, 22, 38, 48, 26, 27, 43, 39, 44,
       35, 29, 25, 72, 47, 46, 62, 34, 30, 67, 96, 36, 70], dtype=int64)
"""

df['ECLO'].describe()
"""
count    28460.000000
mean         4.586297
std          3.215800
min          1.000000
25%          3.000000
50%          3.000000
75%          5.000000
max         96.000000
Name: ECLO, dtype: float64
"""

#%% ECLO 이상치(75% 값 이상 > 5) 제거
ndf = df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령', 'ECLO']]
eclo_df = ndf.loc[ndf['ECLO'] <= 5,:]

X = eclo_df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']]
y = eclo_df['ECLO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

print('훈련 데이터 : ', X_train.shape)
print('검증 데이터 : ', X_test.shape)
"""
훈련 데이터 :  (17256, 7)
검증 데이터 :  (4314, 7)
"""

lr = LinearRegression()

lr.fit(X_train, y_train)

r_square = lr.score(X_test, y_test)
print(r_square)
## 0.0024064154498751877

print('X 변수의 계수 a :', lr.coef_)
## X 변수의 계수 a : [-0.00116631  0.01267771  0.00162728 -0.00416119 -0.01083867  0.03293624  0.0031771 ]
print('상수항 b : ', lr.intercept_)
## 상수항 b :  3.190326394970098

y_hat = lr.predict(X)

plt.figure(figsize = (10,5))
ax1 = sns.kdeplot(y_test, label = 'y_test')
ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
plt.legend()
plt.show()

#%%
## 함수화
def ml_col(df, col_ls) :
    X = df[col_ls]
    y = df['ECLO']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    print('훈련 데이터 : ', X_train.shape)
    print('검증 데이터 : ', X_test.shape)
    
    print("\n")

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    r_square = lr.score(X_test, y_test)
    print(r_square)

    print("\n")
    
    print('X 변수의 계수 a :', lr.coef_)
    print('상수항 b : ', lr.intercept_)

    y_hat = lr.predict(X)

    plt.figure(figsize = (10,5))
    ax1 = sns.kdeplot(y_test, label = 'y_test')
    ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
    plt.legend()
    plt.show()
    
#%%    
# ECLO 5 이하 데이터에서 X(가해운전자 정보)로만 제한
col_ls = ['car', 'sex', '가해운전자 연령']    

ml_col(eclo_df, col_ls)
## 0.0019038655670952087

#%%
# 노인 운전자 기준 예측
old_df = df.loc[df['가해운전자 연령'] >= 65,:]

ml_col(old_df, ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령'])
## -0.0014010822884187313
