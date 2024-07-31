# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df = pd.read_excel('The elderly driver traffic accidents(suwon).xlsx')
df.info()

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1880 entries, 0 to 1879
Data columns (total 22 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   사고번호        1880 non-null   int64 
 1   사고일시        1880 non-null   object
 2   요일          1880 non-null   object
 3   시군구         1880 non-null   object
 4   사고내용        1880 non-null   object
 5   사망자수        1880 non-null   int64 
 6   중상자수        1880 non-null   int64 
 7   경상자수        1880 non-null   int64 
 8   부상신고자수      1880 non-null   int64 
 9   사고유형        1880 non-null   object
 10  법규위반        1880 non-null   object
 11  노면상태        1880 non-null   object
 12  기상상태        1880 non-null   object
 13  도로형태        1880 non-null   object
 14  가해운전자 차종    1880 non-null   object
 15  가해운전자 성별    1880 non-null   object
 16  가해운전자 연령    1880 non-null   object
 17  가해운전자 상해정도  1880 non-null   object
 18  피해운전자 차종    1833 non-null   object
 19  피해운전자 성별    1833 non-null   object
 20  피해운전자 연령    1833 non-null   object
 21  피해운전자 상해정도  1833 non-null   object
dtypes: int64(5), object(17)
memory usage: 323.3+ KB
'''

# 보니까 non값 있음
'''
 18  피해운전자 차종    1833 non-null   object
 19  피해운전자 성별    1833 non-null   object
 20  피해운전자 연령    1833 non-null   object
 21  피해운전자 상해정도  1833 non-null   object
''' 

#%%
# 날짜처리하기
# [사고일시] -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')  ## 2023-01-01 00:00:00

#   날짜(object)                   
df['날짜'] = df['사고일시'].dt.date                                            ## 2023-01-01
df['날짜2'] = df['사고일시'].dt.date   #  복제
#   시간(int)
df['시간'] = df['사고일시'].dt.hour                                            ## 0

#%%
# 문자열 포맷팅(시군구, 사고유형, 도로형태)
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

#%%
# 결측치 처리하기
# [피해운전자] nan -> 0
""" 
df.iloc[:, 18:22].columns 
Index(['피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도'], dtype='object')
"""

df.iloc[:, 18:22] = df.iloc[:, 18:22].fillna(0)


#%%

# [연령] 00세(object) -> 00(int)
# '가해운전자'
df['가해운전자 연령'].unique()

"""
array(['68세', '66세', '69세', '80세', '75세', '65세', '67세', '79세', '82세',
       '73세', '78세', '70세', '74세', '87세', '72세', '71세', '77세', '76세',
       '83세', '81세', '86세', '85세', '84세', '93세', '88세'], dtype=object)
"""

df['가해운전자 연령'] = df['가해운전자 연령'].str[:-1]

df['가해운전자 연령'] = df['가해운전자 연령'].astype('int64')



# '피해운전자'
df['피해운전자 연령'].unique()
"""
array(['54세', '68세', '25세', '55세', '72세', '50세', '39세', '53세', '45세',
       '47세', '48세', '59세', '33세', '46세', 0, '38세', '42세', '57세', '75세',
       '21세', '43세', '29세', '35세', '84세', '71세', '31세', '66세', '51세',
       '27세', '56세', '64세', '49세', '78세', '32세', '62세', '23세', '73세',
       '19세', '26세', '60세', '40세', '58세', '77세', '30세', '36세', '37세',
       '52세', '18세', '41세', '13세', '74세', '7세', '61세', '44세', '8세', '22세',
       '20세', '63세', '28세', '85세', '65세', '70세', '34세', '24세', '82세',
       '81세', '67세', '80세', '10세', '3세', '83세', '88세', '15세', '17세',
       '86세', '9세', '14세', '69세', '11세', '87세', '미분류', '79세', '16세',
       '92세', '6세', '95세', '2세', '90세', '76세', '12세'], dtype=object)
"""
df['피해운전자 연령'] = df['피해운전자 연령'].str[:-1]
## -> nan(0->nan), '미분'('미분류')

# -> '미분류' : 0
df['피해운전자 연령'] = df['피해운전자 연령'].replace('미분', 0)
# -> nan : 0
df['피해운전자 연령'] = df['피해운전자 연령'].fillna(0)

df['피해운전자 연령'] = df['피해운전자 연령'].astype('int64')

#%%
# 중간점검
df.shape # (1880, 31)
df.head()
'''
               사고번호                사고일시   요일  ... 사고유형2 도로형태1  도로형태2
0  2021010100100234 2021-01-01 21:00:00  금요일  ...    기타   단일로     기타
1  2021010200100103 2021-01-02 12:00:00  토요일  ...    기타   단일로     기타
2  2021010400100010 2021-01-04 03:00:00  월요일  ...   횡단중   단일로     기타
3  2021010400100112 2021-01-04 11:00:00  월요일  ...    기타   단일로     기타
4  2021010400100133 2021-01-04 12:00:00  월요일  ...  측면충돌   교차로   교차로안

[5 rows x 31 columns]
'''

# describe()로 통계정보 보기 
df.describe(include='all')

'''
                사고번호                           사고일시  ...         year          day
count   1.880000e+03                           1880  ...  1880.000000  1880.000000
unique           NaN                            NaN  ...          NaN          NaN
top              NaN                            NaN  ...          NaN          NaN
freq             NaN                            NaN  ...          NaN          NaN
mean    2.022165e+15  2022-08-11 01:50:35.106383104  ...  2022.096277    15.674468
min     2.021010e+15            2021-01-01 21:00:00  ...  2021.000000     1.000000
25%     2.021111e+15            2021-11-08 16:30:00  ...  2021.000000     8.000000
50%     2.022090e+15            2022-09-03 19:30:00  ...  2022.000000    16.000000
75%     2.023052e+15            2023-05-17 01:15:00  ...  2023.000000    23.000000
max     2.023123e+15            2023-12-31 06:00:00  ...  2023.000000    31.000000
std     8.309211e+11                            NaN  ...     0.832283     8.775018

[11 rows x 34 columns]
'''

# 사고관련 정수형 자료 컬럼 확인
print(df['사망자수'].unique())  # [1 0]
print(df['중상자수'].unique())  # [0 1 2 3 5 4]
print(df['경상자수'].unique())  # [ 0  1  2  4  3  5  6 10  7]
print(df['부상신고자수'].unique())  # [0 1 2 4 3]


#%%

# 사고 일수 구하기.
total_days = df['사고일시'].dt.date.nunique()
print(f'사고일수: {total_days}일') # 사고일수: 877일


# 총 사고 건수 구하기.
df['사고건수'] = '1' # 새로운 열추가해서 사고건수를 건당 1개로 잡기
df['사고건수'].info() # 1880 non-null   object
'''
<class 'pandas.core.series.Series'>
RangeIndex: 1880 entries, 0 to 1879
Series name: 사고건수
Non-Null Count  Dtype 
--------------  ----- 
1880 non-null   object
dtypes: object(1)
memory usage: 14.8+ KB
'''

df['사고건수']  = df['사고건수'].astype('int') # 새로운 열이 object라 int 타입변경
df['사고건수'].info()  # 1880 non-null   int32
'''
<class 'pandas.core.series.Series'>
RangeIndex: 1880 entries, 0 to 1879
Series name: 사고건수
Non-Null Count  Dtype
--------------  -----
1880 non-null   int32
dtypes: int32(1)
memory usage: 7.5 KB
'''

total_accidents = df['사고건수'].sum() # 1로 만든 열의 값 합치기
print('총 사고건수:',total_accidents) # 총 사고건수: 1880


# 일 평균 사고건수(소수점 첫째 자리 처리)
accidents_per_day = total_accidents / total_days
print(f'일 평균 사소건수: {accidents_per_day:,.1f}일') # 일 평균 사소건수: 2.1일

#%%
# 시각화하기 위해 폰트 설정
plt.rc('font', family='Malgun Gothic')


# 연도별, 월별, 일별, 요일별 사고 분석
# 연도 =  year
df['날짜2'].dtypes #  dtype('O')
df['날짜2']= pd.to_datetime(df['날짜2'])
df['날짜2'].dtypes  # dtype('<M8[ns]')

df['year'] = df['날짜2'].dt.year
df['year']

'''
0       2021
1       2021
2       2021
3       2021
4       2021

1875    2023
1876    2023
1877    2023
1878    2023
1879    2023
Name: year, Length: 1880, dtype: int32
'''

# 연별 사고 빈도 정렬
frequency_count_year = df.groupby('year')['사고건수'].sum()
frequency_count_year = frequency_count_year.sort_values(ascending=False)
frequency_count_year_df = pd.DataFrame(frequency_count_year) # 데이터 프레임으로 만들기
frequency_count_year_df.head()

'''
      사고건수
year      
2023   750
2021   569
2022   561
'''

# 연도별 사고 건수 막대그래프(미분리)

plt.rcParams['figure.figsize'] = (10, 5)  # 그래프 크기 설정하기
sns.barplot(data=frequency_count_year_df, x=frequency_count_year_df.index, y='사고건수')
plt.title('연도별 사고건수')
plt.xlabel('연도별')
plt.ylabel('사고건수')
plt.show()


#%%

# 월별 =  month
df['month'] = df['날짜2'].dt.month
df['month'] 

'''
0        1
1        1
2        1
3        1
4        1
        ..
1875    12
1876    12
1877    12
1878    12
1879    12
Name: month, Length: 1880, dtype: int32
'''

# 월별 사고 빈도 정렬
frequency_count_month = df.groupby('month')['사고건수'].sum()
frequency_count_month = frequency_count_month.sort_values(ascending=False)
frequency_count_month_df = pd.DataFrame(frequency_count_month) # 데이터 프레임으로 만들기
frequency_count_month_df.head()
'''
       사고건수
month      
10      182
5       179
12      167
6       165
11      161
'''

# 21-23년도 데이터를 분리없이 월별로 합쳐서 보기

plt.figure(figsize=(10, 5))  # 그래프 크기 설정
sns.barplot(data=frequency_count_month_df, x=frequency_count_month_df.index, y='사고건수', palette='viridis')
plt.title('월별 사고 빈도')
plt.xlabel('월')
plt.ylabel('사고 건수')
plt.show()


#%%
# 일별 = days
df['days'] = df['날짜2'].dt.day
df['days'] 

'''
0        1
1        2
2        4
3        4
4        4
        ..
1875    28
1876    29
1877    29
1878    29
1879    31
Name: day, Length: 1880, dtype: int32
'''

# 일자로 정리한 사고 빈도 정렬
frequency_count_days = df.groupby('날짜2')['사고건수'].sum()
frequency_count_days = frequency_count_days.sort_values(ascending=False)
frequency_count_days_df = pd.DataFrame(frequency_count_days) # 데이터 프레임으로 만들기
frequency_count_days_df.head()

'''
            사고건수
날짜2             
2023-05-15     9
2023-09-02     7
2023-09-20     7
2023-05-08     7
2023-08-21     7
'''
# 전체 일자별로 나와있는 사고건수를 산점도를 흩뿌리면?

plt.figure(figsize=(10, 5))  # 그래프 크기 설정
plt.scatter(frequency_count_days_df.index, frequency_count_days_df['사고건수'], color='b', marker='o')
plt.title('일별 사고 빈도(산점도)')
plt.xlabel('일')
plt.ylabel('사고 건수')
plt.grid(True)  # 그리드 표시
plt.show()


# 일별 사고 빈도 정렬
frequency_count_days2 = df.groupby('days')['사고건수'].sum()
frequency_count_days2 = frequency_count_days2.sort_values(ascending=False)
frequency_count_days2_df = pd.DataFrame(frequency_count_days2) # 데이터 프레임으로 만들기
frequency_count_days2_df.head()


'''
      사고건수
days      
26      78
8       77
10      77
20      75
14      72
'''

# 21-23년도 데이터를 분리없이 일별로 합쳐서 보기
plt.figure(figsize=(10, 5))  # 그래프 크기 설정
sns.barplot(data=frequency_count_days2_df, x=frequency_count_days2_df.index, y='사고건수')
plt.title('일별 사고 빈도')
plt.xlabel('일')
plt.ylabel('사고 건수')
plt.show()



#%%
# 현재 막힌구간
# 요일별 = day
accidents_by_day = df.groupby('요일')['사고건수'].sum()
accidents_by_day.head()
'''
요일
금요일    333
목요일    261
수요일    260
월요일    302
일요일    178
Name: 사고건수, dtype: int32
'''

# 요일별 사고 빈도 정렬

frequency_count_day = df.groupby('요일')['사고건수'].sum()
frequency_count_day = frequency_count_day.sort_values(ascending=False)
frequency_count_day_df = pd.DataFrame(frequency_count_day) # 데이터 프레임으로 만들기
frequency_count_day_df.head()

'''
     사고건수
요일       
금요일   333
월요일   302
화요일   282
토요일   264
목요일   261
'''

# 21-23년도 데이터를 분리없이 요일별로 합쳐서 보기
plt.figure(figsize=(10, 5))  # 그래프 크기 설정
sns.barplot(data=frequency_count_day_df, x=frequency_count_day_df.index, y='사고건수', palette='coolwarm')
plt.title('요일별 사고 빈도')
plt.xlabel('요일')
plt.ylabel('사고 건수')
plt.show()

#%%


# 시간대별 사고 빈도 정렬
frequency_count_hour = df.groupby('시간')['사고건수'].sum()
frequency_count_hour = frequency_count_hour.sort_values(ascending=False)
frequency_count_hour_df = pd.DataFrame(frequency_count_hour) # 데이터 프레임으로 만들기
frequency_count_hour_df.head()
'''
    사고건수
시간      
16   152
13   134
17   132
15   132
11   130
'''

# 21-23년도 데이터를 분리없이 요일별로 합쳐서 보기
plt.figure(figsize=(10, 5))  # 그래프 크기 설정
sns.barplot(data=frequency_count_hour_df, x=frequency_count_hour_df.index, y='사고건수')
plt.title('시간대별 사고 빈도')
plt.xlabel('시간대')
plt.ylabel('사고 건수')
plt.show()

#%%
# 가장 많은 사고가 난 시간은 어떤 사고들이지?
df[df['시간'] == 16]
'''
                  사고번호                사고일시   요일  ...  year month  day
5     2021010400100232 2021-01-04 16:00:00  월요일  ...  2021     1    4
10    2021011400100359 2021-01-14 16:00:00  목요일  ...  2021     1   14
11    2021011400100360 2021-01-14 16:00:00  목요일  ...  2021     1   14
38    2021020200100292 2021-02-02 16:00:00  화요일  ...  2021     2    2
41    2021020300100302 2021-02-03 16:00:00  수요일  ...  2021     2    3
               ...                 ...  ...  ...   ...   ...  ...
1822  2023113000100371 2023-11-30 16:00:00  목요일  ...  2023    11   30
1838  2023120800100376 2023-12-08 16:00:00  금요일  ...  2023    12    8
1854  2023121500100415 2023-12-15 16:00:00  금요일  ...  2023    12   15
1865  2023122100100371 2023-12-21 16:00:00  목요일  ...  2023    12   21
1871  2023122600100361 2023-12-26 16:00:00  화요일  ...  2023    12   26

[152 rows x 35 columns]
'''
# 가장 많은 사고가 난 날은 어떤 사고들이지?
df[df['날짜2'] == '2023-05-15']
'''
                  사고번호                사고일시   요일  ...  year month  day
1398  2023051500100050 2023-05-15 07:00:00  월요일  ...  2023     5   15
1399  2023051500100167 2023-05-15 10:00:00  월요일  ...  2023     5   15
1400  2023051500100299 2023-05-15 14:00:00  월요일  ...  2023     5   15
1401  2023051500100342 2023-05-15 15:00:00  월요일  ...  2023     5   15
1402  2023051500100344 2023-05-15 15:00:00  월요일  ...  2023     5   15
1403  2023051500100345 2023-05-15 15:00:00  월요일  ...  2023     5   15
1404  2023051500100372 2023-05-15 16:00:00  월요일  ...  2023     5   15
1405  2023051500100378 2023-05-15 16:00:00  월요일  ...  2023     5   15
1406  2023051500100499 2023-05-15 19:00:00  월요일  ...  2023     5   15

[9 rows x 35 columns]
'''


# 우리 자료에서 사고 첫날과 마지막날 보기 
print('첫 사고 일시:', df['사고일시'].min()) # 첫 사고 일시: 2021-01-01 21:00:00
print('마지막 사고 일시:',df['사고일시'].max()) # 마지막 사고 일시: 2023-12-31 06:00:00
