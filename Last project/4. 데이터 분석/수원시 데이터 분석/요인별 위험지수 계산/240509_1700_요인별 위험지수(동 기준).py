# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:29:05 2024

@author: ksj
"""

import pandas as pd
# ## 데이터
df = pd.read_excel('./The elderly driver traffic accidents(suwon).xlsx')

# ## 전처리
# [사고일시] -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')  ## 2023-01-01 00:00:00
#   시간(int)
df['시간'] = df['사고일시'].dt.hour                                            ## 0

# [시군구] -> 구/ 동
dong = []
for i in range(len(df)) :
    dong.append(df['시군구'].str.split(' ')[i][2] + " " + df['시군구'].str.split(' ')[i][3])
df['동'] = dong

# 시간대 -> 주간/야간
""" 기준
주간: 오전 7시부터 오후 8시까지 (13시간)
야간: 오후 8시부터 다음 날 오전 7시까지 (11시간)
"""
df['주야간'] = df['시간'].apply(lambda x: '주간' if 7 <= x <= 20 else '야간')

# 사고건수
df['사고건수'] = 1

#%%
###############################################################################
# 요인의 모든 조합 df 생성
###############################################################################
# 주야간
time_list = ['주간', '야간']
# 동
dong_list = list(df['동'].unique())

print(len(dong_list)) ## 54

# 노면상태
road_list = ['건조', '서리/결빙', '젖음/습기', '기타', '적설']
# 기상상태
weather_list = ['맑음', '눈', '흐림', '기타', '비']

# 요인df 생성
time = []
for x in time_list :
    for _ in range(len(dong_list) * len(road_list) * len(weather_list)) :
        time.append(x)

dong = []
for x in dong_list :
    for _ in range(len(road_list) * len(weather_list)) :
        dong.append(x)
dong = dong * len(time_list)

road = []
for x in road_list :
    for _ in range(len(weather_list)) :
        road.append(x)
road = road * len(time_list) * len(dong_list)

weather = weather_list.copy()
weather = weather * len(time_list) * len(dong_list) * len(road_list)


element_df = pd.DataFrame({'주야간': time, '동': dong, '노면상태': road, '기상상태': weather })

print(element_df)
""" 2 * 54 * 5 * 5 = 2700
     주야간         동 노면상태 기상상태
0     주간  권선구 오목천동   건조   맑음
1     주간  권선구 오목천동   건조    눈
2     주간  권선구 오목천동   건조   흐림
3     주간  권선구 오목천동   건조   기타
4     주간  권선구 오목천동   건조    비
  ..       ...  ...  ...
2695  야간  권선구 대황교동   적설   맑음
2696  야간  권선구 대황교동   적설    눈
2697  야간  권선구 대황교동   적설   흐림
2698  야간  권선구 대황교동   적설   기타
2699  야간  권선구 대황교동   적설    비

[2700 rows x 4 columns]
"""

#%%
###############################################################################
# 각 요인 위험지수(순위)
###############################################################################
# ## 필요한 열만 추출
df_table = df.loc[:, ['주야간', '동', '노면상태', '기상상태', '사고건수']]

# ## 랭크 함수
#   - 사고건수가 많은 요인의 값이 크도록 'ascending=True' 설정
def rank_score(df, element) :
    rank_df = df[element].value_counts().rank(ascending=True).astype('int64')
    return rank_df

# ## 각 요인의 위험지수(순위 역순) df 생성
rank_time = rank_score(df_table, '주야간')
"""
주야간
주간    2
야간    1
Name: count, dtype: int64
""" 
rank_area = rank_score(df_table, '동')
"""
동
장안구 정자동      54
권선구 권선동      53
팔달구 인계동      52
장안구 영화동      51
팔달구 우만동      50
권선구 세류동      49
팔달구 화서동      48
영통구 이의동      47
팔달구 매산로1가    46
영통구 매탄동      45
영통구 영통동      44
권선구 고색동      43
권선구 구운동      42
영통구 원천동      41
장안구 송죽동      39
장안구 파장동      39
장안구 연무동      39
장안구 조원동      37
권선구 호매실동     36
팔달구 지동       35
권선구 금곡동      34
권선구 서둔동      33
장안구 율전동      32
권선구 탑동       31
권선구 평동       29
영통구 망포동      29
장안구 천천동      28
권선구 곡반정동     27
권선구 오목천동     26
팔달구 고등동      25
팔달구 매교동      24
팔달구 매산로2가    23
장안구 이목동      22
팔달구 팔달로2가    21
권선구 당수동      19
권선구 입북동      19
팔달구 북수동      19
팔달구 팔달로3가    17
팔달구 중동       16
팔달구 매산로3가    15
팔달구 영동       13
팔달구 장안동      13
영통구 하동       11
영통구 신동       11
팔달구 남수동       9
팔달구 팔달로1가     9
팔달구 구천동       7
장안구 하광교동      7
팔달구 매향동       5
팔달구 교동        5
팔달구 신풍동       3
권선구 대황교동      3
권선구 평리동       2
팔달구 남창동       1
Name: count, dtype: int64
"""
rank_road = rank_score(df_table, '노면상태')
"""
노면상태
건조       5
젖음/습기    4
서리/결빙    3
기타       2
적설       1
Name: count, dtype: int64
"""
rank_weather = rank_score(df_table, '기상상태')
"""
기상상태
맑음    5
비     4
흐림    3
눈     2
기타    1
Name: count, dtype: int64
"""

#%%
###############################################################################
# 요인별 위험지수 병합
###############################################################################
element_risk = element_df.copy()

for x in time_list :
    element_risk.loc[element_risk['주야간'] == x, '주야간_risk'] = rank_time[x]
    
for x in dong_list :
    element_risk.loc[element_risk['동'] == x, '동_risk'] = rank_area[x]

for x in road_list :
    element_risk.loc[element_risk['노면상태'] == x, '노면상태_risk'] = rank_road[x]

for x in weather_list :
    element_risk.loc[element_risk['기상상태'] == x, '기상상태_risk'] = rank_weather[x]

print(element_risk)
"""
     주야간         동 노면상태 기상상태  주야간_risk  동_risk  노면상태_risk  기상상태_risk
0     주간  권선구 오목천동   건조   맑음       2.0    26.0        5.0        5.0
1     주간  권선구 오목천동   건조    눈       2.0    26.0        5.0        2.0
2     주간  권선구 오목천동   건조   흐림       2.0    26.0        5.0        3.0
3     주간  권선구 오목천동   건조   기타       2.0    26.0        5.0        1.0
4     주간  권선구 오목천동   건조    비       2.0    26.0        5.0        4.0
  ..       ...  ...  ...       ...     ...        ...        ...
2695  야간  권선구 대황교동   적설   맑음       1.0     3.0        1.0        5.0
2696  야간  권선구 대황교동   적설    눈       1.0     3.0        1.0        2.0
2697  야간  권선구 대황교동   적설   흐림       1.0     3.0        1.0        3.0
2698  야간  권선구 대황교동   적설   기타       1.0     3.0        1.0        1.0
2699  야간  권선구 대황교동   적설    비       1.0     3.0        1.0        4.0

[2700 rows x 8 columns]
"""

# 요인 위험지수 총계
element_risk['total_risk'] = element_risk.iloc[:, -4:].sum(axis=1)

element_risk = element_risk.sort_values(by='total_risk', ascending=False).reset_index(drop=True)

print(element_risk)
"""
     주야간        동   노면상태 기상상태  ...  동_risk  노면상태_risk  기상상태_risk  total_risk
0     주간  장안구 정자동     건조   맑음  ...    54.0        5.0        5.0        66.0
1     주간  권선구 권선동     건조   맑음  ...    53.0        5.0        5.0        65.0
2     주간  장안구 정자동  젖음/습기   맑음  ...    54.0        4.0        5.0        65.0
3     주간  장안구 정자동     건조    비  ...    54.0        5.0        4.0        65.0
4     야간  장안구 정자동     건조   맑음  ...    54.0        5.0        5.0        65.0
  ..      ...    ...  ...  ...     ...        ...        ...         ...
2695  야간  팔달구 남창동     적설    눈  ...     1.0        1.0        2.0         5.0
2696  야간  권선구 평리동     적설   기타  ...     2.0        1.0        1.0         5.0
2697  야간  팔달구 남창동     기타   기타  ...     1.0        2.0        1.0         5.0
2698  주간  팔달구 남창동     적설   기타  ...     1.0        1.0        1.0         5.0
2699  야간  팔달구 남창동     적설   기타  ...     1.0        1.0        1.0         4.0

[2700 rows x 9 columns]
"""

#%%
###############################################################################
# 위험지수 총계와 해당 조건에 따른 실제 사고건수 비교
###############################################################################
accidents = df.groupby(['주야간', '동', '노면상태', '기상상태'])['사고건수'].sum().reset_index()
accidents = accidents.sort_values(by='사고건수', ascending=False).reset_index(drop=True)
print(len(accidents))
## 226

accident_risk = accidents.merge(element_risk, on = ['주야간', '동', '노면상태', '기상상태'])
accident_risk = accident_risk[['주야간', '동', '노면상태', '기상상태','사고건수', 'total_risk']]
print(accident_risk)
"""
    주야간        동   노면상태 기상상태  사고건수  total_risk
0    주간  장안구 정자동     건조   맑음    88        66.0
1    주간  팔달구 인계동     건조   맑음    78        64.0
2    주간  권선구 권선동     건조   맑음    75        65.0
3    주간  팔달구 우만동     건조   맑음    75        62.0
4    주간  장안구 영화동     건조   맑음    72        63.0
..   ..      ...    ...  ...   ...         ...
221  야간  팔달구 남수동     건조   맑음     1        20.0
222  주간  영통구 이의동  젖음/습기   흐림     1        56.0
223  주간  영통구 이의동  젖음/습기   맑음     1        58.0
224  주간  영통구 이의동  서리/결빙   맑음     1        57.0
225  주간  팔달구 화서동  젖음/습기   흐림     1        57.0

[226 rows x 6 columns]
"""

#%%
###############################################################################
# 시각화
###############################################################################
import matplotlib.pyplot as plt
import seaborn as sns

# 사고건수 - total_risk 산점도
accident_risk.plot(kind='scatter', x='total_risk', y='사고건수', s=10, figsize=(10,5))
plt.show()

# 요인_risk - total_risk 상관계수
element_corr = element_risk.iloc[:, -5:].corr()
sns.heatmap(element_corr, annot = True, cmap='YlGnBu', linewidth=.5, cbar=False)
plt.show()
