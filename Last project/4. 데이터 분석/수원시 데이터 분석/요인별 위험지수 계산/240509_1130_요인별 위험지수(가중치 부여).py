# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:41:15 2024

@author: ksj
"""

import pandas as pd

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

#%%
###############################################################################
# 요인의 모든 조합 df 생성
###############################################################################
# 주야간
time_list = ['주간', '야간']
# 구
gu_list = ['권선구', '장안구', '팔달구', '영통구']
# 노면상태
road_list = ['건조', '서리/결빙', '젖음/습기', '기타', '적설']
# 기상상태
weather_list = ['맑음', '눈', '흐림', '기타', '비']

# 요인df 생성
time = []
for x in time_list :
    for _ in range(len(gu_list) * len(road_list) * len(weather_list)) :
        time.append(x)

gu = []
for x in gu_list :
    for _ in range(len(road_list) * len(weather_list)) :
        gu.append(x)
gu = gu * len(time_list)

road = []
for x in road_list :
    for _ in range(len(weather_list)) :
        road.append(x)
road = road * len(time_list) * len(gu_list)

weather = weather_list.copy()
weather = weather * len(time_list) * len(gu_list) * len(road_list)

element_df = pd.DataFrame({'주야간': time, '구': gu, '노면상태': road, '기상상태': weather })

#%%
###############################################################################
# 각 요인 위험지수(순위)
###############################################################################
# ## 데이터
df = pd.read_excel('./The elderly driver traffic accidents(suwon).xlsx')

# ## 전처리
# [사고일시] -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')  ## 2023-01-01 00:00:00
#   시간(int)
df['시간'] = df['사고일시'].dt.hour                                            ## 0

# [시군구] -> 구/ 동
gu = []
dong = []
for i in range(len(df)) :
    gu.append(df['시군구'].str.split(' ')[i][2])
    dong.append(df['시군구'].str.split(' ')[i][3])
df['구'] = gu 
df['동'] = dong

# 시간대 -> 주간/야간
""" 기준
주간: 오전 7시부터 오후 8시까지 (13시간)
야간: 오후 8시부터 다음 날 오전 7시까지 (11시간)
"""
df['주야간'] = df['시간'].apply(lambda x: '주간' if 7 <= x <= 20 else '야간')

# 사고건수
df['사고건수'] = 1


# ## 필요한 열만 추출
df_table = df.loc[:, ['주야간', '구', '노면상태', '기상상태', '사고건수']]

# ## 랭크 함수
#   - 사고건수가 많은 요인의 값이 크도록 'ascending=True' 설정
def rank_score(df, element) :
    rank_df = df[element].value_counts().rank(ascending=True).astype('int64')
    return rank_df

# ## 각 요인의 위험지수(순위 역순) df 생성
rank_time = rank_score(df_table, '주야간')
rank_area = rank_score(df_table, '구')
rank_road = rank_score(df_table, '노면상태')
rank_weather = rank_score(df_table, '기상상태')

#%%
###############################################################################
# 요인별 위험지수 병합
###############################################################################
element_risk = element_df.copy()

for x in time_list :
    element_risk.loc[element_risk['주야간'] == x, '주야간_risk'] = rank_time[x]
    
for x in gu_list :
    element_risk.loc[element_risk['구'] == x, '구_risk'] = rank_area[x]

for x in road_list :
    element_risk.loc[element_risk['노면상태'] == x, '노면상태_risk'] = rank_road[x]

for x in weather_list :
    element_risk.loc[element_risk['기상상태'] == x, '기상상태_risk'] = rank_weather[x]

# 요인 위험지수 총계
element_risk['total_risk'] = element_risk.iloc[:, -4:].sum(axis=1)

element_risk = element_risk.sort_values(by='total_risk', ascending=False).reset_index(drop=True)

#%%
###############################################################################
# 위험지수 총계와 해당 조건에 따른 실제 사고건수 비교
###############################################################################
accidents = df_table.groupby(['주야간', '구', '노면상태', '기상상태'])['사고건수'].sum().reset_index()
accidents = accidents.sort_values(by='사고건수', ascending=False).reset_index(drop=True)
print(len(accidents))
## 53

accident_risk = accidents.merge(element_risk, on = ['주야간', '구', '노면상태', '기상상태'])
accident_risk = accident_risk[['주야간', '구', '노면상태', '기상상태','사고건수', 'total_risk']]
print(accident_risk)
"""
   주야간    구   노면상태 기상상태  사고건수  total_risk
0   주간  팔달구     건조   맑음   427        15.0
1   주간  권선구     건조   맑음   413        16.0
2   주간  장안구     건조   맑음   367        14.0
3   주간  영통구     건조   맑음   209        13.0
4   야간  권선구     건조   맑음    81        15.0
5   야간  팔달구     건조   맑음    70        14.0
6   야간  장안구     건조   맑음    54        13.0
7   야간  영통구     건조   맑음    45        12.0
8   주간  권선구  젖음/습기    비    33        14.0
9   주간  장안구  젖음/습기    비    30        12.0
10  주간  팔달구  젖음/습기    비    19        13.0
11  야간  팔달구  젖음/습기    비     9        12.0
12  야간  권선구  젖음/습기    비     9        13.0
13  주간  장안구     건조   흐림     9        12.0
14  주간  영통구  젖음/습기    비     8        11.0
15  주간  팔달구     건조   흐림     7        13.0
16  주간  팔달구  젖음/습기   흐림     7        12.0
17  야간  장안구     건조   흐림     7        11.0
18  야간  장안구  젖음/습기    비     7        11.0
19  주간  권선구     건조   흐림     6        14.0
20  주간  영통구     건조   흐림     6        11.0
21  야간  권선구  젖음/습기   흐림     4        12.0
22  야간  영통구  젖음/습기    비     4        10.0
23  야간  영통구  젖음/습기   흐림     3         9.0
24  야간  권선구     건조   흐림     3        13.0
25  주간  권선구     기타   맑음     3        13.0
26  주간  장안구  젖음/습기   맑음     3        13.0
27  주간  영통구  젖음/습기   흐림     3        10.0
28  주간  팔달구  젖음/습기    눈     2        11.0
29  주간  팔달구  서리/결빙    눈     2        10.0
30  주간  장안구  젖음/습기   흐림     2        11.0
31  주간  장안구     적설    눈     2         7.0
32  야간  팔달구     건조   흐림     2        12.0
33  야간  권선구     건조   기타     2        11.0
34  주간  영통구  젖음/습기   맑음     2        12.0
35  주간  영통구     기타   맑음     2        10.0
36  야간  장안구  젖음/습기   흐림     2        10.0
37  주간  권선구  서리/결빙   맑음     1        14.0
38  야간  팔달구  젖음/습기   흐림     1        11.0
39  주간  권선구     건조   기타     1        12.0
40  주간  팔달구  젖음/습기   기타     1        10.0
41  주간  팔달구     적설   흐림     1         9.0
42  야간  팔달구     건조   기타     1        10.0
43  야간  권선구  서리/결빙   맑음     1        13.0
44  야간  영통구     건조   기타     1         8.0
45  주간  권선구  젖음/습기   맑음     1        15.0
46  주간  영통구  서리/결빙   맑음     1        11.0
47  주간  장안구     적설   맑음     1        10.0
48  주간  권선구  젖음/습기   흐림     1        13.0
49  주간  장안구  서리/결빙   맑음     1        12.0
50  주간  장안구  서리/결빙    눈     1         9.0
51  주간  장안구     기타   맑음     1        11.0
52  야간  팔달구  서리/결빙   맑음     1        12.0
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

# 요인_risk - total_risk 산점도
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(1,4,1)
ax2 = fig.add_subplot(1,4,2)
ax3 = fig.add_subplot(1,4,3)
ax4 = fig.add_subplot(1,4,4)
sns.scatterplot(element_risk, x='total_risk', y='주야간_risk', ax=ax1)
sns.scatterplot(element_risk, x='total_risk', y='구_risk', ax=ax2)
sns.scatterplot(element_risk, x='total_risk', y='노면상태_risk', ax=ax3)
sns.scatterplot(element_risk, x='total_risk', y='기상상태_risk', ax=ax4)
plt.show()

# 요인_risk - total_risk 상관계수
element_corr = element_risk.iloc[:, -5:].corr()
sns.heatmap(element_corr, annot = True, cmap='YlGnBu', linewidth=.5, cbar=False)
plt.show()

#%%
###############################################################################
# 요인 가중치 -> 상관계수
###############################################################################
corr_time = element_corr.loc['주야간_risk', 'total_risk'] + 1
corr_gu = element_corr.loc['구_risk', 'total_risk'] + 1
corr_road = element_corr.loc['노면상태_risk', 'total_risk'] + 1
corr_weather = element_corr.loc['기상상태_risk', 'total_risk'] + 1

corr_risk = element_risk.copy()

corr_risk['corr_risk'] = corr_risk['주야간_risk']*corr_time + corr_risk['구_risk']*corr_gu + corr_risk['노면상태_risk']*corr_road + corr_risk['기상상태_risk']*corr_weather 

print(corr_risk)
"""
    주야간    구   노면상태 기상상태  ...  노면상태_risk  기상상태_risk  total_risk  corr_risk
0    주간  권선구     건조   맑음  ...        5.0        5.0        16.0  22.363554
1    주간  권선구     건조    비  ...        5.0        4.0        15.0  20.760531
2    주간  팔달구     건조   맑음  ...        5.0        5.0        15.0  20.886822
3    주간  권선구  젖음/습기   맑음  ...        4.0        5.0        15.0  20.760531
4    야간  권선구     건조   맑음  ...        5.0        5.0        15.0  22.150353
..   ..  ...    ...  ...  ...        ...        ...         ...        ...
195  주간  영통구     적설   기타  ...        1.0        1.0         5.0   5.109178
196  야간  영통구     기타   기타  ...        2.0        1.0         5.0   6.499000
197  야간  장안구     적설   기타  ...        1.0        1.0         5.0   6.372709
198  야간  영통구     적설    눈  ...        1.0        2.0         5.0   6.499000
199  야간  영통구     적설   기타  ...        1.0        1.0         4.0   4.895977

[200 rows x 10 columns]
"""

corr_risk.to_csv('요인별 위험지수(가중치부여).csv')

#%%
accident_corr = accidents.merge(corr_risk, on = ['주야간', '구', '노면상태', '기상상태'])
accident_corr = accident_corr[['주야간', '구', '노면상태', '기상상태','사고건수', 'corr_risk']]

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
# 사고건수 - total_risk 산점도
sns.scatterplot(accident_risk, x='total_risk', y='사고건수', ax=ax1)
ax1.set_title('요인별 risk 단순합')
# 사고건수 - corr_risk 산점도
sns.scatterplot(accident_corr, x='corr_risk', y='사고건수', ax=ax2)
ax2.set_title('요인별 risk 가중치 부여')
plt.show()

