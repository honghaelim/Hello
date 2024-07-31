# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:29:05 2024

@author: ksj
"""

import pandas as pd
import seaborn as sns

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
""" len(dong_list) ## 54
['오목천동',
 '연무동',
 '권선동',
 '우만동',
 '인계동',
 '화서동',
 '파장동',
 '팔달로1가',
 '영화동',
 '고색동',
 '송죽동',
 '팔달로2가',
 '매산로1가',
 '정자동',
 '이목동',
 '원천동',
 '영통동',
 '서둔동',
 '율전동',
 '매산로3가',
 '호매실동',
 '세류동',
 '망포동',
 '팔달로3가',
 '매탄동',
 '천천동',
 '탑동',
 '매교동',
 '북수동',
 '구운동',
 '당수동',
 '곡반정동',
 '조원동',
 '영동',
 '입북동',
 '이의동',
 '평동',
 '매산로2가',
 '금곡동',
 '하광교동',
 '지동',
 '고등동',
 '신동',
 '구천동',
 '하동',
 '남수동',
 '매향동',
 '중동',
 '평리동',
 '남창동',
 '신풍동',
 '장안동',
 '교동',
 '대황교동']
"""

# 노면상태
road_list = ['건조', '서리/결빙', '젖음/습기', '기타', '적설']

# 기상상태
weather_list = ['맑음', '눈', '흐림', '기타', '비']

# ## 요인df 생성
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
"""
     주야간     동 노면상태 기상상태
0     주간  오목천동   건조   맑음
1     주간  오목천동   건조    눈
2     주간  오목천동   건조   흐림
3     주간  오목천동   건조   기타
4     주간  오목천동   건조    비
  ..   ...  ...  ...
2695  야간  대황교동   적설   맑음
2696  야간  대황교동   적설    눈
2697  야간  대황교동   적설   흐림
2698  야간  대황교동   적설   기타
2699  야간  대황교동   적설    비

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

# ## 각 요인의 위험지수(순위 역순) Series 생성
rank_time = rank_score(df_table, '주야간')
rank_area = rank_score(df_table, '동')
rank_road = rank_score(df_table, '노면상태')
rank_weather = rank_score(df_table, '기상상태')

#%%
###############################################################################
# 요인별 위험지수 병합
###############################################################################
element_risk = element_df.copy()

# 주야간
for x in time_list :
    element_risk.loc[element_risk['주야간'] == x, '주야간_risk'] = rank_time[x]
    
# 동 - 구 매칭
area = df.loc[:, ['동','구']].drop_duplicates().set_index('동')
area = area.loc[:, '구'] # DataFrame -> Series (area['오목천동'] 형태로 호출하기 위해)
for x in dong_list :
    element_risk.loc[element_risk['동'] == x, '동_risk'] = rank_area[x]
    element_risk.loc[element_risk['동'] == x, '구'] = area[x]

# 노면상태
for x in road_list :
    element_risk.loc[element_risk['노면상태'] == x, '노면상태_risk'] = rank_road[x]

# 기상상태
for x in weather_list :
    element_risk.loc[element_risk['기상상태'] == x, '기상상태_risk'] = rank_weather[x]

# 열 순서 변경
element_risk = element_risk.loc[:,['구', '동', '주야간', '노면상태', '기상상태', '동_risk', '주야간_risk', '노면상태_risk', '기상상태_risk']]
print(element_risk)
"""
        구     동 주야간 노면상태 기상상태  동_risk  주야간_risk  노면상태_risk  기상상태_risk
0     권선구  오목천동  주간   건조   맑음    26.0       2.0        5.0        5.0
1     권선구  오목천동  주간   건조    눈    26.0       2.0        5.0        2.0
2     권선구  오목천동  주간   건조   흐림    26.0       2.0        5.0        3.0
3     권선구  오목천동  주간   건조   기타    26.0       2.0        5.0        1.0
4     권선구  오목천동  주간   건조    비    26.0       2.0        5.0        4.0
  ...   ...  ..  ...  ...     ...       ...        ...        ...
2695  권선구  대황교동  야간   적설   맑음     3.0       1.0        1.0        5.0
2696  권선구  대황교동  야간   적설    눈     3.0       1.0        1.0        2.0
2697  권선구  대황교동  야간   적설   흐림     3.0       1.0        1.0        3.0
2698  권선구  대황교동  야간   적설   기타     3.0       1.0        1.0        1.0
2699  권선구  대황교동  야간   적설    비     3.0       1.0        1.0        4.0

[2700 rows x 9 columns]
"""

# ## 요인 위험지수 총계 계산(total_risk)
element_risk['total_risk'] = element_risk.iloc[:, -4:].sum(axis=1)
element_risk = element_risk.sort_values(by='total_risk', ascending=False).reset_index(drop=True)

print(element_risk)
"""
        구    동 주야간   노면상태  ... 주야간_risk  노면상태_risk  기상상태_risk  total_risk
0     장안구  정자동  주간     건조  ...      2.0        5.0        5.0        66.0
1     권선구  권선동  주간     건조  ...      2.0        5.0        5.0        65.0
2     장안구  정자동  주간  젖음/습기  ...      2.0        4.0        5.0        65.0
3     장안구  정자동  주간     건조  ...      2.0        5.0        4.0        65.0
4     장안구  정자동  야간     건조  ...      1.0        5.0        5.0        65.0
  ...  ...  ..    ...  ...      ...        ...        ...         ...
2695  팔달구  남창동  야간     적설  ...      1.0        1.0        2.0         5.0
2696  권선구  평리동  야간     적설  ...      1.0        1.0        1.0         5.0
2697  팔달구  남창동  야간     기타  ...      1.0        2.0        1.0         5.0
2698  팔달구  남창동  주간     적설  ...      2.0        1.0        1.0         5.0
2699  팔달구  남창동  야간     적설  ...      1.0        1.0        1.0         4.0

[2700 rows x 10 columns]
"""

#%%
###############################################################################
# 요인 가중치 -> 상관계수
###############################################################################
# 요인_risk - total_risk 상관계수
element_corr = element_risk.iloc[:, -5:].corr()
sns.heatmap(element_corr, annot = True, cmap='YlGnBu', linewidth=.5, cbar=False)

corr_time = element_corr.loc['주야간_risk', 'total_risk'] + 1
corr_dong = element_corr.loc['동_risk', 'total_risk'] + 1
corr_road = element_corr.loc['노면상태_risk', 'total_risk'] + 1
corr_weather = element_corr.loc['기상상태_risk', 'total_risk'] + 1

corr_risk = element_risk.copy()

# 가중치 부여한 위험지수 계산(corr_risk)
corr_risk['corr_risk'] = corr_risk['주야간_risk']*corr_time + corr_risk['동_risk']*corr_dong + corr_risk['노면상태_risk']*corr_road + corr_risk['기상상태_risk']*corr_weather 

print(corr_risk)
"""
        구    동 주야간   노면상태  ... 노면상태_risk  기상상태_risk  total_risk   corr_risk
0     장안구  정자동  주간     건조  ...       5.0        5.0        66.0  120.496625
1     권선구  권선동  주간     건조  ...       5.0        5.0        65.0  118.505120
2     장안구  정자동  주간  젖음/습기  ...       4.0        5.0        65.0  119.407400
3     장안구  정자동  주간     건조  ...       5.0        4.0        65.0  119.407400
4     장안구  정자동  야간     건조  ...       5.0        5.0        65.0  119.465079
  ...  ...  ..    ...  ...       ...        ...         ...         ...
2695  팔달구  남창동  야간     적설  ...       1.0        2.0         5.0    6.290725
2696  권선구  평리동  야간     적설  ...       1.0        1.0         5.0    7.193005
2697  팔달구  남창동  야간     기타  ...       2.0        1.0         5.0    6.290725
2698  팔달구  남창동  주간     적설  ...       1.0        1.0         5.0    6.233046
2699  팔달구  남창동  야간     적설  ...       1.0        1.0         4.0    5.201500

[2700 rows x 11 columns]
"""

corr_risk.to_csv('요인별 위험지수_동(가중치부여).csv')
