# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:21:36 2024

@author: ksj
"""

import pandas as pd

risk_score = pd.read_csv('요인별 위험지수(가중치부여).csv', index_col='Unnamed: 0')
print(risk_score)
"""
    주야간    구   노면상태 기상상태  ...  노면상태_risk  기상상태_risk  total_risk  corr_risk
0    주간  권선구     건조   맑음  ...        5.0        5.0        16.0  24.363554
1    주간  권선구     건조    비  ...        5.0        4.0        15.0  22.760531
2    주간  팔달구     건조   맑음  ...        5.0        5.0        15.0  22.886822
3    주간  권선구  젖음/습기   맑음  ...        4.0        5.0        15.0  22.760531
4    야간  권선구     건조   맑음  ...        5.0        5.0        15.0  23.150353
..   ..  ...    ...  ...  ...        ...        ...         ...        ...
195  주간  영통구     적설   기타  ...        1.0        1.0         5.0   7.109178
196  야간  영통구     기타   기타  ...        2.0        1.0         5.0   7.499000
197  야간  장안구     적설   기타  ...        1.0        1.0         5.0   7.372709
198  야간  영통구     적설    눈  ...        1.0        2.0         5.0   7.499000
199  야간  영통구     적설   기타  ...        1.0        1.0         4.0   5.895977

[200 rows x 10 columns]
"""

#%%
###############################################################################
# 조건에 따른 위험도(corr_risk) 추출
###############################################################################
# 주야간
# 현재 시간 가져오기
from datetime import datetime
current_time = datetime.now()
""" 주간/야간 구분 함수
> 주간: 오전 7시부터 오후 8시까지 (13시간)
> 야간: 오후 8시부터 다음 날 오전 7시까지 (11시간) """
def timesplit(current_time):
    if current_time.hour>=7 and current_time.hour<20:
        current="주간"
    else:
        current="야간"
    return current
        
time = timesplit(current_time)
print(time)
## 주간

# 구
gu_list = ['권선구', '장안구', '팔달구', '영통구']
# 노면상태
road_list = ['건조', '서리/결빙', '젖음/습기', '기타', '적설']
# 기상상태
weather_list = ['맑음', '눈', '흐림', '기타', '비']

#%%
# 조건 설정
op_time = risk_score['주야간'] == time
op_road = risk_score['노면상태'] == road_list[0]
op_weather = risk_score['기상상태'] == weather_list[0]

fil_score = risk_score.loc[op_time & op_road & op_weather, :]
fil_score = fil_score.reset_index(drop=True)
print(fil_score)
"""
  주야간    구 노면상태 기상상태  ...  노면상태_risk  기상상태_risk  total_risk  corr_risk
0  주간  권선구   건조   맑음  ...        5.0        5.0        16.0  24.363554
1  주간  팔달구   건조   맑음  ...        5.0        5.0        15.0  22.886822
2  주간  장안구   건조   맑음  ...        5.0        5.0        14.0  21.410091
3  주간  영통구   건조   맑음  ...        5.0        5.0        13.0  19.933360

[4 rows x 10 columns]
"""

gu_score = fil_score.loc[:, ['구', 'corr_risk']]
print(gu_score)    
"""
     구  corr_risk
0  권선구  24.363554
1  팔달구  22.886822
2  장안구  21.410091
3  영통구  19.933360
"""

#%%
from datetime import datetime

# 함수 1 : 구별 값(table) 반환
"""
요인
- 주야간(자동) : 현재 시간 기준 자동반영
- 노면상태(입력) :
- 기상상태(입력) : 
- 구(출력) : 수원시 4개 '구'별 위험도 table 반환
    -> 지도 시각화 때 표에서 값 추출?
"""
def timesplit(current_time):
    if current_time.hour>=7 and current_time.hour<20:
        current="주간"
    else:
        current="야간"
    return current

def now_risk(road, weather) :
    risk_score = pd.read_csv('요인별 위험지수(가중치부여).csv', index_col='Unnamed: 0')

    op_time = risk_score['주야간'] == timesplit(datetime.now())
    op_road = risk_score['노면상태'] == road
    op_weather = risk_score['기상상태'] == weather

    fil_score = risk_score.loc[op_time & op_road & op_weather, :]
    fil_score = fil_score.reset_index(drop=True)
    print(fil_score)
    
    gu_score = fil_score.loc[:, ['구', 'corr_risk']]
    
    return gu_score

gu_score = now_risk('서리/결빙','흐림')
"""
  주야간    구   노면상태 기상상태  ...  노면상태_risk  기상상태_risk  total_risk  corr_risk
0  주간  권선구  서리/결빙   흐림  ...        3.0        3.0        12.0  17.951463
1  주간  팔달구  서리/결빙   흐림  ...        3.0        3.0        11.0  16.474731
2  주간  장안구  서리/결빙   흐림  ...        3.0        3.0        10.0  14.998000
3  주간  영통구  서리/결빙   흐림  ...        3.0        3.0         9.0  13.521269

[4 rows x 10 columns]
"""
print(gu_score)
"""
     구  corr_risk
0  권선구  17.951463
1  팔달구  16.474731
2  장안구  14.998000
3  영통구  13.521269
"""

#%%
# 함수 2 : 값 반환
"""
요인
- 주야간(자동) : 현재 시간 기준 자동반영
- 노면상태(입력) -> 기상청 기준 자동반영 필요
- 기상상태(입력) -> 기상청 기준 자동반영 필요
- 구(입력) -> 지도 시각화 값 삽입?
"""
def timesplit(current_time):
    if current_time.hour>=7 and current_time.hour<20:
        current="주간"
    else:
        current="야간"
    return current

def now_risk(gu, road, weather) :
    risk_score = pd.read_csv('요인별 위험지수(가중치부여).csv', index_col='Unnamed: 0')

    op_time = risk_score['주야간'] == timesplit(datetime.now())
    op_gu = risk_score['구'] == gu
    op_road = risk_score['노면상태'] == road
    op_weather = risk_score['기상상태'] == weather

    fil_score = risk_score.loc[op_time & op_gu & op_road & op_weather, :]
    fil_score = fil_score.reset_index(drop=True)
    print(fil_score)
    
    return fil_score.iloc[0,-1]

score = now_risk('권선구','서리/결빙','흐림')
"""
  주야간    구   노면상태 기상상태  ...  노면상태_risk  기상상태_risk  total_risk  corr_risk
0  주간  권선구  서리/결빙   흐림  ...        3.0        3.0        12.0  17.951463
"""
print(score)
"""
17.95146274613557
"""
