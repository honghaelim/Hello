# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:13:54 2024

@author: ksj
"""

import pandas as pd

df_18 = pd.read_csv('E:/Workspace/!project_team/4.18-23수원교통사고/accidentInfoList_18-20.csv', low_memory=False, encoding='CP949')
df_21 = pd.read_csv('E:/Workspace/!project_team/4.18-23수원교통사고/accidentInfoList_21-23.csv', low_memory=False, encoding='CP949')

df = pd.concat([df_18, df_21])
df = df.reset_index()

#%%
# [사고일시] -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')            
#   날짜(object)                   
df['연'] = df['사고일시'].dt.year                                         
#   월
df['월'] = df['사고일시'].dt.month
#   일
df['일'] = df['사고일시'].dt.day                             
#   시간(int)
df['시간'] = df['사고일시'].dt.hour      

# [시군구] -> 구/ 동
gu = []
dong = []
for i in range(len(df)) :
    gu.append(df['시군구'].str.split(' ')[i][2])
    dong.append(df['시군구'].str.split(' ')[i][3])
df['구'] = gu 
df['동'] = dong

#%%
# [연령] 00세(object) -> 00(int)
# '가해운전자'
df['가해운전자 연령'].unique()
"""
array(['39세', '44세', '53세', '52세', '30세', '37세', '54세', '48세', '62세',
       '49세', '55세', '24세', '26세', '28세', '58세', '41세', '57세', '66세',
       '18세', '64세', '59세', '43세', '70세', '61세', '63세', '19세', '36세',
       '38세', '40세', '50세', '27세', '69세', '45세', '72세', '56세', '47세',
       '32세', '23세', '31세', '76세', '68세', '60세', '42세', '46세', '51세',
       '25세', '67세', '78세', '33세', '65세', '22세', '21세', '75세', '미분류',
       '71세', '29세', '35세', '14세', '20세', '34세', '17세', '77세', '74세',
       '73세', '16세', '13세', '9세', '83세', '8세', '79세', '10세', '82세', '12세',
       '80세', '11세', '4세', '81세', '85세', '84세', '15세', '87세', '88세',
       '86세', '89세', '96세', '6세', '93세'], dtype=object)
"""

df['가해운전자 연령'] = df['가해운전자 연령'].str[:-1]
# -> '미분류' : 0
df['가해운전자 연령'] = df['가해운전자 연령'].replace('미분', 0)
df['가해운전자 연령'] = df['가해운전자 연령'].astype('int64')

df['가해운전자 연령'].unique()
"""
array([39, 44, 53, 52, 30, 37, 54, 48, 62, 49, 55, 24, 26, 28, 58, 41, 57,
       66, 18, 64, 59, 43, 70, 61, 63, 19, 36, 38, 40, 50, 27, 69, 45, 72,
       56, 47, 32, 23, 31, 76, 68, 60, 42, 46, 51, 25, 67, 78, 33, 65, 22,
       21, 75,  0, 71, 29, 35, 14, 20, 34, 17, 77, 74, 73, 16, 13,  9, 83,
        8, 79, 10, 82, 12, 80, 11,  4, 81, 85, 84, 15, 87, 88, 86, 89, 96,
        6, 93], dtype=int64)
"""
#%%
df.to_excel('accidentInfoList_18-23.xlsx')

#%%
df.columns
"""
Index(['index', '사고번호', '사고일시', '요일', '시군구', '사고내용', '사망자수', '중상자수', '경상자수',
       '부상신고자수', '사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해운전자 차종',
       '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별',
       '피해운전자 연령', '피해운전자 상해정도', '연', '월', '일', '시간', '구', '동'],
      dtype='object')
"""

df_age = df.loc[:, ['사고일시', '연', '월', '일', '요일', '시간', '구', '동',  
                '가해운전자 차종', '가해운전자 성별', '가해운전자 연령',
                '사망자수', '중상자수', '경상자수', '부상신고자수']]

df_age['연령대'] = df_age['가해운전자 연령'].apply(lambda x: '노인' if x >= 65 else '일반')

df_age['사고건수'] = 1

#%%
#%% ECLO 계산 함수
def cal_eclo(df) :
    df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1
    df['ECLO/사고건수'] = df['ECLO']/df['사고건수']
    return df

#%% 막대그래프_사고건수, ECLO
import matplotlib.pyplot as plt

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

def plot_bar(df, col) :
    df = df.reset_index()
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.bar(df[col], df['사고건수'], label ='사고건수')
    plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    plt.bar(df[col], df['ECLO/사고건수'], label ='ECLO')
    plt.legend(loc='best')
    
    plt.savefig('./graph/'+ f'graph_{col}별 교통사고.png')
    plt.show()

def plot_bar2(df, col):
    # 데이터 전처리
    df = df.reset_index()

    # 연령대별 데이터 그룹화
    grouped_data = df.groupby('연령대')

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(15, 5))

    # 각 연령대별 데이터 처리 및 막대 그래프 추가
    for age_group, data in grouped_data:
        color = 'blue' if age_group == '노인' else 'red'
        label = f'{age_group} ({data["사고건수"].sum()})'
        ax.bar(data[col], data['사고건수'], label=label, color=color, alpha=0.7)

    # 그래프 설정
    ax.set_title(f'{col}별 연령대별 사고 현황')
    ax.set_xlabel(col)
    ax.set_ylabel('사고 건수')
    ax.legend(loc='best')
    ax.grid(True)

    # 그래프 저장
    filename = f'graph_{col}별 연령대별 교통사고.png'
    plt.savefig(f'./{filename}')
    #plt.show()  # 모든 그래프 화면 출력 시 주석 해제

#%% 연도별 교통사고 현황
year_table = df_age.groupby(['연','연령대'])[['사고건수', '사망자수', '중상자수', '경상자수', '부상신고자수']].sum()
year_table = cal_eclo(year_table)
print(year_table)
"""
          사고건수  사망자수  중상자수  경상자수  부상신고자수   ECLO  ECLO/사고건수
연    연령대                                                  
2018 노인    421     7   102   410      44   1854   4.403800
     일반   4575    32  1331  4714     546  21663   4.735082
2019 노인    498     6   121   510      71   2266   4.550201
     일반   4422    25  1051  4706     585  20208   4.569878
2020 노인    470     3    91   494      54   2021   4.300000
     일반   4010    26   924  4404     492  18584   4.634414
2021 노인    569     6   123   589      67   2509   4.409490
     일반   4046    19   845  4602     441  18662   4.612457
2022 노인    561     6   116   576      62   2430   4.331551
     일반   4144    19   921  4479     415  18647   4.499759
2023 노인    750     4   168   844      69   3481   4.641333
     일반   3994    26   914  4339     354  18201   4.557086
"""

plot_bar2(year_table, '연')









