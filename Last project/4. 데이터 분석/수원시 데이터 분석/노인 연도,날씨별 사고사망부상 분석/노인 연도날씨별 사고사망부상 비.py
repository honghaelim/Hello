# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:52:13 2024

@author: soyoung
"""

import pandas as pd

df=pd.read_excel(r"D:\Workspace\Python\mini\가해운전자 연령층별 기상상태별 교통사고.xls",header=1)

df.info()

df_older=df.iloc[18:24]
df_older = df_older.reset_index(drop=True)

df_older.info()

#열이름 변경
df_older.rename(columns={'합계': '2019합계'}, inplace=True)
df_older.rename(columns={'맑음': '2019맑음'}, inplace=True)
df_older.rename(columns={'흐림': '2019흐림'}, inplace=True)
df_older.rename(columns={'비': '2019비'}, inplace=True)
df_older.rename(columns={'안개': '2019안개'}, inplace=True)
df_older.rename(columns={'눈': '2019눈'}, inplace=True)
df_older.rename(columns={'기타/불명': '2019기타'}, inplace=True)

df_older.rename(columns={'합계.1': '2020합계'}, inplace=True)
df_older.rename(columns={'맑음.1': '2020맑음'}, inplace=True)
df_older.rename(columns={'흐림.1': '2020흐림'}, inplace=True)
df_older.rename(columns={'비.1': '2020비'}, inplace=True)
df_older.rename(columns={'안개.1': '2020안개'}, inplace=True)
df_older.rename(columns={'눈.1': '2020눈'}, inplace=True)
df_older.rename(columns={'기타/불명.1': '2020기타'}, inplace=True)

df_older.rename(columns={'합계.2': '2021합계'}, inplace=True)
df_older.rename(columns={'맑음.2': '2021맑음'}, inplace=True)
df_older.rename(columns={'흐림.2': '2021흐림'}, inplace=True)
df_older.rename(columns={'비.2': '2021비'}, inplace=True)
df_older.rename(columns={'안개.2': '2021안개'}, inplace=True)
df_older.rename(columns={'눈.2': '2021눈'}, inplace=True)
df_older.rename(columns={'기타/불명.2': '2021기타'}, inplace=True)

df_older.rename(columns={'합계.3': '2022합계'}, inplace=True)
df_older.rename(columns={'맑음.3': '2022맑음'}, inplace=True)
df_older.rename(columns={'흐림.3': '2022흐림'}, inplace=True)
df_older.rename(columns={'비.3': '2022비'}, inplace=True)
df_older.rename(columns={'안개.3': '2022안개'}, inplace=True)
df_older.rename(columns={'눈.3': '2022눈'}, inplace=True)
df_older.rename(columns={'기타/불명.3': '2022기타'}, inplace=True)

df_older.rename(columns={'합계.4': '2023합계'}, inplace=True)
df_older.rename(columns={'맑음.4': '2023맑음'}, inplace=True)
df_older.rename(columns={'흐림.4': '2023흐림'}, inplace=True)
df_older.rename(columns={'비.4': '2023비'}, inplace=True)
df_older.rename(columns={'안개.4': '2023안개'}, inplace=True)
df_older.rename(columns={'눈.4': '2023눈'}, inplace=True)
df_older.rename(columns={'기타/불명.4': '2023기타'}, inplace=True)


# 61세~64세 + 65세 이상 합쳐서 새로운 행
new_row = df_older.iloc[0] + df_older.iloc[3]
df_older = pd.concat([df_older, new_row.to_frame().T], ignore_index=True)

new_row = df_older.iloc[1] + df_older.iloc[4]
df_older = pd.concat([df_older, new_row.to_frame().T], ignore_index=True)

new_row = df_older.iloc[2] + df_older.iloc[5]
df_older = pd.concat([df_older, new_row.to_frame().T], ignore_index=True)


# 결과 확인
print(df_older.tail(3))
"""
     기본계획 연령별1        기준년도   2019합계   2019맑음  ...   2023비 2023안개  2023눈 2023기타
6  61~64세65세이상  사고[건]사고[건]  53086.0  48271.0  ...  3810.0     55  193.0  564.0
7  61~64세65세이상  사망[명]사망[명]   1074.0    944.0  ...    70.0      9    3.0   13.0
8  61~64세65세이상  부상[명]부상[명]  77736.0  70591.0  ...  5462.0     69  379.0  736.0 """


# 결과 확인
df_older.info()

#정수형으로 변환
for col in df_older.columns[2:]:
    df_older[col] = df_older[col].astype(int)


#그래프
import matplotlib.pyplot as plt
import seaborn as sns
# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)


data_to_plot = df_older.iloc[6:, 2:8]

# 데이터프레임 전치
df_transposed = df_older[6:].transpose()

# 결과 확인
print(df_transposed)
"""
                     6            7            8
기본계획 연령별1  61~64세65세이상  61~64세65세이상  61~64세65세이상
기준년도        사고[건]사고[건]   사망[명]사망[명]   부상[명]부상[명]
2019합계           53086         1074        77736
2019맑음           48271          944        70591
2019흐림            1479           60         2124
2019비             2917           60         4427
2019안개              65            5          111
2019눈               78            0          132
2019기타             276            5          351
2020합계           49247         1001        70506
2020맑음           43902          836        62635
2020흐림            1686           63         2465
2020비             3176           86         4668
2020안개              51            4           96
2020눈               94            1          170
2020기타             338           11          472
2021합계           50419          987        71489
2021맑음           45073          821        63767
2021흐림            1637           64         2304
2021비             2955           79         4345
2021안개              57            8           74
2021눈              237            6          385
2021기타             460            9          614
2022합계           53812         1032        76847
2022맑음           49140          889        70049
2022흐림            1555           52         2170
2022비             2325           68         3443
2022안개              39            2           57
2022눈              227            5          398
2022기타             526           16          730
2023합계           59660          993        84792
2023맑음           53094          852        75395
2023흐림            1944           46         2751
2023비             3810           70         5462
2023안개              55            9           69
2023눈              193            3          379
2023기타             564           13          736  """

#데이터 연도별로 분할
data2019=df_transposed[3:9]
data2020=df_transposed[10:16]
data2021=df_transposed[17:23]
data2022=df_transposed[24:30]
data2023=df_transposed[31:37]

#열이름 변경 
data2019.rename(columns={6: '사고'}, inplace=True)
data2019.rename(columns={7: '사망'}, inplace=True)
data2019.rename(columns={8: '부상'}, inplace=True)

data2020.rename(columns={6: '사고'}, inplace=True)
data2020.rename(columns={7: '사망'}, inplace=True)
data2020.rename(columns={8: '부상'}, inplace=True)

data2021.rename(columns={6: '사고'}, inplace=True)
data2021.rename(columns={7: '사망'}, inplace=True)
data2021.rename(columns={8: '부상'}, inplace=True)

data2022.rename(columns={6: '사고'}, inplace=True)
data2022.rename(columns={7: '사망'}, inplace=True)
data2022.rename(columns={8: '부상'}, inplace=True)

data2023.rename(columns={6: '사고'}, inplace=True)
data2023.rename(columns={7: '사망'}, inplace=True)
data2023.rename(columns={8: '부상'}, inplace=True)


#
#2019
plt.figure(figsize=(10, 6))
for col in ['사망', '사고', '부상']:
    plt.plot(data2019.index, data2019[col], marker='o', label=col)

plt.title('2019년도')
plt.xlabel('날씨')
plt.ylabel('수')
plt.legend()
plt.grid(True)
plt.ylim(0, 75000)
plt.show()

#2020
plt.figure(figsize=(10, 6))
for col in ['사망', '사고', '부상']:
    plt.plot(data2020.index, data2020[col], marker='o', label=col)

plt.title('2020년도')
plt.xlabel('날씨')
plt.ylabel('수')
plt.legend()
plt.grid(True)
plt.ylim(0, 75000)
plt.show()

#2021
plt.figure(figsize=(10, 6))
for col in ['사망', '사고', '부상']:
    plt.plot(data2021.index, data2021[col], marker='o', label=col)

plt.title('2021년도')
plt.xlabel('날씨')
plt.ylabel('수')
plt.legend()
plt.grid(True)
plt.ylim(0, 75000)
plt.show()

#2022
plt.figure(figsize=(10, 6))
for col in ['사망', '사고', '부상']:
    plt.plot(data2022.index, data2022[col], marker='o', label=col)

plt.title('2022년도')
plt.xlabel('날씨')
plt.ylabel('수')
plt.legend()
plt.grid(True)
plt.ylim(0, 75000)
plt.show()

#2023
plt.figure(figsize=(10, 6))
for col in ['사망', '사고', '부상']:
    plt.plot(data2023.index, data2023[col], marker='o', label=col)

plt.title('2023년도')
plt.xlabel('날씨')
plt.ylabel('수')
plt.legend()
plt.grid(True)
plt.ylim(0, 75000)
plt.show()