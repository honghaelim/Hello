###############################################################################
# 동일 폴더 내 필요 파일
#   - The elderly driver traffic accidents(suwon).xlsx
###############################################################################
###############################################################################
# 아나콘다 파워셸 프롬프트
#   >>> conda activate YSIT24
#   >>> cd (파일저장경로)
#   >>> streamlit run 240516_1700_graph_suwon.py
###############################################################################
# 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

###############################################################################
# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)
###############################################################################
# config.set_option() 함수를 사용하여 widemode 기본값 설정
st.set_page_config(layout='wide')

#%% 데이터 호출
df = pd.read_excel('The elderly driver traffic accidents(suwon).xlsx')

#%% 데이터 전처리
# [사고일시] -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')  
#   날짜(object)                   
df['날짜'] = df['사고일시'].dt.date   

# [시군구] -> 구/ 동
gu = []
dong = []
for i in range(len(df)) :
    gu.append(df['시군구'].str.split(' ')[i][2])
    dong.append(df['시군구'].str.split(' ')[i][3])
df['구'] = gu 
df['동'] = dong

# [가해운전자 연령] 00세(object) -> 00(int)
df['가해운전자 연령'] = df['가해운전자 연령'].str[:-1]
df['가해운전자 연령'] = df['가해운전자 연령'].astype('int64')

# [피해운전자] nan -> 0
df.iloc[:, 18:22] = df.iloc[:, 18:22].fillna(0)

# [피해운전자 연령] 00세(object) -> 00(int)
df['피해운전자 연령'] = df['피해운전자 연령'].str[:-1]
## -> nan(0->nan), '미분'('미분류') 존재
#       -> '미분류' : 0
df['피해운전자 연령'] = df['피해운전자 연령'].replace('미분', 0)
#       -> nan : 0
df['피해운전자 연령'] = df['피해운전자 연령'].fillna(0)
# int 변환
df['피해운전자 연령'] = df['피해운전자 연령'].astype('int64')

#%% 일부 열 추출
df_table = df.loc[:, ['사고일시', '날짜', '요일', '구', '동', '노면상태', '기상상태', '도로형태', 
                      '법규위반', '사고유형', '사고내용',
                      '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도', 
                      '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도',
                      '사망자수', '중상자수', '경상자수', '부상신고자수'
                      ]]

#%% 
# ## 사이드바
st.sidebar.subheader('수원시')
# '구' 선택을 위한 체크박스
ck_gu1 = st.sidebar.checkbox('권선구', value = True)
ck_gu2 = st.sidebar.checkbox('장안구', value = True)
ck_gu3 = st.sidebar.checkbox('팔달구', value = True)
ck_gu4 = st.sidebar.checkbox('영통구', value = True)

clicked = st.sidebar.button('데이터 가져오기')

if clicked == True :        
    selected_gu = []
    if ck_gu1 == True :
        selected_gu.append('권선구')
    if ck_gu2 == True :
        selected_gu.append('장안구')
    if ck_gu3 == True :
        selected_gu.append('팔달구')
    if ck_gu4 == True :
        selected_gu.append('영통구')
    
    if selected_gu == [] :
        st.subheader('지역을 선택하세요.')
    else :        
        df_gu = df_table.copy()        
        op = df_gu['구'].isin(selected_gu)     
            
    df_gu = df_gu.loc[op, :]         
    df_gu = df_gu.sort_values(by='사고일시', ascending=False).reset_index(drop=True)
    
    # ## 메인화면
    st.title(f'수원시 {selected_gu} 고령 교통사고 현황')
    
    st.subheader('교통사고 데이터')
    
    df_gu['사고건수'] = 1
    st.markdown(f"- 2021년부터 2023년까지 **총 {df_gu['사고건수'].sum():,d}건**의 교통사고 발생")    
    
    df_gu_table = df_gu.loc[:, ['구', '동', '날짜', '요일', '도로형태', '기상상태', '노면상태', '법규위반',
                           '사고유형', '가해운전자 연령', '가해운전자 성별', '가해운전자 차종',                            
                           '사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수']]   
    st.dataframe(df_gu_table)    
    
    # DataFrame 데이터를 csv 데이터(csv_data)로 변환
    csv_data = df_gu.to_csv()
    # DataFrame 데이터를 엑셀 데이터(excel_data)로 변환
    from io import BytesIO
    excel_data = BytesIO()
    df_gu.to_excel(excel_data)
    
    columns = st.columns(5) 
    with columns[3] :
        st.download_button('CSV 다운로드', csv_data, file_name = f'{selected_gu}_accident_2021-2023.csv')
    with columns[4] :
        st.download_button('엑셀 다운로드', excel_data, file_name = f'{selected_gu}_accident_2021-2023.xlsx')

    #%% 그래프 생성
    ###############################################################################         
    [col1, col2] = st.columns(2)
    with col1 :
        # ## 지역(구)별/ 일별 사고발생 건수    
        daily_accidents_gu = df_gu.groupby(['구','날짜'])['사고건수'].sum().reset_index()
        daily_accidents_gu.columns = ['구', '사고일자', '사고건수']
                
        # 차트 생성
        fig1, ax = plt.subplots(figsize=(20, 10))
        for i in daily_accidents_gu['구'].unique() :
            ax.plot(daily_accidents_gu.loc[daily_accidents_gu['구']==i, '사고일자'], daily_accidents_gu.loc[daily_accidents_gu['구']==i, '사고건수'], label=i)
        ax.set_xlabel('일자', fontsize = 20)
        ax.set_ylabel('건수', fontsize = 20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(linestyle='--', alpha=0.2)
        plt.legend(loc='best')
        
        # 스트림릿에서 차트 출력
        st.subheader('지역별/ 일별 교통사고 건수') 
        
        for i in daily_accidents_gu['구'].unique() :
            max_df = daily_accidents_gu.loc[daily_accidents_gu['구']==i, :]
            max_cnt = max_df['사고건수'].max()
            max_day = max_df.loc[max_df['사고건수']==max_cnt, '사고일자'].values[0]        
            st.markdown(f"- {i} 교통사고 **최다 발생** : {max_day.year}년 {max_day.month}월 {max_day.day}일 **{max_cnt:,d}건**")
        
        st.pyplot(fig1)  

    with col2 :
        # ## 연도별/ 월별 사고발생 건수(작년 기준)
        df_gu['연'] = df_gu['사고일시'].dt.year
        df_gu['월'] = df_gu['사고일시'].dt.month
        
        monthly_accident = df_gu.groupby(['연','월'])['사고건수'].sum().reset_index()
        monthly_accident.columns = ['연', '사고월', '사고건수']
        
        # 차트 생성
        fig2, ax = plt.subplots(figsize=(20, 10))
        for i in monthly_accident['연'].unique() :
            ax.plot(monthly_accident.loc[monthly_accident['연']==i, '사고월'], monthly_accident.loc[monthly_accident['연']==i, '사고건수'], label=i)
        ax.set_xlabel('월', fontsize = 20)
        ax.set_ylabel('건수', fontsize = 20)
        plt.xticks(range(1,12+1), fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(linestyle='--', alpha=0.2)
        plt.legend(loc='best')
        
        # 스트림릿에서 차트 출력        
        st.subheader('연도별/ 월별 교통사고 건수')
        
        for i in monthly_accident['연'].unique() :            
            max_df = monthly_accident.loc[monthly_accident['연']==i, :]
            max_cnt = max_df['사고건수'].max()
            max_month = max_df.loc[max_df['사고건수']==max_cnt, '사고월'].values[0]      
            st.markdown(f"- {i}년 **최다 발생** : {max_month}월 **{max_cnt:,d}건**")
        
        st.pyplot(fig2)    