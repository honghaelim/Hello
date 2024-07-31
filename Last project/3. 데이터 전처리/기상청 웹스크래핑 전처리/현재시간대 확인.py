# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:59:02 2024

@author: soyoung
"""


from datetime import datetime

# 현재 시간 가져오기
current_time = datetime.now()

# 시간을 문자열로 변환하여 출력
print("현재 시간:", current_time)

"""지역별 기준:
> 주간: 오전 7시부터 오후 8시까지 (13시간)
> 야간: 오후 8시부터 다음 날 오전 7시까지 (11시간) """

def timesplit(current_time):
    if current_time.hour>=7 and current_time.hour<20:
        current="주간"
    else:
        current="야간"
    return current
        
result=timesplit(current_time)
print("현재 시간대(주간,야간):",result)

