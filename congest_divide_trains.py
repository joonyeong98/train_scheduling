#!/usr/bin/env python
# coding: utf-8


# 라이브러리 호출
import openpyxl
import pandas as pd
import numpy as np
import csv
import sys

# 데이터 불러오기

def read_schedule (filename):
    df = pd.read_excel(filename)
    return df

# 역코드 정보
# 모란기지 : 99, 모란 : 17, 암사 : 1
# 출발역, 출발시간, 도착역, 도착시간
def interest_data_preprocessing(df):
    new_data = pd.DataFrame({'출발역코드': df['출발역코드'],
                       '출발시간' : df['출발시간'],
                       '도착역코드': df['도착역코드'],
                       '도착시간' : df['도착시간']})
    # 시간 x시-> 0x시
    for i in range(len(new_data['출발시간'])):
        if len(new_data['출발시간'][i]) == 7:
            new_data['출발시간'][i] = '0'+new_data['출발시간'][i]
    for i in range(len(new_data['도착시간'])):
        if len(new_data['도착시간'][i]) == 7:
            new_data['도착시간'][i] = '0'+new_data['도착시간'][i]

    for i in range(len(new_data)):
        new_data['출발시간'][i]=new_data['출발시간'][i][:5]
        new_data['도착시간'][i]=new_data['도착시간'][i][:5]

    # 모란기지가 출발이나 도착인 데이터 제거
    temp = new_data.drop(index=new_data[new_data['출발역코드'] == 99].index)
    out_data = temp.drop(index=temp[temp['도착역코드'] == 99].index)
    return out_data



def get_amsa(df):
    # 암사행 데이터 추출
    dst_amsa0 = df[df['출발역코드'] == 17]
    dst_amsa0['index']=list(range(0,len(dst_amsa0)))
    dst_amsa0 = dst_amsa0.rename(columns={'도착역코드':'del도착역코드',
                                          '도착시간':'del도착시간'})
    dst_amsa1 = df[df['도착역코드'] == 1]
    dst_amsa1['index'] = list(range(0,len(dst_amsa1)))
    dst_amsa1 = dst_amsa1.rename(columns={'출발역코드':'del출발역코드',
                                          '출발시간':'del출발시간'})

    dst_amsa2 = pd.merge(dst_amsa0, dst_amsa1)

    dst_amsa3 = dst_amsa2.drop(['del도착역코드', 'del도착시간','del출발역코드',
                              'del출발시간','index'], axis='columns')


    dst_amsa = dst_amsa3.sort_values(by='출발시간')
    return dst_amsa

def get_moran(df):
    # 모란행 데이터 추출
    dst_moran0 = df[df['출발역코드']==1]
    dst_moran0['index']=list(range(0,len(dst_moran0)))
    dst_moran0 = dst_moran0.rename(columns={'도착역코드':'del도착역코드',
                                            '도착시간':'del도착시간'})
    dst_moran1 = df[df['도착역코드'] == 17]
    dst_moran1['index'] = list(range(0, len(dst_moran1)))
    dst_moran1 = dst_moran1.rename(columns={'출발역코드':'del출발역코드',
                                            '출발시간':'del출발시간'})

    dst_moran2=pd.merge(dst_moran0, dst_moran1)

    dst_moran3=dst_moran2.drop(['del도착역코드', 'del도착시간','del출발역코드',
                                'del출발시간','index'], axis='columns')

    dst_moran = dst_moran3.sort_values(by='출발시간')
    return dst_moran


def make_timetable(dst_st1, dst_st2):
    timetable = pd.concat([dst_st1, dst_st2]).sort_values(by='출발시간')
    timetable = timetable.reset_index()
    timetable = timetable.drop(['index'], axis='columns')

    tl=[]
    for i in range(len(timetable)):
        start = int(timetable['출발시간'][i][:2] + timetable['출발시간'][i][3])
        end = int(timetable['도착시간'][i][:2] + timetable['도착시간'][i][3])
        tmp = list(range(start, end+1))
        tl.append(tmp)
    timetable['times'] = tl
    return timetable

def make_train_dict(timetable):
    train_dict={}
    # x시 y_분을 xy로 표현함
    for i in range(50,290):
        train_dict[str(i)]=0

    for i in range(len(timetable['times'])):
        for j in range(50,290):
            cnt = timetable['times'][i].count(j)
            train_dict[str(j)] = train_dict[str(j)] + cnt
    # 불필요하게 생성된 시간 삭제
    for i in range(50,290):
        if int(str(i)[-1]) >= 6:
            train_dict.pop(str(i), None)
    return train_dict

def save_train_dict(train_dict, file_name):
    trains = pd.Series(train_dict)
    # 5시부터 다음날 새벽 4시 50분까지의 열차 수
    num_train = pd.DataFrame(trains)
    num_train.to_csv(file_name)

def calc_congest_against_trains(file_name, train_dict):
    data = pd.read_csv(file_name)
    congest = list((data.columns))
    train = list(train_dict.values())

    output = []
    for i in range(len(congest)):
        try:
            ans = float(congest[i])/train[i]
        except:
            ans=0
        output.append(ans)
    return output

def save_output(output, file_name):
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output)


if __name__ == "__main__":

    data = read_schedule(sys.argv[1])

    new_data = interest_data_preprocessing(data)
    dst_amsa = get_amsa(new_data)
    dst_moran = get_moran(new_data)
    timetable = make_timetable(dst_amsa, dst_moran)
    train_dict = make_train_dict(timetable)

    save_train_dict(train_dict, sys.argv[2])

    result = calc_congest_against_trains(sys.argv[3], train_dict)
    save_output(result, sys.argv[4])
