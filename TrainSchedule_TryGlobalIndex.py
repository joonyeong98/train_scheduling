#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import numpy as np
import random
import pandas as pd
import od_lstm
import sys


# In[2]:


# 데이터 읽어오기

def read_line_dist (filename, line):
    # 하행 데이터 읽어오기
    data_des = pd.read_excel(filename, sheet_name=0)
    # 상행 데이터 읽어오기
    data_asc = pd.read_excel(filename, sheet_name=1)
    # 하행 호선
    data_des_line = data_des[(data_des['호선']==line)]
    # 상행 호선
    data_asc_line = data_asc[(data_asc['호선']==line)]
    return data_des_line, data_asc_line 


def read_line_info (filename, line):
    # 데이터 읽어오기
    data = pd.read_excel(filename)
    data_info= data[(data['호선']==line)]
    return data_info 


# In[3]:


# 시간 포맷 관련

def time_to_string(time):
    seconds = time.total_seconds()
    time_str = '%d:%02d:%02d' % (seconds / 3600, seconds / 60 % 60, seconds % 60)
    return time_str

def time_split(time_string):
    temp = time_string.to_string(index=False).split(':')
    #if len(temp) != 2:
    #    print("time_error")
    date = datetime.timedelta(minutes=int(temp[0]), seconds=int(temp[1]))
    return date

def index_to_time(index):
    return 16200 + (index * 600)


# In[4]:


# 역 클래스

class States:
    def __init__(self):
        self.reward = 0
        self.qvalue = [[-50,-50,-50],[-50,-50,-50]]
        #self.qvalue = [[0,0,0],[0,0,0]]
        #self.qvalue = [0,0,0]
        
    def q_update(self, action_, next_state, asc):
        if asc == 0:
            action = action_ - 3
        else:
            action = action_
        self.qvalue[asc][action] = (1-learning_rate) * self.qvalue[asc][action] + learning_rate * (next_state.reward + discount_factor*max(sum(next_state.qvalue,[])))
        #self.qvalue[action] = (1-learning_rate) * self.qvalue[action] + learning_rate * (self.reward + discount_factor*max(next_state.qvalue))
    def maxq(self,asc):
        #m = max(self.qvalue[asc])
        #temp = [i for i,v in enumerate(self.qvalue[asc]) if v == m]
        #return np.random.choice(temp)
        #max_idx_list = np.argwhere(self.qvalue[asc] == np.amax(self.qvalue[asc]))
        max_idx_list = np.argwhere(self.qvalue == np.amax(self.qvalue))
        max_idx_list = max_idx_list.flatten().tolist()
        return random.choice(max_idx_list)


# In[24]:


# 열차 학습 클래스

class TrainSchedule:
    def __init__(self, filenameinfo, filenamedist):
        self.global_index = 1
        self.asc_headway = datetime.timedelta(minutes=9, seconds=0)
        self.desc_headway = datetime.timedelta(minutes=9, seconds=0)
        self.asc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
        self.desc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
        self.train_descend_data, self.train_ascend_data = read_line_dist(filenamedist, 8)
        #self.info_data = read_line_info('ST_INFO.xlsx', 8)
        self.info_data = read_line_info(filenameinfo, 8)
        self.train_list = []
        self.state_index = 0
        for i in range(16):
            self.train_list.append(Train(self.info_data,self.train_descend_data,self.train_ascend_data))
        self.init_train()
        self.state_dict = {}
        self.state = []
        #for i in range(300000):
        #    self.state.append(States())
        self.state.append(States())
    
    def init_train(self):
        for i in range(16):
            self.train_list[i].asc = 1
            station_row = self.info_data[-2:-1]
            self.train_list[i].station = int(station_row['역 코드'])
            #self.train_list[i].index = i+1
            self.train_list[i].trip = 0
            self.train_list[i].arrive_time = datetime.timedelta(hours=0, minutes=0, seconds=0)

        self.train_list[11].asc = 0
        station_row = self.info_data[0:1]
        self.train_list[11].trip = 0
        self.train_list[11].station = int(station_row['역 코드'])
        self.train_list[12].asc = 0
        station_row = self.info_data[0:1]
        self.train_list[12].trip = 0
        self.train_list[12].station = int(station_row['역 코드'])
        self.train_list[13].asc = 0
        station_row = self.info_data[0:1]
        self.train_list[13].trip = 0
        self.train_list[13].station = int(station_row['역 코드'])
        self.train_list[14].asc = 0
        station_row = self.info_data[0:1]
        self.train_list[14].trip = 0
        self.train_list[14].station = int(station_row['역 코드'])
    
    def epsilon_greedy(self, current_state,asc,recent_action):
        if ((recent_action > 2 and asc == 1) or (recent_action < 3 and asc == 0)) and (epsilon/2 > random.random()) and (recent_action != 99):
            if asc == 1:
                action = recent_action - 3
            else:
                action = recent_action + 3
            return action
        elif epsilon > random.random():
            action = random.randrange(0,3)
        else:
            action = current_state.maxq(asc)
        
        if asc == 0:
            action = action + 3
            
        return action
            
    def optimize_step(self):
        self.init_train()
        self.asc_headway = datetime.timedelta(minutes=9, seconds=0)
        self.desc_headway = datetime.timedelta(minutes=9, seconds=0)
        self.asc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
        self.desc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
        
        current_time_asc = datetime.timedelta(hours=5, minutes=10, seconds=0)
        current_time_desc = datetime.timedelta(hours=5, minutes=15, seconds=0)
        current_state = self.state[0]
        action_list = [0]
        self.state_dict[tuple(action_list)] = current_state
        recent_action = 99
        
        while True:
            if current_time_asc < current_time_desc:
                asc = 1
                action = self.epsilon_greedy(current_state,asc,recent_action)
                index, trip = self.search_train(asc,current_time_asc)
                if index == 99:
                    current_time_asc = current_time_asc + datetime.timedelta(hours=0, minutes=0, seconds=30)
                    self.asc_headway += datetime.timedelta(seconds=30)
                    #action_list.pop()
                    continue
                self.calc_headway(action,asc)
                action_list.append(action)
                action_tuple = tuple(action_list)
                if action_tuple in self.state_dict :
                    next_state = self.state_dict[action_tuple]
                else:
                    self.state.append(States())
                    next_state = self.state[self.state_index+1]
                    self.state_index += 1
                total_time = self.train_list[index].operation(current_time_asc)
                current_time_asc = current_time_asc + self.asc_headway
                #current_state.reward = self.headway_hit(current_time_asc, self.asc_headway)
                next_state.reward = self.headway_hit(current_time_asc, self.asc_headway)
                self.state_dict[action_tuple] = next_state
                
            else:
                asc = 0
                action = self.epsilon_greedy(current_state,asc,recent_action)
                index, trip = self.search_train(asc,current_time_desc)
                if index == 99:
                    current_time_desc = current_time_desc + datetime.timedelta(hours=0, minutes=0, seconds=30)
                    self.desc_headway += datetime.timedelta(seconds=30)
                    #action_list.pop()
                    continue
                self.calc_headway(action,asc)
                if trip == 2:
                    self.desc_headway -= datetime.timedelta(seconds=30)
                action_list.append(action)
                action_tuple = tuple(action_list)
                if action_tuple in self.state_dict :
                    next_state = self.state_dict[action_tuple]
                else:
                    self.state.append(States())
                    next_state = self.state[self.state_index+1]
                    self.state_index += 1
                total_time = self.train_list[index].operation(current_time_desc)
                current_time_desc = current_time_desc + self.desc_headway
                #current_state.reward = self.headway_hit(current_time_desc, self.desc_headway)
                next_state.reward = self.headway_hit(current_time_desc, self.desc_headway)
                self.state_dict[action_tuple] = next_state
                
            if trip == 0:
                #current_state.reward = current_state.reward - 10
                next_state.reward = next_state.reward - 10
            current_state.q_update(action, next_state, asc)
            
            #print(time_to_string(current_time_asc),self.asc_headway,time_to_string(current_time_desc),self.desc_headway,current_state.reward, max_headway)
            #self.train_info()
            
            current_state = next_state
            recent_action=action
            #if asc == 0:
            #    recent_action -= 3
            
            if (current_time_asc > last_time and current_time_desc > last_time) :
                break
    
    #def train_info(self):
    #    for i in range(15):
    #        print(i, self.train_list[i].station, self.train_list[i].trip, self.train_list[i].arrive_time)
    
    def target_time_hit(self, current_time):
        global max_headway
        global min_headway
        
        """
        if (current_time >= target_time_start) and (current_time <= target_time_end):
            self.asc_headway_diff = datetime.timedelta(minutes=0, seconds=30)
            self.desc_headway_diff = datetime.timedelta(minutes=0, seconds=30)
            max_headway = datetime.timedelta(hours=0,minutes=6,seconds=0)
            min_headway = datetime.timedelta(hours=0,minutes=4,seconds=0)
            return 1
        elif (current_time >= target_time_start2) and (current_time <= target_time_end2):
            self.asc_headway_diff = datetime.timedelta(minutes=0, seconds=30)
            self.desc_headway_diff = datetime.timedelta(minutes=0, seconds=30)
            max_headway = datetime.timedelta(hours=0,minutes=6,seconds=0)
            min_headway = datetime.timedelta(hours=0,minutes=4,seconds=0)
            return 1
        else:
            self.asc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
            self.desc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
            max_headway = datetime.timedelta(hours=0,minutes=12,seconds=0)
            min_headway = datetime.timedelta(hours=0,minutes=7,seconds=0)
            return 0
        """
        
        for i in range(len(lstm.pred)):
            seconds = 16200 + (i * 600)
            if current_time.total_seconds() > seconds:
                if current_time.total_seconds() <= seconds + 600:
                    if lstm.pred[i] > 35:
                        self.asc_headway_diff = datetime.timedelta(minutes=0, seconds=30)
                        self.desc_headway_diff = datetime.timedelta(minutes=0, seconds=30)
                        max_headway = datetime.timedelta(hours=0,minutes=6,seconds=0)
                        min_headway = datetime.timedelta(hours=0,minutes=4,seconds=0)
                        return 1
                    else:
                        self.asc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
                        self.desc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
                        max_headway = datetime.timedelta(hours=0,minutes=12,seconds=0)
                        min_headway = datetime.timedelta(hours=0,minutes=7,seconds=0)
                        return 0
            else:
                return 0
                        
            
    
    def headway_hit(self, current_time, headway_):
        headway = (self.desc_headway + self.asc_headway) / 2
        reward = 0
        
        if self.target_time_hit(current_time) == 1:
            reward = 200 - abs((target_headway - headway).total_seconds())
            if reward == 200:
                reward += 3000
        else:
            reward = (headway - target_headway).total_seconds() / 4 - 20
            
        #if abs((self.asc_headway - self.desc_headway).total_seconds()) > 90:
        #    reward -= 200
            
        return reward
    
    def calc_headway(self, action, asc):
        #if asc == 1:
        if action == 0:
            if self.asc_headway > min_headway:
                if not (self.asc_headway - self.desc_headway).total_seconds() < -120:
                    self.asc_headway = self.asc_headway - self.asc_headway_diff
                else:
                    action = 1
            else:
                #self.asc_headway = min_headway
                action = 2
                self.asc_headway = self.asc_headway + self.asc_headway_diff
        elif action == 1:
            if self.asc_headway < min_headway:
                action = 2
                self.asc_headway = self.asc_headway + self.asc_headway_diff
            if self.asc_headway > max_headway:
                action = 0
                self.asc_headway = self.asc_headway - self.asc_headway_diff
        elif action == 2:
            if self.asc_headway < max_headway:
                if not (self.asc_headway - self.desc_headway).total_seconds() > 120:
                    self.asc_headway = self.asc_headway + self.asc_headway_diff
                else:
                    action = 1
            else:
                #self.asc_headway = max_headway
                action = 0
                self.asc_headway = self.asc_headway - self.asc_headway_diff

        #elif asc == 0:
        if action == 3:
            if self.desc_headway > min_headway:
                if not (self.asc_headway - self.desc_headway).total_seconds() > 120:
                    self.desc_headway = self.desc_headway - self.desc_headway_diff
                else:
                    action = 4
            else:
                #self.desc_headway = min_headway
                action = 5
                self.desc_headway = self.desc_headway + self.desc_headway_diff    
        elif action == 4:
            if self.desc_headway < min_headway:
                action = 5
                self.desc_headway = self.desc_headway + self.desc_headway_diff
            if self.desc_headway > max_headway:
                action = 3
                self.desc_headway = self.desc_headway - self.desc_headway_diff
        elif action == 5:
            if self.desc_headway < max_headway:
                if not (self.asc_headway - self.desc_headway).total_seconds() < -120:
                    self.desc_headway = self.desc_headway + self.desc_headway_diff    
                else:
                    action = 4
            else:
                #self.desc_headway = max_headway
                action = 3
                self.desc_headway = self.desc_headway - self.desc_headway_diff 
            
        #elif asc == 0:
        #if action == 3:
        #    if self.asc_headway > min_headway:
        #        self.asc_headway = self.asc_headway - self.asc_headway_diff
        #elif action == 4:
        #    pass
        #elif action == 5:
        #    if self.asc_headway < max_headway:
        #        self.asc_headway = self.asc_headway + self.asc_headway_diff
        #self.desc_headway = self.asc_headway
                
    def search_train(self, asc, current_time):
        minimum_time = datetime.timedelta(hours=55, minutes=0, seconds=0)
        find_index = 99
        trip = 1
        temp_index = []
        
        for i in range(16):
            temp_train = self.train_list[i]
            if temp_train.asc == asc:
                if temp_train.trip == 1:
                    if minimum_time > temp_train.arrive_time:
                        if current_time - temp_train.arrive_time > max_trip :
                            if asc == 1:
                                temp_train.trip = 0
                                continue
                            elif asc == 0:
                                minimum_time = temp_train.arrive_time
                                find_index = i
                                trip = 2
                                continue
                                #return i, 2
                        if current_time - temp_train.arrive_time < min_trip :
                            continue
                        minimum_time = temp_train.arrive_time
                        find_index = i
                    
                        
        if find_index == 99:
            for i in range(16):
                temp_train = self.train_list[i]
                if temp_train.asc == asc:
                    if temp_train.trip == 0:
                        temp_index.append(i)
            if len(temp_index) > 0:
                find_index = random.choice(temp_index)
                
        return find_index, trip
    
    def learn(self):
        epsilon = 1
        for i in range(int(epoch + 1)):
            self.optimize_step()
            epsilon = epsilon-epsilon_decrease
        epsilon = 0
        self.optimize_step()
        

    def run(self):
        self.init_train()
        self.asc_headway = datetime.timedelta(minutes=9, seconds=0)
        self.desc_headway = datetime.timedelta(minutes=9, seconds=0)
        self.asc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
        self.desc_headway_diff = datetime.timedelta(minutes=1, seconds=0)
        
        current_time_asc = datetime.timedelta(hours=5, minutes=10, seconds=0)
        current_time_desc = datetime.timedelta(hours=5, minutes=15, seconds=0)
        current_state = self.state[0]
        action_list = [0]
        self.state_dict[tuple(action_list)] = current_state
        recent_action = 99
        
        while True:
            if current_time_asc < current_time_desc:
                asc = 1
                action = self.epsilon_greedy(current_state,asc,recent_action)
                index, trip = self.search_train_run(asc,current_time_asc)
                if index == 99:
                    current_time_asc = current_time_asc + datetime.timedelta(hours=0, minutes=0, seconds=30)
                    self.asc_headway += datetime.timedelta(seconds=30)
                    continue
                self.calc_headway(action,asc)
                action_list.append(action)
                action_tuple = tuple(action_list)
                if action_tuple in self.state_dict :
                    next_state = self.state_dict[action_tuple]
                else:
                    self.state.append(States())
                    next_state = self.state[self.state_index+1]
                    self.state_index += 1
                total_time = self.train_list[index].full_operation(current_time_asc,index)
                current_time_asc = current_time_asc + self.asc_headway
                next_state.reward = self.headway_hit(current_time_asc, self.asc_headway)
                self.state_dict[action_tuple] = next_state
                
            else:
                asc = 0
                action = self.epsilon_greedy(current_state,asc,recent_action)
                index, trip = self.search_train_run(asc,current_time_desc)
                if index == 99:
                    current_time_desc = current_time_desc + datetime.timedelta(hours=0, minutes=0, seconds=30)
                    self.desc_headway += datetime.timedelta(seconds=30)
                    continue
                self.calc_headway(action,asc)
                if trip == 2:
                    self.desc_headway -= datetime.timedelta(seconds=30)
                action_list.append(action)
                action_tuple = tuple(action_list)
                if action_tuple in self.state_dict :
                    next_state = self.state_dict[action_tuple]
                else:
                    self.state.append(States())
                    next_state = self.state[self.state_index+1]
                    self.state_index += 1
                total_time = self.train_list[index].full_operation(current_time_desc,index)
                current_time_desc = current_time_desc + self.desc_headway
                next_state.reward = self.headway_hit(current_time_desc, self.desc_headway)
                self.state_dict[action_tuple] = next_state
                
            if trip == 0:
                current_state.reward = current_state.reward - 10
            current_state.q_update(action, next_state, asc)
            
            #print(time_to_string(current_time_asc),self.asc_headway,time_to_string(current_time_desc),self.desc_headway,current_state.reward)
            #self.train_info()
            
            current_state = next_state
            recent_action=action
            if asc == 0:
                recent_action -= 3
            
            if (current_time_asc > last_time and current_time_desc > last_time) :
                break                
        
    def search_train_run(self, asc, current_time):
        minimum_time = datetime.timedelta(hours=55, minutes=0, seconds=0)
        find_index = 99
        trip = 1
        temp_index = []
        
        for i in range(16):
            temp_train = self.train_list[i]
            if temp_train.asc == asc:
                if temp_train.trip == 1:
                    if minimum_time > temp_train.arrive_time:
                        if current_time - temp_train.arrive_time > max_trip :
                            if asc == 1:
                                temp_train.trip = 0
                                temp_train.inbound()
                                continue
                            elif asc == 0:
                                minimum_time = temp_train.arrive_time
                                find_index = i
                                trip = 2
                                continue
                                #return i, 2
                        if current_time - temp_train.arrive_time < min_trip :
                            continue
                        minimum_time = temp_train.arrive_time
                        find_index = i
                        
        if find_index == 99:
            for i in range(16):
                temp_train = self.train_list[i]
                if temp_train.asc == asc:
                    if temp_train.trip == 0:
                        temp_index.append(i)
            if len(temp_index) > 0:
                find_index = random.choice(temp_index)
                self.train_list[find_index].index = self.global_index
                self.global_index += 1
                if self.train_list[find_index].asc == 1:
                    self.train_list[find_index].outbound(current_time)
                
        return find_index, trip
                
    def create_shcedule(self):
        total_schedule = pd.concat([self.train_list[0].train_schedule,self.train_list[1].train_schedule], ignore_index=True)
        for i in range(16):
            if i == 0 or i ==1:
                continue
            total_schedule = pd.concat([total_schedule,self.train_list[i].train_schedule], ignore_index=True)
            
        total_schedule.sort_values(by=['운행 번호','운행 순서','열번순서'],axis=0,inplace=True)
        print(total_schedule)
        total_schedule.to_excel("TRAIN_SCHD_8_1.xlsx",index = False)
            
        
    


# In[16]:


# 열차 클래스

class Train:
    def __init__(self,data,d_data,a_data):
        #self.descend_data, self.ascend_data = read_line_dist('ST_DIST.xlsx', 8)
        self.descend_data = d_data
        self.ascend_data = a_data
        self.info_data = data
        self.asc = 1
        self.index = 0
        station_row = self.info_data[-2:-1]
        self.station = int(station_row['역 코드'])
        self.trip = 0
        self.arrive_time = datetime.timedelta(hours=0, minutes=0, seconds=0)
        self.train_schedule = pd.DataFrame(columns=['운행 번호','운행 순서','열번','열번순서','내/외선','소속','차량 종류','열차 주행거리(km)','자선 주행거리(km)','타선 주행거리(km)','열차 운행 시간(s)','출발역코드','출발역','출발시간','도착역코드','도착역','도착시간','기운영타입'])
        self.seq = 1
        self.driven = 0      
        
    def operation(self,current_time):
        total_time = datetime.timedelta(hours=0, minutes=0, seconds=0)
        
        while True:
            if self.asc == 1:
                temp_data = self.ascend_data[(self.ascend_data['출발역코드']==self.station)]
            elif self.asc == 0:
                temp_data = self.descend_data[(self.descend_data['출발역코드']==self.station)]
            else:
                pass
            
            if len(temp_data) <=0:
                break
            if (self.asc==0 and int(temp_data['도착역코드'] == 99)):
                break
            self.station = int(temp_data['도착역코드'])
            total_time = total_time + time_split(temp_data['운행시간'])
            
        if self.asc == 1:
            self.asc = 0
        elif self.asc == 0:
            self.asc = 1
            
            
        self.arrive_time = current_time + total_time
        self.trip = 1
        
        return total_time
    
    def inbound(self):
        temp_driven = self.driven + 1.9
        temp_time = self.arrive_time + datetime.timedelta(minutes=5)
        self.train_schedule.loc[len(self.train_schedule)] = [self.index, self.seq, 8000+self.index, 1, 1, 'S', 'DC', temp_driven, '1.9' ,'0','05:00', '17','모란',time_to_string(self.arrive_time),'99','모란기지',time_to_string(temp_time), 0]
        self.index = self.index + 16
        self.driven = 0
        self.seq = 1
        
    def outbound(self,current_time):
        self.driven += 1.9
        temp_time = current_time - datetime.timedelta(minutes=5)
        self.train_schedule.loc[len(self.train_schedule)] = [self.index, self.seq, 8000+self.index, 1, 2, 'S', 'DC', self.driven, '1.9' ,'0','05:00', '99','모란기지',time_to_string(temp_time),'17','모란',time_to_string(current_time), 0]
        self.seq += 1
        self.arrive_time = current_time
        
    def full_operation(self,current_time,index):
        total_time = datetime.timedelta(hours=0, minutes=0, seconds=0)
        temp_time = datetime.timedelta(hours=0, minutes=0, seconds=0)
        k = 1
        
        while True:
            if self.asc == 1:
                temp_data = self.ascend_data[(self.ascend_data['출발역코드']==self.station)]
            elif self.asc == 0:
                temp_data = self.descend_data[(self.descend_data['출발역코드']==self.station)]
            else:
                pass
            
            if len(temp_data) <=0:
                break
            if (self.asc==0 and int(temp_data['도착역코드'] == 99)):
                break
            if (self.asc==0 and int(temp_data['도착역코드'] == 2)):
                self.arrive_time = current_time
            if (self.asc==1 and int(temp_data['도착역코드'] == 16)):
                self.arrive_time = current_time
            if self.arrive_time.total_seconds() == 0:
                self.arrive_time = current_time
            self.driven += float(temp_data['자선거리'])
            self.station = int(temp_data['도착역코드'])
            temp_time = self.arrive_time
            self.arrive_time = self.arrive_time + time_split(temp_data['운행시간'])
            total_time = total_time + time_split(temp_data['운행시간'])
            self.train_schedule.loc[len(self.train_schedule)] = [self.index, self.seq, 8000+self.index, k, self.asc + 1, 'S', 'DC', self.driven,float(temp_data['자선거리']) ,'0',temp_data['운행시간'].values[0], int(temp_data['출발역코드']),temp_data['출발역'].values[0],time_to_string(temp_time),int(temp_data['도착역코드']),temp_data['도착역'].values[0],time_to_string(self.arrive_time), 0]
            k += 1
            
        if self.asc == 1:
            self.asc = 0
        elif self.asc == 0:
            self.asc = 1
            
        self.seq += 1
        self.trip = 1
        
        return total_time
    


# In[7]:


if __name__ == "__main__":
    
    # Hyperparameter
    discount_factor = 0.9
    learning_rate = 0.1
    epsilon = 1
    epoch = 200
    epsilon_decrease = 1/epoch

    max_trip = datetime.timedelta(hours=0,minutes=15,seconds=0)
    min_trip = datetime.timedelta(hours=0,minutes=6,seconds=0)

    max_headway = datetime.timedelta(hours=0,minutes=12,seconds=0)
    min_headway = datetime.timedelta(hours=0,minutes=7,seconds=0)

    #target_headway = datetime.timedelta(hours=0,minutes=4,seconds=30)
    temp_arg_time = sys.argv[5].split(':')
    target_headway = datetime.timedelta(minutes=int(temp_arg_time[0]), seconds=int(temp_arg_time[1]))

    #target_time_start = datetime.timedelta(hours=6,minutes=0,seconds=0)
    #target_time_end = datetime.timedelta(hours=9,minutes=0,seconds=0)
    #target_time_start2 = datetime.timedelta(hours=17,minutes=0,seconds=0)
    #target_time_end2 = datetime.timedelta(hours=20,minutes=0,seconds=0)
    
    last_time = datetime.timedelta(hours=25, minutes=30, seconds=0)
    
    # 혼잡도
    #lstm = od_lstm.lstm("./congest data/Weekday/20190107_혼잡도(10분 단위).xlsx","./congest data/Weekday/20190408_혼잡도(10분 단위).xlsx")
    lstm = od_lstm.lstm(sys.argv[3],sys.argv[4])

    lstm.learn()
    
    print("혼잡도 예측 완료!", file=sys.stdout)
    print("학습 시작...", file=sys.stdout)
    
    #학습
    ts = TrainSchedule(sys.argv[1], sys.argv[2])
    ts.learn()
    ts.run()
    ts.create_shcedule()

    print("학습 완료!", file=sys.stdout)


