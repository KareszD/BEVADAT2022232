import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NJCleaner:
    def __init__(self,path):
        self.data = pd.read_csv(path)

    def order_by_scheduled_time(self):
        return pd.DataFrame.sort_values(self.data, by=['scheduled_time'])

    def drop_columns_and_nan(self):
        drop = self.data.drop(['from','to'],axis=1)
        drop.dropna()

        return drop

    def convert_date_to_day(self):
        dates = self.data
        dates['date'] = pd.to_datetime(dates['date'])
        dates['day'] = dates['date'].dt.day_name()
        dates = dates.drop(['date'],axis=1)

        return dates

    def convert_scheduled_time_to_part_of_the_day(self):
        schedule = self.data
        schedule['scheduled_time'] = pd.DatetimeIndex(schedule['scheduled_time'])
        #pd.DatetimeIndex
        schedule.set_index(keys='scheduled_time',inplace=True)
        #schedule = pd.DataFrame(schedule,index=schedule['scheduled_time'])
        schedule['part_of_the_day'] = ""
        schedule.loc[schedule.between_time('4:00','7:59').index,'part_of_the_day'] = 'early_morning'
        schedule.loc[schedule.between_time('8:00','11:59').index,'part_of_the_day'] = 'morning'
        schedule.loc[schedule.between_time('12:00','15:59').index,'part_of_the_day'] = 'afternoon'
        schedule.loc[schedule.between_time('16:00','19:59').index,'part_of_the_day'] = 'evening'
        schedule.loc[schedule.between_time('20:00','23:59').index,'part_of_the_day'] = 'night'
        schedule.loc[schedule.between_time('0:00','3:59').index,'part_of_the_day'] = 'late_night'
        schedule.reset_index(drop=True,inplace=True)
        #schedule.drop(['scheduled_time'],axis=1)
        #return schedule
        #schedule.between_time('4:00','7:49')['part_of_the_day']="XD"
        #print(schedule.between_time('4:00','7:49'))
        return schedule

    def convert_delay(self):
        'delay_minutes'
        delayes = self.data
        delayes['delay'] = 0
        #delayes.loc[delayes['delay_minutes']< 5,['delay']] = 0
        delayes.loc[delayes['delay_minutes'] >= 5, ['delay']] = 1

        return delayes

    def drop_unnecessary_columns(self):
        dropos = self.data
        dropos.drop(['train_id','actual_time','delay_minutes'],axis=1,inplace=True)
        return dropos

    def save_first_60k(self,path):
        to_print = self.data.loc[:59999, :].copy()

        to_print.to_csv(path, index=False)


    def prep_df(self,path='data/NJ.csv'):
        self.data = self.order_by_scheduled_time()
        self.data = self.drop_columns_and_nan()
        self.data = self.convert_date_to_day()
        self.data = self.convert_scheduled_time_to_part_of_the_day()
        self.data = self.convert_delay()
        self.data = self.drop_unnecessary_columns()

        NJCleaner.save_first_60k(self,path)
        #self.data.head()

#njcl = NJCleaner('2018_03.csv')

#njcl.prep_df('lul.csv')

#print('XD')