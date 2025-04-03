import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


#================================================================================
class Dataset_Custom(Dataset):
    def __init__(self, root_path, num_people=832, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'): # num_people=1이었음 scale=True  였음

        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.num_people = num_people
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        cols.remove('person_id')
        df_raw = df_raw[['person_id', 'date'] + cols + [self.target]]
        
        # Gather all data for scaling
        all_data = []
        for person_id in range(1, self.num_people + 1):
            df_person = df_raw[df_raw['person_id'] == person_id]
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_person.columns[2:]
                df_data = df_person[cols_data]
            elif self.features == 'S':
                df_data = df_person[[self.target]]
            all_data.append(df_data)
        
        all_data = pd.concat(all_data, axis=0)
        if self.scale:
            self.scaler.fit(all_data.values)

        # Process data for each person
        data_x, data_y, data_stamp = [], [], []
        for person_id in range(1, self.num_people + 1):
            df_person = df_raw[df_raw['person_id'] == person_id]
            num_vali = int(7)
            num_test = int(7)
            num_train = int(77-14)
            
            border1s = [0, num_train - self.seq_len, len(df_person) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_person)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_person.columns[2:]
                df_data = df_person[cols_data]
            elif self.features == 'S':
                df_data = df_person[[self.target]]

            if self.scale:
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_person[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp_person = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp_person = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp_person = data_stamp_person.transpose(1, 0)

            data_x.append(data[border1:border2])
            data_y.append(data[border1:border2])
            data_stamp.append(data_stamp_person)

        self.data_x = np.array(data_x)
        self.data_y = np.array(data_y)
        self.data_stamp = np.array(data_stamp)

    def __getitem__(self, index):
        person_idx = index // (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        seq_idx = index % (len(self.data_x[0]) - self.seq_len - self.pred_len + 1)
        
        s_begin = seq_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[person_idx, s_begin:s_end]
        seq_y = self.data_y[person_idx, r_begin:r_end]
        seq_x_mark = self.data_stamp[person_idx, s_begin:s_end]
        seq_y_mark = self.data_stamp[person_idx, r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x[0]) - self.seq_len - self.pred_len + 1) * self.num_people

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, num_people=832): # num_people=1이었음, inverse=False였음, scale=True였음
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # 초기화
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.num_people = num_people
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')

        df_raw = df_raw[['date'] + cols + [self.target]]

        data_list = []
        stamp_list = []
        for i in range(self.num_people):
            # 각 사람의 데이터 분할
            person_data = df_raw.iloc[i * len(df_raw) // self.num_people : (i + 1) * len(df_raw) // self.num_people]

            if self.features == 'M' or self.features == 'MS':
                cols_data = person_data.columns[1:]
                df_data = person_data[cols_data]
            elif self.features == 'S':
                df_data = person_data[[self.target]]

            if self.scale:
                self.scaler.fit(df_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            tmp_stamp = person_data[['date']]
            tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
            pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

            df_stamp = pd.DataFrame(columns=['date'])
            df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            data_list.append(data)
            stamp_list.append(data_stamp)

        self.data_x = np.array(data_list)
        self.data_stamp = np.array(stamp_list)

    def __getitem__(self, index):
        person_idx = index // (len(self.data_x[0]) - self.seq_len + 1)
        time_idx = index % (len(self.data_x[0]) - self.seq_len + 1)

        s_begin = time_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[person_idx, s_begin:s_end]
        seq_y = self.data_x[person_idx, r_begin:r_begin + self.label_len] if self.inverse else self.data_x[person_idx, r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[person_idx, s_begin:s_end]
        seq_y_mark = self.data_stamp[person_idx, r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x[0]) - self.seq_len + 1) * self.num_people

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)