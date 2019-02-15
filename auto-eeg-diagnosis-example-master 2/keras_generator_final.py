from dataset import get_all_sorted_file_names_and_labels
import numpy as np
from sklearn.externals import joblib
from random import shuffle
import config
from auto_diagnosis import run_exp
from keras.utils import Sequence

def pad_list(listy):
    a=10-len(listy)
    new_list=[0]*a
    return listy+new_list

def yword_to_token(y):
    word_to_token=joblib.load('words_to_tokens')

    t2c=joblib.load('token_to_cui')
    t2c[1]='NOCUI'
    #new_list=[]
    token_list=[]
    for i in y.tolist():
        stringy='ssss '
        listy=[1]
        temp=i.split(" ")
        if len(temp) > 10: #terminate sequences longer than 30
            temp=temp[0:9]
        
        for j in temp:
            try:
                token=word_to_token[j.lower()]
                if t2c[token] != 'NOCUI':
                    stringy+=j.lower()+' '
                    listy.append(float(token))
            except:

                try:
                    token=word_to_token[j[:-1].lower()]
                    if t2c[token] != 'NOCUI':
                        stringy+=j.lower()+' '
                        listy.append(float(token))
                except:
                    pass
        stringy+='eeee'
        listy.append(2)
        #new_list.append(stringy)
        if len(listy)<10:
            listy=pad_list(listy)
        token_list.append(listy)
    return np.array(token_list)

class Mygenerator(Sequence): 
    def __init__(self, batch_size):
        self.batch_size=batch_size
        self.file_list,labels=get_all_sorted_file_names_and_labels('train',['../normal/',
    '../abnormal/'])
        assert len(self.file_list)==2717
        a=list(range(2717))
        shuffle(a)
        matching_dict={}
        for i in range(2717):
            matching_dict[a[i]]=self.file_list[i]
        self.dict=matching_dict
    
    def __len__(self):
        return int(np.ceil(2717/float(self.batch_size)))

    def __getitem__(self,idx):
        start=idx*self.batch_size
        end=(idx+1)*self.batch_size
        if end> 2717:
            end=2717
        file_list=[]
        for i in range(start,end):
            file_list.append(self.dict[i])
        X,y,y_bin=run_exp(config.data_folders,config.n_recordings,config.sensor_types,config.n_chans,config.max_recording_mins,config.sec_to_cut, config.duration_recording_mins,config.test_recording_mins,config.max_abs_val,config.sampling_freq,
        config.divisor,
        config.test_on_eval,
        config.n_folds, config.i_test_fold,
        config.shuffle,
        config.model_name,
        config.n_start_chans, config.n_chan_factor,
        config.input_time_length, config.final_conv_length,
        config.model_constraint,
        config.init_lr,
        config.batch_size, config.max_epochs,config.cuda,file_list)
        y=yword_to_token(y)
        a=y_bin.shape
        y_bin=np.reshape(y_bin,(a[0],1))
        
        y_input = y[:, 0:-1]
        y_output = y[:, 1:]
        y_output = np.expand_dims(y_output, axis=-1)
        return [X,y_input],[y_bin, y_output]
    
class MygeneratorVal(Sequence): 
    def __init__(self):
        self.file_list,labels=get_all_sorted_file_names_and_labels('eval',['../normal/',
    '../abnormal/'])
        
    
    def __len__(self):
        return 276

    def __getitem__(self,idx):
        file_list=[self.file_list[idx]]
        X,y,y_bin=run_exp(config.data_folders,config.n_recordings,config.sensor_types,config.n_chans,config.max_recording_mins,config.sec_to_cut, config.duration_recording_mins,config.test_recording_mins,config.max_abs_val,config.sampling_freq,
        config.divisor,
        config.test_on_eval,
        config.n_folds, config.i_test_fold,
        config.shuffle,
        config.model_name,
        config.n_start_chans, config.n_chan_factor,
        config.input_time_length, config.final_conv_length,
        config.model_constraint,
        config.init_lr,
        config.batch_size, config.max_epochs,config.cuda,file_list)
        print('1',X)
        y=yword_to_token(y)
        a=y_bin.shape
        y_bin=np.reshape(y_bin,(a[0],1))
        print('y',y)
        y_input = y[:, 0:-1]
        y_output = y[:, 1:]
        y_output = np.expand_dims(y_output, axis=-1)
        return [X,y_input],[y_bin, y_output]


        
        
