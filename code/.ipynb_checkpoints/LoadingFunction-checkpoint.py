from sklearn.utils import shuffle
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc

datalabel="ATP_Binding"


def data_label():
    return datalabel

def MCNN_data_load(DATA_TYPE,NUM_CLASSES,NUMDEPENDENT):
    MAXSEQ=NUMDEPENDENT*2+1
    
    path_x_train = "/media/b1607/X/backup/atp_binding/github/dataset/Series"+str(MAXSEQ)+"/"+DATA_TYPE+"/atp-388"
    path_y_train = "/media/b1607/X/backup/atp_binding/github/data/FASTA/atp-388/label"
    #/media/b1607/X/backup/atp_binding/github/dataset/Series11/atp-388
    #print(path_x_train)
    #print(path_y_train)
    x_train,y_train=data_load(path_x_train,path_y_train,NUM_CLASSES)
    path_x_test = "/media/b1607/X/backup/atp_binding/github/dataset/Series"+str(MAXSEQ)+"/"+DATA_TYPE+"/atp-41"
    path_y_test ="/media/b1607/X/backup/atp_binding/github/data/FASTA/atp-41/label"
    #print(path_x_test)
    #print(path_y_test)
    x_test,y_test=data_load(path_x_test,path_y_test,NUM_CLASSES)
    
    return(x_train,y_train,x_test,y_test)

def data_load(x_folder, y_folder,NUM_CLASSES,):
    x_train = []
    y_train = []

    x_files = [file for file in os.listdir(x_folder) if file.endswith('.set.npy')]
    
    # Iterate through x_folder with tqdm
    for file in tqdm(x_files, desc="Loading data", unit="file"):
        x_path = os.path.join(x_folder, file)
        x_data = np.load(x_path)
        
        x_train.append(x_data.astype('float16'))
        #x_train.append(x_data)

        # Get the corresponding y file
        y_file = file[:-8] + '.label'
        y_path = os.path.join(y_folder, y_file)

        with open(y_path, 'r') as y_f:
            lines = y_f.readlines()
            y_data = np.array([int(x) for x in lines[1].strip()])
            y_train.append(y_data.astype('float16'))
            
            del y_data
            gc.collect()
            
        del x_data
        del y_file
        gc.collect()
        
    # Concatenate all the data
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Add new dimensions to x_train and y_train
    x_train = np.expand_dims(x_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
    
    del x_files
    gc.collect()
    
    return x_train, y_train