# %% [code]
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime as dt
import pickle
import zipfile
import PIL
import shutil
from IPython.display import FileLink
from time import time
from IPython.display import display
import numpy as np # linear algebra
import IPython.display as ipd
np.warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.client import device_lib

import os

global last_pred

def check_GPU_ON():
    d_list_devices = device_lib.list_local_devices()
    d_list_devices_GPU = [x.name for x in d_list_devices if 'GPU' in x.name]
    print ('GPU подключен') if d_list_devices_GPU else  print('!АХТУНГ! GPU не подключен')
    return

def clean_HDD_befor_RUN_and_SAVE (d_PATH_to_WORKDIR,
                                d_list_dir, 
                                d_list_file, 
                                d_list_masks):
    print('Очистка HDD kaggle перед Run and Save:')
    if (d_list_dir != []) and (d_list_dir != ''):
        print('Удаление папок из списка:')
        for i, d_path_dir in enumerate(d_list_dir):
            
            if os.path.exists(d_path_dir):
                print(f'{i+1}. Ссылка на ({d_path_dir}) существует.', end='')
                if os.path.isdir(d_path_dir):
                    shutil.rmtree(d_path_dir)
                    print(f'..... Папка удалена.)')
                else:
                    print(f'Это не папка.  .........   АХТУНГ!АХТУНГ!АХТУНГ!')
            else:
                print(f'{i+1}. Ссылка на папку ({d_path_dir}) НЕ существует.  .........   АХТУНГ!АХТУНГ!АХТУНГ!')
        print('===')
    if (d_list_file != []) and (d_list_file != ''):
        print('Удаление файлов из списка:')
        for i, d_file in enumerate(d_list_file):
            
            d_path_file = d_PATH_to_WORKDIR+d_file
            
            if os.path.exists(d_path_file):
                print(f'{i+1}. Ссылка на ({d_file}) существует.', end='')
                if os.path.isfile(d_path_file):
                    os.remove(d_path_file)
                    print('.... Файл удален.)')
                else:
                    print(f'Это не файл.  .........   АХТУНГ!АХТУНГ!АХТУНГ!')

            else:
                print(f'{i+1}. Ссылка на файл ({d_file}) НЕ существует.  .........   АХТУНГ!АХТУНГ!АХТУНГ!')
        print('===')
    if (d_list_masks != []) and (d_list_masks != ''):
        print('Удаление файлов по маске из списка:')
        d_list_from_dir = os.listdir(d_PATH_to_WORKDIR)
        for d_mask in d_list_masks:    
            d_sum = 0
            for item in d_list_from_dir:
                if d_mask in item:
                    d_path_item = d_PATH_to_WORKDIR+item
                    os.remove(d_path_item)
                    print(f'{d_sum+1}. Файл ({item}) удален.)')
                    d_sum += 1
        if d_sum == 0:
            print(f'Файлов по маскам из списка не найдено')
        else:
            print(f'Всего удалено:= {d_sum} файлов')
        print('===')
        return

def plot_res_dif_exp_in_one(
        d_list_of_num_exp,
        d_list_title,
        d_df_results,
        d_sdvig):
    
    temp_df = d_df_results[d_df_results['NUM_EXP'].isin(d_list_of_num_exp)]
    
    len_exp = len(d_list_of_num_exp)
    d_str_title = ''
    for i in range(len_exp):
        d_str_title += str(d_list_of_num_exp[i])+', '
    d_str_title = d_str_title[:-2]
    
    y=np.array([[0.0 for j in range(len_exp)] for i in range(4)])
    
    for d_num_exp in range(len_exp):
        temp_df2 = temp_df[temp_df['NUM_EXP']==d_list_of_num_exp[d_num_exp]]
        
        list1 = temp_df2['R_VAL_ACC'].values[0][1:-1].split(', ')
        y1 = [float(i) for i in list1]
        
        max1 = np.max(y1)
        
        y[1,d_num_exp] = max1
        
        ind_max1 = y1.index(max1)
        
        list2 = temp_df2['R_ACC'].values[0][1:-1].split(', ')
        y2 = [float(i) for i in list2]
        y[0,d_num_exp] = y2[ind_max1]
        
        list3 = temp_df2['R_LOSS'].values[0][1:-1].split(', ')
        y3 = [float(i) for i in list3]
        y[2,d_num_exp] = y3[ind_max1]
        
        list4 = temp_df2['R_VAL_LOSS'].values[0][1:-1].split(', ')
        y4 = [float(i) for i in list4]
        y[3,d_num_exp] = y4[ind_max1]
        
    
    
    
    
    x = range(0,len(d_list_of_num_exp))
    
    min_y1 = min(np.min(y[0,:]), np.min(y[1,:]))
    max_y1 = max(np.max(y[0,:]), np.max(y[1,:]))
    
    min_y2 = min(np.min(y[2,:]), np.min(y[3,:]))
    max_y2 = max(np.max(y[2,:]), np.max(y[3,:]))
    
    
    k_razlet = 2

#     Plot Line1 (Left Y Axis)
    
    plt.style.use('seaborn-paper')
    sns.set(font_scale=1.1)
    color_text = plt.get_cmap('PuBu')(0.85)
    color_line1 = plt.get_cmap('PuBu')(0.95)
    color_line2 = plt.get_cmap('PuBu')(0.65)

    fig, ax1 = plt.subplots(1,1,figsize=(12,7), dpi= 80)
    ax1.plot(x, y[0,:], color=color_line1, lw=3, marker = 'o', ms = 10, label='acc')
    ax1.plot(x, y[1,:], color=color_line2, ls = '--', marker = 'o', ms = 10, label='val_acc')
    

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y[2,:], color='tab:red', lw=3, marker = 'o', ms = 10, label='loss')
    ax2.plot(x, y[3,:], color='lightcoral', ls = '--', marker = 'o', ms = 10, label='val_loss')

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel('Номер эксперимента', fontsize=20, color = color_text)
    ax1.tick_params(axis='x', rotation=0, labelsize=12, labelcolor=color_text)
    ax1.set_ylabel('Точность (accuracy)', color=color_line1, fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor=color_line1)
    ax1.minorticks_on()
    
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.8)
    
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    ax1.set_ylim(min_y1-k_razlet*(max_y1-min_y1), max_y1*1.05)
    
    y_title = (max_y1*1.05 + (min_y1-k_razlet*(max_y1-min_y1)))/2*1.1
    
    for i_t, title in enumerate(d_list_title):
        plt.text(i_t - i_t/len_exp*d_sdvig, y_title, title, 
             fontsize = 18, 
             color = color_text)
        if i_t > 0:
            proc = y[1,:][i_t]/y[1,:][i_t-1]
            proc = round((proc-1)*100,2)
            ax1.text(i_t - i_t/len_exp*d_sdvig*0.8, y[1,:][i_t]*0.95, f'+{proc}%', 
                 fontsize = 20, 
                 color = 'black')
            
            

    # ax2 (right Y axis)
    ax2.set_ylabel("loss-функция", color='tab:red', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_xticks(np.arange(0, len(d_list_of_num_exp), 1))
    ax2.set_xticklabels(d_list_of_num_exp, rotation=90, fontdict={'fontsize':10})
    ax2.set_title(f'Сравнение результатов экспериментов ({d_str_title})', 
                  fontsize=22, 
                  color = color_line1)
    ax2.set_ylim(min_y2*0, max_y2+(max_y2-min_y2)*k_razlet)
    
    for i_t, title in enumerate(d_list_title):
        if i_t > 0:
            proc2 = y[3,:][i_t]/y[3,:][i_t-1]
            proc2 = round((1-proc2)*100,2)
            ax2.text(i_t - i_t/len_exp*d_sdvig*0.6, y[3,:][i_t]*1.2, f'-{proc2}%', 
                 fontsize = 20, 
                 color = 'black')
    fig.legend(bbox_to_anchor=(0.5, -0.015), loc='lower center', ncol=4)
    fig.tight_layout()
    plt.show()
    return

def soun_of_end(d_rate):
    beep = np.sin(2*np.pi*400*np.arange(10000*2)/10000)
    display(ipd.Audio(beep, rate=d_rate, autoplay=True))
    return

def show_result_exp(d_num_exp, 
                    d_df_results, 
                    d_dict,
                    d_title):

    def str_to_arr(d_str):
        d_temp_list = d_str[1:-1].split(' ')
        d_clean_temp_list = [x for x in d_temp_list if (x != '') and (x !='...')]
        d_temp = [float(i) for i in d_clean_temp_list]
        d_temp_arr = np.array(d_temp)
        return d_temp_arr

    def descr_after_point(d_str):
        d_ind = d_str.index('.')
        return d_str[d_ind+2:]

    print(f'Гиперпараметры и результаты обучения нейросети по эксперименту:= {d_num_exp} (NUM_EXP){d_title}')
    temp_df = d_df_results[d_df_results['NUM_EXP']==d_num_exp]

    temp_dict = {}
    temp_dict['----Гиперпараметры аугментации----'] = '',''
    for col in temp_df.columns:
        if 'AUG_' in col:
            temp_dict[col] = temp_df[col].values[0], descr_after_point(d_dict[col])
    
    temp_dict['----Гиперпараметры модели----'] = '',''
    for col in temp_df.columns:
        if 'M_' in col and (col != 'NUM_EXP'):
            temp_dict[col] = temp_df[col].values[0], descr_after_point(d_dict[col])
    
    temp_dict['----Гиперпараметры головы----'] = '',''
    for col in temp_df.columns:
        if ('H_' in col) and ('R_' not in col):
            temp_dict[col] = temp_df[col].values[0], descr_after_point(d_dict[col])
    
    temp_dict['----Гиперпараметры компиляции----'] = '',''
    for col in temp_df.columns:
        if 'C_' in col:
            temp_dict[col] = temp_df[col].values[0], descr_after_point(d_dict[col])

    temp_dict['----Результаты обучения----'] = '',''
    col = 'R_EVA_VAL_ACC'
    temp_dict[col] = temp_df[col].values[0], descr_after_point(d_dict[col])
    col = 'R_EPOCH_TIME'
    temp_arr = str_to_arr(temp_df[col].values[0])
    temp_dict[col+'_mean'] = temp_arr.mean(), 'Среднее время обучения каждой эпохи (сек.)'
    
    col = 'R_BATCH_TIME'
    temp_arr = str_to_arr(temp_df[col].values[0])
    temp_dict[col+'_mean'] = temp_arr.mean(), 'Среднее время обучения каждого batch (сек.)'

    col = 'R_TRAIN_TIME'
    temp_dict[col] = temp_df[col].values[0], descr_after_point(d_dict[col])

    temp_df2 = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['Значение', 'Описание'])
    display(temp_df2)
    return

def new_exp_without_stop_session(d_results_of_exp):
    last_NUM_EXP = d_results_of_exp.loc[d_results_of_exp.index.max(),'NUM_EXP']
    d_results_of_exp.loc[d_results_of_exp.index.max()+1]=d_results_of_exp.loc[d_results_of_exp.index.max()]
    d_results_of_exp.loc[d_results_of_exp.index.max(), 'NUM_EXP']=last_NUM_EXP+1
    for col in d_results_of_exp.columns:
        if 'R_' in col:
            d_results_of_exp.loc[d_results_of_exp.index.max(), col]=''
    d_results_of_exp.loc[d_results_of_exp.index.max(), 'TIME']=''
    print(f'Новый эксперимент без завершения сессии инициализирован.')
    print(f'NUM_EXP = {last_NUM_EXP+1} ...')
    return last_NUM_EXP+1

def show_image(d_path, d_subtitle, d_x, d_y, d_y_subtitle):
    plt.style.use('seaborn-paper')
    color_text = plt.get_cmap('PuBu')(0.95)

    plt.figure(figsize=(d_x,d_y))
    plt.subplot(1, 1, 1)
    im = PIL.Image.open(d_path)
    plt.imshow(im)
    plt.suptitle(d_subtitle, y = d_y_subtitle, fontsize = 18, color = color_text)
    plt.axis('off')
    return


def load_result_last_cnn_fit(d_PATH_to_FILE_RESULT, d_PATH_to_FILE_descr):
    temp_df = pd.read_csv(d_PATH_to_FILE_RESULT)
    with open(d_PATH_to_FILE_descr, 'rb') as f:
        temp_dict = pickle.load(f)
    temp = temp_df['NUM_EXP'].to_numpy()[-1].max()
    print('Инициализация нового эксперимента после загрузки результатов предыдущих экспериментов')
    d_NUM_EXP = hyperp('NUM_EXP',int(temp+1),'',temp_df,temp_dict)
    return d_NUM_EXP, temp_df, temp_dict

def plot_acc_loss_fit_model_in_one(d_num_exp, 
                                   d_df_results,
                                   d_title):
    
    temp_df = d_df_results[d_df_results['NUM_EXP']==d_num_exp]
    
    d_date = temp_df['TIME'].values[0]
    d_evol_val_acc = float(temp_df['R_EVA_VAL_ACC'].values[0])
    
    d_time = int(float(temp_df['R_TRAIN_TIME'].values[0][1:-1]))
    
    list1 = temp_df['R_ACC'].values[0][1:-1].split(', ')
    y1 = [float(i) for i in list1]
    list2 = temp_df['R_VAL_ACC'].values[0][1:-1].split(', ')
    y2 = [float(i) for i in list2]
    list3 = temp_df['R_LOSS'].values[0][1:-1].split(', ')
    y3 = [float(i) for i in list3]
    list4 = temp_df['R_VAL_LOSS'].values[0][1:-1].split(', ')
    y4 = [float(i) for i in list4]
    x = range(1,len(y1)+1)
    
    temp_list = y3+y4
    min_y, max_y = min(temp_list), max(temp_list)
    y_for_text = (min_y+max_y)*0.4
    x_for_text = len(y1)*0.75

    # Plot Line1 (Left Y Axis)
    
    plt.style.use('seaborn-paper')
    sns.set(font_scale=1.1)
    color_text = plt.get_cmap('PuBu')(0.85)
    color_line1 = plt.get_cmap('PuBu')(0.95)
    color_line2 = plt.get_cmap('PuBu')(0.65)

#     plt.figure(figsize=(12, 7))
    fig, ax1 = plt.subplots(1,1,figsize=(12,7), dpi= 80)
    ax1.plot(x, y1, color=color_line1, lw=3, marker = 'o', ms = 10, label='acc')
    ax1.plot(x, y2, color=color_line2, ls = '--', marker = 'o', ms = 10, label='val_acc')
    

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y3, color='tab:red', lw=3, marker = 'o', ms = 10, label='loss')
    ax2.plot(x, y4, color='lightcoral', ls = '--', marker = 'o', ms = 10, label='val_loss')

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel('epochs', fontsize=20, color = color_text)
    ax1.tick_params(axis='x', rotation=0, labelsize=12, labelcolor=color_text)
    ax1.set_ylabel('Точность (accuracy)', color=color_line1, fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor=color_line1)
    ax1.minorticks_on()
    # ax1.majorticks_on()
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.8)
    # ax1.grid(alpha=.4)
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    plt.text(x_for_text, y_for_text, f'NUM_EXP = {d_num_exp} \n{d_date}\nVAL_ACC:={round(d_evol_val_acc,4)}\ntime_fit:={d_time} сек. ', 
             fontsize = 18, 
             color = color_text)

    # ax2 (right Y axis)
    ax2.set_ylabel("loss-функция", color='tab:red', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_xticks(np.arange(1, len(y1)+1, 1))
    ax2.set_xticklabels(x, rotation=90, fontdict={'fontsize':10})
    ax2.set_title(f'Графики метрик обучения модели по эпохам{d_title}', 
                  fontsize=22, 
                  color = color_line1)
    
    fig.legend(bbox_to_anchor=(0.5, -0.015), loc='lower center', ncol=4)
    fig.tight_layout()
    plt.show()
    return


def hyperp_without_print(d_name_const,
                        d_const_value,
                        d_const_desc,
                        d_df,
                        d_dict):
    d_len = len(d_df)
    if d_name_const not in list(d_df.columns):
        d_df[d_name_const] = None
        d_df[d_name_const] = d_df[d_name_const].astype('object')
        d_df.loc[d_len-1,d_name_const] = str(d_const_value)
        d_dict[d_name_const]= d_const_desc
    else:
        d_df.loc[d_len-1,d_name_const] = str(d_const_value)
    return d_const_value

def to_zip(d_path_to_file_zip, d_list_links_files):
    with zipfile.ZipFile(d_path_to_file_zip, 'w') as d_zip_file:
        for link in d_list_links_files:
            d_zip_file.write(link)
    for link in d_list_links_files:
        if os.path.isfile(link):
            os.remove(link)
        else:
            print(f'Ошибка {link} не найден')
        
    size_in_MB=round(os.path.getsize(d_path_to_file_zip)/(1024*1024),2)
    file_zip = f'r{d_path_to_file_zip}'
    link_ = os.path.basename(d_path_to_file_zip)
    display(f'{link_} успешно создан. {size_in_MB} Mb. Ссылка для скачивания ниже:',FileLink(link_))
    return


def save_model(d_PATH_to_RESULTS,
               d_PATH_to_BEST_MODELS,
               d_model, 
               d_history, 
               d_time_cb, 
               d_results_of_exp, 
               d_descr_hyperp_of_exp,
               d_zip = True):

    if d_history != []:
        R_EPOCH_TIME = hyperp_without_print('R_EPOCH_TIME', d_time_cb.epoch_time, 'Результат обучения. Время обучения каждой эпохи',d_results_of_exp,d_descr_hyperp_of_exp)
        R_BATCH_TIME = hyperp_without_print('R_BATCH_TIME', d_time_cb.batch_time, 'Результат обучения. Время обучения каждого batch',d_results_of_exp,d_descr_hyperp_of_exp)
        R_TRAIN_TIME = hyperp_without_print('R_TRAIN_TIME', d_time_cb.train_time, 'Результат обучения. Время обучения',d_results_of_exp,d_descr_hyperp_of_exp)

    if d_time_cb != []:
        R_LOSS = hyperp_without_print('R_LOSS', d_history.history['loss'], 'Результат обучения. Значения функции потерь по каждой эпохе по тренировочной выборке',d_results_of_exp,d_descr_hyperp_of_exp)
        R_ACC = hyperp_without_print('R_ACC', d_history.history['accuracy'], 'Результат обучения. Значения метрики точности (accuracy) по каждой эпохе по тренировочной выборке',d_results_of_exp,d_descr_hyperp_of_exp)
        R_VAL_LOSS = hyperp_without_print('R_VAL_LOSS', d_history.history['val_loss'], 'Результат обучения. Значения функции потерь по каждой эпохе на валидационной выборке',d_results_of_exp,d_descr_hyperp_of_exp)
        R_VAL_ACC = hyperp_without_print('R_VAL_ACC', d_history.history['val_accuracy'], 'Результат обучения. Значения метрики точности (accuracy) по каждой эпохе на валидационной выборке',d_results_of_exp,d_descr_hyperp_of_exp)
    
    
    d_NUM_EXP = d_results_of_exp['NUM_EXP'].to_numpy()[-1]
    
    time_now = dt.now().strftime('%Y%m%d__%H_%M')
    TIME = hyperp_without_print('TIME', time_now, 'Результат обучения. Время эксперимента',d_results_of_exp,d_descr_hyperp_of_exp)
    
    model_file_name = d_PATH_to_BEST_MODELS+f'best_model_{int(d_NUM_EXP)}_{time_now}.hdf5'
    results_data_file_name = d_PATH_to_RESULTS+f'results_{int(d_NUM_EXP)}_{time_now}.csv'
    descr_hyperp_file_name = d_PATH_to_RESULTS+f'descr_{int(d_NUM_EXP)}_{time_now}.pkl'
    
    d_model.save(model_file_name)
    d_results_of_exp.to_csv(results_data_file_name, index=False)
    with open(descr_hyperp_file_name, 'wb') as f:
        pickle.dump(d_descr_hyperp_of_exp, f, pickle.HIGHEST_PROTOCOL)
    if d_zip == True:
        name_zip_1 = d_PATH_to_RESULTS+f'zip_results_{int(d_NUM_EXP)}_{time_now}.zip'
        name_zip_2 = d_PATH_to_RESULTS+f'zip_model_{int(d_NUM_EXP)}_{time_now}.zip'
        list_1 = [results_data_file_name,descr_hyperp_file_name]
        list_2 = [model_file_name]
        to_zip(name_zip_1,list_1)
        to_zip(name_zip_2,list_2)
    else:
        print(f'Лучшая модель и результаты всех тестов успешно сохранены без архивирования')
    return results_data_file_name


def callbacks_assembler(d_M_CALLBACKS_TYPE, d_M_LR, d_M_LR_UPDATE, d_M_EPOCHS_DROP, d_time_cb):
    
    def scheduler(epoch):
        return d_M_LR * math.pow(d_M_LR_UPDATE, math.floor((1+epoch)/d_M_EPOCHS_DROP))
    
    callback1 = ModelCheckpoint('best_model.hdf5',    
                                monitor='val_accuracy', 
                                verbose=1, 
                                mode='max', 
                                save_best_only=True)
    callback2 = EarlyStopping(monitor='val_accuracy', 
                              patience=4, 
                              restore_best_weights=True,
                              verbose=1)
    callback3 = LearningRateScheduler(scheduler, 
                                      verbose=1)
    
    mix_callbacks = []
    if 'MC' in d_M_CALLBACKS_TYPE:
        mix_callbacks.append(callback1)
        
    if 'ES' in d_M_CALLBACKS_TYPE:
        mix_callbacks.append(callback2)
        
    if 'LRS' in d_M_CALLBACKS_TYPE:
        mix_callbacks.append(callback3)
        
    if 'T' in d_M_CALLBACKS_TYPE:
        mix_callbacks.append(d_time_cb)
        
    
    return mix_callbacks


class TimingCallback(Callback):
    def __init__(self):
        super(TimingCallback, self).__init__()
        self.epoch_logs=[]
        self.batch_logs=[]
        self.train_logs=[]

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_starttime = time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_logs.append(time() - self.epoch_starttime)

    def on_batch_begin(self, batch, logs=None):
        self.batch_starttime = time()

    def on_batch_end(self, batch, logs=None):
        self.batch_logs.append(time() - self.batch_starttime)

    def on_train_begin(self, logs=None):
        self.train_starttime = time()

    def on_train_end(self, logs=None):
        self.train_logs.append(time() - self.train_starttime)

    @property
    def epoch_time(self):
        return np.array(self.epoch_logs)

    @property
    def batch_time(self):
        return np.array(self.batch_logs)

    @property
    def train_time(self):
        return np.array(self.train_logs)


def model_summary_short(d_model,
                        d_base_model,
                        d_title):
    trainable_count = np.sum([K.count_params(w) for w in d_model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in d_model.non_trainable_weights])
    
    count_l_fr_m, count_l_not_fr_m, count_l_bn_m = 0, 0, 0
    len_d_model = len(d_model.layers)
    count_l_m, num_first_fr_l_m = len_d_model, len_d_model
    for i in range(len_d_model, 0, -1):
        layer = d_model.layers[i-1]
        if layer.trainable:
            count_l_not_fr_m += 1
            if not isinstance(layer, BatchNormalization):
                num_first_fr_l_m = count_l_m
        else:
            count_l_fr_m += 1
            if isinstance(layer, BatchNormalization):
                count_l_bn_m +=1
        count_l_m -= 1
    
    if d_base_model != '':
        count_l_fr_bm, count_l_not_fr_bm, count_l_bn_bm = 0, 0, 0
        len_d_base_model = len(d_base_model.layers)
        count_l_bm, num_first_fr_l_bm = len_d_base_model, len_d_base_model
        for i in range(len_d_base_model, 0, -1):
            layer = d_base_model.layers[i-1]
            if layer.trainable:
                count_l_not_fr_bm += 1
                if not isinstance(layer, BatchNormalization):
                    num_first_fr_l_bm = count_l_bm
            else:
                count_l_fr_bm += 1
                if isinstance(layer, BatchNormalization):
                    count_l_bn_bm +=1
            count_l_bm -= 1
    
    
    print(f'Краткая информация о моделе{d_title}:')
    temp_dict = {}
    temp_dict['--------Параметры модели -------'] = '', '--------Params model--------'
    temp_dict['Всего параметров'] = '{:,}'.format(trainable_count + int(non_trainable_count)), 'Total params'
    temp_dict['Тренируемых параметров'] = '{:,}'.format(trainable_count), 'Trainable params'
    temp_dict['Нетренируемых параметров'] = '{:,}'.format(int(non_trainable_count)), 'Non-trainable params'
    
    temp_dict['--------Слои модели--------'] = '', '--------Layers model--------'
    temp_dict['Всего слоев'] = len_d_model, 'Total layers'
    temp_dict['Тренируемых слоев (не заморожен.)'] = count_l_not_fr_m, 'Trainable layers (no frozen)'
    temp_dict['Нетренируемых слоев (заморожен.)'] = count_l_fr_m, 'Non-trainable layers (frozen)'
    temp_dict['.. среди них слоев bn'] = count_l_bn_m, '.. among them layers bn'
    
    if num_first_fr_l_m == len_d_model:
        d_temp = 'Отсутствует'
    else:
        d_temp = num_first_fr_l_m
    
    temp_dict['Номер первого тренируемого слоя'] = d_temp, 'Count num first trainable layer'
    
    if d_base_model != '':
        temp_dict['--------Слои базовой модели--------'] = '', '--------Layers base model--------'
        temp_dict['Всего слоев бм'] = len_d_base_model, 'Total layers'
        temp_dict['Тренируемых слоев бм (не заморожен.)'] = count_l_not_fr_bm, 'Trainable layers (no frozen)'
        temp_dict['Нетренируемых слоев бм (заморожен.)'] = count_l_fr_bm, 'Non-trainable layers (frozen)'
        temp_dict['... среди них слоев bn'] = count_l_bn_bm, '.. among them layers bn'
        
        if num_first_fr_l_bm == len_d_base_model:
            d_temp = 100
        else:
            d_temp = (num_first_fr_l_m-1)/len_d_base_model
            d_temp = int(round(d_temp,2)*100)
        temp_dict['% заморозки базовой модели'] = f'{d_temp} %', '% freeze base model'
        
        temp_dict['----Архитектура головы модели----'] = '', '----Head architecture----'
        temp_dict['Всего слоев головы'] = len_d_model - len_d_base_model, 'Total layers'
        temp_dict['Полносвязных слоев без выходного слоя'] = 1, 'Dense layers without output layer'
        temp_dict['BatchNormalization'] = True, 'BatchNormalization'
        temp_dict['Функция активации скрытого слоя'] = 'relu', 'Activation func for hidden layer'
        temp_dict['Функция активации выходного слоя'] = 'softmax', 'Activation func for output layer'

        

    temp_df2 = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['Значение', ''])
    display(temp_df2)
    return


def model_assembler(d_base_model, d_head):
    d_outputs = d_base_model.output
    
    for l in d_head.layers:
        d_outputs = l(d_outputs)

    return Model(inputs=d_base_model.input, 
                 outputs=d_outputs)

def train_test_datagen(d_rotation_range,
                    d_brightness_range,
                    d_width_shift_range,
                    d_height_shift_range,
                    d_horizontal_flip,
                    d_rescale,
                    d_validation_split):
    train_datagen = ImageDataGenerator(rotation_range=d_rotation_range,
                                       brightness_range=d_brightness_range,
                                       width_shift_range=d_width_shift_range,
                                       height_shift_range=d_height_shift_range,
                                       horizontal_flip=d_horizontal_flip,
                                       rescale = d_rescale,
                                       validation_split=d_validation_split)

    test_datagen = ImageDataGenerator(rescale = d_rescale)
    return train_datagen, test_datagen


# Обертка для генераторов данных
def train_valid_test_generators(
                    d_rotation_range,
                    d_brightness_range,
                    d_width_shift_range,
                    d_height_shift_range,
                    d_horizontal_flip,
                    d_rescale,
                    d_validation_split,
                    d_path_to_train,
                    d_path_to_test,
                    d_df_submision,
                    d_IMG_SIZE,
                    d_BATCH_SIZE,
                    d_RANDOM_SEED): 
    # создаем объекты с аугментацией
    train_datagen, test_datagen = train_test_datagen(
                    d_rotation_range,
                    d_brightness_range,
                    d_width_shift_range,
                    d_height_shift_range,
                    d_horizontal_flip,
                    d_rescale,
                    d_validation_split)

    # генератор для тренировочной выборки
    train_generator = train_datagen.flow_from_directory(
        d_path_to_train,
        target_size=(d_IMG_SIZE, d_IMG_SIZE),
        batch_size=d_BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=d_RANDOM_SEED,
        subset='training'
    )

    # генератор для валидационной выборки
    validation_generator = train_datagen.flow_from_directory(
        d_path_to_train,
        target_size=(d_IMG_SIZE, d_IMG_SIZE),
        batch_size=d_BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=d_RANDOM_SEED,
        subset='validation'
    )

    # генератор для тестовых данных
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=d_df_submision,
        directory=d_path_to_test,
        x_col="Id",
        y_col=None,
        target_size=(d_IMG_SIZE, d_IMG_SIZE),
        batch_size=d_BATCH_SIZE,
        class_mode=None,
        shuffle=False,
        seed=d_RANDOM_SEED
    )
    return train_generator, validation_generator, test_generator


def hyperp (d_name_const,
            d_const_value,
            d_const_desc,
            d_df,
            d_dict):
    d_len = len(d_df)
    if d_name_const not in list(d_df.columns):
        d_df[d_name_const] = None
        d_df[d_name_const] = d_df[d_name_const].astype('object')
        if d_name_const == 'NUM_EXP':
            d_df.loc[d_len,d_name_const] = d_const_value
        else:
            d_df.loc[d_len-1,d_name_const] = str(d_const_value)
        d_dict[d_name_const]= d_const_desc
        print(f'{d_name_const} = {d_const_value} ({d_dict[d_name_const]}). Константа инициализирована.' )
    else:
        if d_name_const == 'NUM_EXP':
            d_df.loc[d_len,d_name_const] = d_const_value
        else:
            d_df.loc[d_len-1,d_name_const] = str(d_const_value)
        print(f'{d_name_const} = {d_const_value} ...')
    return d_const_value


def images_from_dataset_with_path(d_subtitle ,
                                    d_title ,
                                    d_path,
                                    d_df,
                                    d_name_column,
                                    d_list,
                                    d_name_column_with_path,
                                    d_rs,
                                    b_title=True):
    np.random.seed(d_rs)

    plt.style.use('seaborn-paper')
    color_text = plt.get_cmap('PuBu')(0.95)

    plt.figure(figsize=(14,5))

    for num, i in enumerate(d_list):
        random_image = d_df[d_df[d_name_column]==i].sample(1)
        random_image_path = random_image.iloc[0][d_name_column_with_path]
        
        im = PIL.Image.open(d_path+f'{i}/{random_image_path}')
        plt.subplot(2,5, num+1)
        plt.imshow(im)
        if b_title:
            plt.title(d_title+str(i), fontsize=14, color = color_text)
        plt.text(im.size[0]//2-80,im.size[1]+30, im.size, fontsize=8, color = color_text)
        plt.axis('off')
    plt.suptitle(d_subtitle, y = 0.98, fontsize = 18, color = color_text)
    plt.show()
    return


def simple_plot_barv(d_name_title,
                    d_name_column,
                    d_df,
                    d_my_font_scale,
                    d_name_axis_x,
                    d_name_axis_y ):
    """
    
    """
    list_values = list(d_df[d_name_column].unique())

    temp_df = d_df[d_name_column].value_counts()
    d_mean = temp_df.values.mean()

    plt.style.use('seaborn-paper')
    sns.set(font_scale=d_my_font_scale)
    color_text = plt.get_cmap('PuBu')(0.95)
    color_bar = plt.get_cmap('PuBu')(0.8)
    _, ax = plt.subplots(figsize=(12, 5))

    category_colors = plt.get_cmap('PuBu')(
            np.linspace(0.85, 0.35, len(list_values)))

    widths = temp_df.values
    ax.bar(temp_df.index, height = widths,  width=0.7, color=category_colors)  
    for (x, y) in zip(temp_df.index, temp_df.values):
        ax.text(x,d_mean, str(int((y-d_mean)/d_mean*100))+'%', fontsize=12, weight = 'bold', ha='center', va='center',  color=color_text)
    ax.set_title(d_name_title+' (критерий '+d_name_column+')', loc ='center', fontsize=12, color = category_colors[0])
    ax.set_xlabel(d_name_axis_x, fontsize=15, color = color_text)
    ax.set_ylabel(d_name_axis_y, fontsize=15, color = color_text)
    

    plt.show()
    return


def unzip(d_path_in, d_path_out, d_list_names_zips, d_comment):
    print('Распаковываем zip-архивы ',d_comment,':', sep='')
    i = 1
    max_len = 0
    for elem in d_list_names_zips:
        d_elem = len(elem)
        if d_elem > max_len:
            max_len = d_elem
    if max_len > 0:
        for data_zip in d_list_names_zips:
            print(i, '. ', data_zip, '.'*(20-len(data_zip)+max_len), sep='', end='')
            with zipfile.ZipFile(d_path_in+data_zip,"r") as z:
                z.extractall(d_path_out)
            print('. Распакован.')
            i += 1
        print('===')
        print(f'Текущее состояние папки ({d_path_out}):= ', end='')
        print(os.listdir(d_path_out))
    else:
        print('Распаковка прервана. Причина - Список zip-архивов пустой.')
    return

def mkdir(d_path,d_name_dir):
    path_dir = d_path + d_name_dir + '/'
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        print(f'Директория с именем ({d_name_dir}) успешно создана в ({d_path}).')
        print(f'Путь к новой папке ({path_dir}) возвращен в переменную.')
    else:
        print('Директория с таким именем уже существует.') 
        print(f'Путь к папке ({path_dir}) возвращен в переменную.')
    return path_dir


def result_EDA_feature(d_feature, d_train, d_test, d_n_cols, d_EDA_done_cols, d_old_len_train):
    # записываем признак в список проанализированных признаков
    d_EDA_done_cols.append(d_feature)
    d_len_ = len(d_EDA_done_cols)
    print(f'В результате после EDA признака:= {d_feature}, обработано признаков:= {d_len_}, осталось:= {d_n_cols-d_len_}')
    # смотрим как ведет себя трейн
    temp = len(d_train)
    print('Кол-во строк в трейне:= ', temp, '. Убрали на данном шаге:= ', d_old_len_train-temp)
    d_old_len_train = temp
    # проверяем что мы случайно не испортили тест
    print('Кол-во строк в тесте:= ', len(d_test))
    return d_old_len_train, d_EDA_done_cols


def nunique_not_found(d_df1, d_df2, d_col):

    temp_set = set()
    temp_set2 = set()
    temp_set = set(d_df1[d_col].unique())
    temp_set2 = set(d_df2[d_col].unique())

    print(f'в столбце:= {d_col} в трейне НЕ НАЙДЕНО:= {len(temp_set2-temp_set)} уникальных значений из теста')

    return list(temp_set2-temp_set)

def check_df_before_merg(d_df1,d_df2):
    
    list_of_names1 = list(d_df1.columns)
    temp_dict = {}
    temp_dict['# уник_1'] = d_df1.nunique().values
    temp_dict['в первой строке_1'] =d_df1.loc[0].values
    temp_dict['тип_1'] = d_df1.dtypes
    temp_dict['имя признака_1'] = list_of_names1
    temp_df1 = pd.DataFrame.from_dict(temp_dict)
    
    
    list_of_names2 = list(d_df2.columns)
    temp_dict2 = {}
    temp_dict2['имя признака_2'] = list_of_names2
    temp_dict2['тип_2'] = d_df2.dtypes
    temp_dict2['в первой строке_2'] =d_df2.loc[0].values
    temp_dict2['# уник_2'] = d_df2.nunique().values
    temp_df2 = pd.DataFrame.from_dict(temp_dict2)
    
    temp_df = pd.concat([temp_df1,temp_df2], axis=1, sort=False)
    temp_df.reset_index(inplace = True)
    del temp_df['index']
    display(temp_df)

    temp_dict3 = {}
    temp_df3= pd.DataFrame(temp_df)
    temp_list  = []
    temp_list2  = []
    temp_list3  = []
    temp_list4  = []
    temp_list5  = []

    for i in range(len(temp_df)):
        if str(temp_df3['тип_2'][i]) != str(temp_df3['тип_1'][i]):
            temp_list.append(temp_df3['имя признака_1'][i])
            temp_list2.append(temp_df3['имя признака_2'][i])
            temp_list3.append(str(temp_df3['тип_1'][i]) + '!=' + str(temp_df3['тип_2'][i]))
            temp_list4.append(i)
        if temp_df3['# уник_2'][i]>0 and temp_df3['# уник_1'][i]/temp_df3['# уник_2'][i] > 2:
            temp_list5.append(i)
            
    temp_dict3['index']= temp_list4
    temp_dict3['имя признака_1']= temp_list
    temp_dict3['не совпадают типы'] = temp_list3
    temp_dict3['имя признака_2']= temp_list2

    temp_df4 = pd.DataFrame.from_dict(temp_dict3)
    temp_df4.set_index('index',inplace=True)

    print(f'Резюме:\n 1. Не совпали типы в:= {len(temp_df4)} столбцах\n')
    print(f'2. Уникальные значения заоблачно различаются в:= {len(temp_list5)} столбцах {temp_list5}')
    display(temp_df4)



    return


def hbar_group_pivot_table(d_bodyType, 
                        d_group_col, 
                        d_df, 
                        d_year_start, 
                        d_year_end,
                        d_my_font_scale):
    temp_df = d_df.copy()
    temp_df2 = temp_df[(temp_df['bodyType']==d_bodyType) & (temp_df['modelDate']>=d_year_start) & (temp_df['modelDate']<=d_year_end)]
    

    temp_pt_mean = pd.pivot_table(temp_df2, values =d_group_col, index =['bodyType','brand'], columns =['modelDate'],aggfunc = np.mean, margins=True)
    temp_list = list(temp_pt_mean['All'][d_bodyType].index)

    temp_pt_std = pd.pivot_table(temp_df2, values =d_group_col, index =['bodyType','brand'], columns =['modelDate'],aggfunc = np.std, margins=True)
    temp_std = temp_pt_std['All'][d_bodyType]['BMW']
    temp_mean = temp_pt_mean['All'][d_bodyType]['BMW']
    a = temp_pt_mean['All'][d_bodyType]['BMW']-temp_std/2
    b = temp_pt_mean['All'][d_bodyType]['BMW']+temp_std/2

    temp_list2 = list(temp_pt_std['All'][d_bodyType].index)

    temp_list_std =[]
    list_overlapp_brands =[]
    for brand in temp_list:
        if brand in temp_list2:
            std_ = temp_pt_std['All'][d_bodyType][brand]/2
        else:
            std_ = 0
        temp_list_std.append(std_)
        c = temp_pt_mean['All'][d_bodyType][brand] - std_
        d = temp_pt_mean['All'][d_bodyType][brand] + std_
        if brand != 'BMW' and ((b>=c and d>=a) or (a<=c and d<=b)):
            list_overlapp_brands.append(brand)
    
    temp_std = temp_pt_std['All'][d_bodyType]['BMW']
    temp_mean = temp_pt_mean['All'][d_bodyType]['BMW']

    plt.style.use('seaborn-paper')
    sns.set(font_scale=d_my_font_scale)
    color_text = plt.get_cmap('PuBu')(0.85)
    color_bar = plt.get_cmap('PuBu')(0.8)

    plt.figure(figsize=(12, 6))
    
    plt.barh(temp_list, width=temp_pt_mean['All'][d_bodyType].values+temp_list_std, color =color_bar)
    plt.barh(temp_list, width=temp_pt_mean['All'][d_bodyType].values, color ='red')
    plt.barh(temp_list, width=temp_pt_mean['All'][d_bodyType].values-temp_list_std, color =color_bar)


    plt.plot([temp_mean,temp_mean], [-1, len(temp_list)+1], color= 'red', label='среднее значение BMW', marker='.', lw=2, ls = '--')
    plt.plot([temp_mean-temp_std/2,temp_mean-temp_std/2], [-1, len(temp_list)+1], color='grey', label='отклонение вниз на std/2', marker='.', lw=3)
    plt.plot([temp_mean+temp_std/2,temp_mean+temp_std/2], [-1, len(temp_list)+1], color='blue', label='отклонение вверх на std/2', marker='.', lw=3)

    plt.xlabel(d_group_col, fontsize=15, color = color_text)
    plt.ylabel('brand', fontsize=15, color = color_text)
    plt.title(f'Среднее и отклонение {d_group_col} сводной таблицы сгруппированной по {d_bodyType}. modelDate с {d_year_start} по {d_year_end}', color = color_text, fontsize=15)
    plt.legend(loc="lower right", fontsize=11)
    # y_min_text = y_min +0.5*max(std_metric_train,std_metric_test)
    plt.text(100, len(temp_list)-0.5, f'кол-во брендов авто попавших в сводную таблицу = {len(temp_list)} из 36 \nкол-во брендов авто попадающих в область значений BMW = {len(list_overlapp_brands)} из 36', fontsize = 14)
    plt.show()
    print('Список релевантных брендов: ',*list_overlapp_brands)
    return list_overlapp_brands


def vis_cross_val_score(d_name_metric, d_vec, d_value_metric, d_my_font_scale):
    num_folds = len(d_vec['train_score'])
    avg_metric_train, std_metric_train = d_vec['train_score'].mean(), d_vec['train_score'].std()
    avg_metric_test, std_metric_test = d_vec['test_score'].mean(), d_vec['test_score'].std()

    plt.style.use('seaborn-paper')
    sns.set(font_scale=d_my_font_scale)
    color_text = plt.get_cmap('PuBu')(0.85)

    plt.figure(figsize=(12, 6))
    plt.plot(d_vec['train_score'], label='тренировочные значения', marker='.', color= 'darkblue')
    plt.plot([0,num_folds-1], [avg_metric_train, avg_metric_train], color='blue', label='среднее трен. значений ', marker='.', lw=2, ls = '--')

    plt.plot(d_vec['test_score'], label='тестовые значения', marker='.', color= 'red')
    plt.plot([0,num_folds-1], [avg_metric_test, avg_metric_test], color='lightcoral', label='среднее тест. значений ', marker='.', lw=2, ls = '--')

    plt.plot([0,num_folds-1], [d_value_metric, d_value_metric], color='grey', label='значение метрики до CV', marker='.', lw=3)

    # plt.xlim([1, num_folds])
    y_max = max(avg_metric_train,avg_metric_test) + 1.5*max(std_metric_train,std_metric_test)
    y_min = min(avg_metric_train,avg_metric_test) - 3*max(std_metric_train,std_metric_test)
    plt.ylim([y_min, y_max])
    plt.xlabel('номер фолда', fontsize=15, color = color_text)
    plt.ylabel(d_name_metric, fontsize=15, color = color_text)
    plt.title(f'Кросс-валидация по метрике {d_name_metric} на {num_folds} фолдах', color = color_text, fontsize=17)
    plt.legend(loc="lower right", fontsize=11)
    y_min_text = y_min +0.5*max(std_metric_train,std_metric_test)
    plt.text(0, y_min_text, f'{d_name_metric} на трейне = {round(avg_metric_train,3)} +/- {round(std_metric_train,3)} \n{d_name_metric} на тесте    = {round(avg_metric_test,3)} +/- {round(std_metric_test,3)} \n{d_name_metric} до CV        = {round(d_value_metric,3)}', fontsize = 15)
    plt.show()
    return


def model_coef(d_columns, d_model_coef_0):

    temp_dict = {}
    temp_dict['имя признака'] = d_columns
    temp_dict['коэффициент модели'] = d_model_coef_0
    temp_dict['модуль коэф'] = abs(temp_dict['коэффициент модели'])
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='columns')
    temp_df = temp_df.sort_values(by='модуль коэф', ascending=False)
    temp_df.reset_index(drop=True,inplace=True)
    
    return temp_df.loc[:,['имя признака','коэффициент модели']]

def GridSearchCV_for_LogReg(d_X_train, d_y_train, d_values_for_C):
    # Добавим типы регуляризации
    penalty = ['l1', 'l2']

    # Зададим ограничения для параметра регуляризации
    C = np.array(d_values_for_C)

    # Создадим гиперпараметры
    hyperparameters = dict(C=C, penalty=penalty)

    model = LogisticRegression(multi_class = 'ovr', class_weight='balanced')
    model.fit(d_X_train, d_y_train)

    # Создаем сетку поиска с использованием 5-кратной перекрестной проверки
    clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0, scoring='f1')

    best_model = clf.fit(d_X_train, d_y_train)

    # View best hyperparameters
    temp_dict = {}
    temp_dict['Penalty'] = [best_model.best_estimator_.get_params()['penalty']]
    temp_dict['C'] = [best_model.best_estimator_.get_params()['C']]

    # temp_dict['Признак'] = [best_model.best_index_]
    # temp_list = sorted(clf.cv_results_.keys())
    # temp_dict['Кол-во'] = [len(temp_list)]
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['Лучшие'])
    display(temp_df)

    temp_dict = {}
    # temp_dict['Лучшие признаки'] = temp_list
    # temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
    # display(temp_df.T)


    return



def where_1_in_corr(d_df, d_y):

    result = []
    drop_list_columns = []
    all_cols = list(d_df.columns)
    for col in all_cols:
        temp_list = d_df.index[d_df[col] == 1].tolist()
        list1 = [x for x in temp_list if x not in [col]]
        if list1 != []:
            list1.append(col)
            drop_list_columns.append(list1)
    for i in range(len(drop_list_columns)//2):
        result.append(drop_list_columns[i][0])
    result= [x for x in result if x not in [d_y]]
    return result


def PR_curve_with_area(d_y_true, d_y_pred_prob, d_my_font_scale):
    

    plt.style.use('seaborn-paper')
    sns.set(font_scale=d_my_font_scale)
    # sns.set_color_codes("muted")

    plt.figure(figsize=(8, 6))
    precision, recall, thresholds = precision_recall_curve(d_y_true, d_y_pred_prob, pos_label=1)
    prc_auc_score_f = auc(recall, precision)
    plt.plot(precision, recall, lw=3, label='площадь под PR кривой = %0.3f)' % prc_auc_score_f)
    
    plt.xlim([-.05, 1.0])
    plt.ylim([-.05, 1.05])
    plt.xlabel('Точность \n Precision = TP/(TP+FP)')
    plt.ylabel('Полнота \n Recall = TP/P')
    plt.title('Precision-Recall кривая')
    plt.legend(loc="upper right")
    plt.show()
    return


def ROC_curve_with_area(d_y_true, d_y_pred_prob, d_my_font_scale):
    roc_auc_score_f = roc_auc_score(d_y_true, d_y_pred_prob)

    plt.style.use('seaborn-paper')
    sns.set(font_scale=d_my_font_scale)
    # sns.set_color_codes("muted")

    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(d_y_true, d_y_pred_prob, pos_label=1)

    plt.plot(fpr, tpr, lw=3, label='площадь под ROC кривой = %0.3f)' % roc_auc_score_f)
    plt.plot([0, 1], [0, 1], color='grey')
    plt.xlim([-.05, 1.0])
    plt.ylim([-.05, 1.05])
    plt.xlabel('Ложно классифицированные \n False Positive Rate (FPR)')
    plt.ylabel('Верно классифицированные \n True Positive Rate (TPR)')
    plt.title('ROC кривая')
    plt.legend(loc="lower right")
    plt.show()
    return

def test_last_pred(d_y_true, d_y_pred, d_y_pred_prob):
    last_pred[0], last_pred[1], last_pred[2] = d_y_true, d_y_pred, d_y_pred_prob
    return



def all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(d_y_true, d_y_pred):
    def r_(d_x):
        '''
        short code for def round
        '''
        return round(d_x,6)


    def r_p(d_x):
        '''
        short code for def round procent
        '''
        return round(d_x,4)


    def MAE(y_true, y_pred):
        '''
        mean absolute error (средняя абсолютная ошибка)
        '''
        return np.mean(np.abs(y_true - y_pred))


    def MPE(y_true, y_pred):
        '''
        mean percentage error (средняя процентная ошибка)
        '''
        return np.mean((y_true - y_pred) / y_true) * 100

    
    def MAPE(y_true, y_pred):
        '''
        mean absolute percentage error (средняя абсолютная процентная ошибка)
        '''
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    
    def SMAPE(y_true, y_pred):
        '''
        symmetric MAPE (симетричная средняя абсолютная процентная ошибка)
        '''
        return np.mean(2*np.abs(y_true - y_pred) / (np.abs(y_true)+np.abs(y_pred))) * 100
    
    def WMAPE(y_true, y_pred):
        '''
        weighted absolute percent error (взвешенная абсолютная процентная ошибка)
        WMAPE / MAD-MEAN RATIO / WAPE — WEIGHTED ABSOLUTE PERCENT ERROR
        '''
        return np.mean(np.abs(y_true - y_pred)) / np.mean(y_true) * 100
    

    def RMSE(y_true, y_pred):
        '''
        root mean squared error (корень из среднеквадратичной ошибки)
        '''
        return mean_squared_error(y_true, y_pred)**0.5

    
    d_y_true_last, d_y_pred_last, d_y_pred_prob_last =  last_pred[0], last_pred[1], last_pred[2]
    temp_dict = {}
    temp1 = r_(MAE(d_y_true, d_y_pred))
    temp2 = r_(MAE(d_y_true_last, d_y_pred_last))
    temp_dict['MAE'] = [temp1, temp1-temp2,'mean absolute error (средняя абсолютная ошибка)']

    temp1 = r_p(MPE(d_y_true, d_y_pred))
    temp2 = r_p(MPE(d_y_true_last, d_y_pred_last))
    temp_dict['MPE'] = [temp1, temp1-temp2,'(%) mean percentage error (средняя процентная ошибка)']
    
    temp1 = r_p(MAPE(d_y_true, d_y_pred))
    temp2 = r_p(MAPE(d_y_true_last, d_y_pred_last))
    temp_dict['MAPE'] = [temp1, temp1-temp2,'(%) mean absolute percentage error (средняя абсолютная процентная ошибка)']
    
    temp1 = r_p(SMAPE(d_y_true, d_y_pred))
    temp2 = r_p(SMAPE(d_y_true_last, d_y_pred_last))
    temp_dict['SMAPE'] = [temp1, temp1-temp2,'(%) symmetric MAPE (симетричная средняя абсолютная процентная ошибка)']    
    
    temp1 = r_p(WMAPE(d_y_true, d_y_pred))
    temp2 = r_p(WMAPE(d_y_true_last, d_y_pred_last))
    temp_dict['WMAPE'] = [temp1, temp1-temp2,'(%) weighted absolute percent error (взвешенная абсолютная процентная ошибка)']
    
    temp1 = r_(mean_squared_error(d_y_true, d_y_pred))
    temp2 = r_(mean_squared_error(d_y_true_last, d_y_pred_last))
    temp_dict['MSE'] = [temp1, temp1-temp2,'mean squared error (среднеквадратичная ошибка)']
    
    temp1 = r_(RMSE(d_y_true, d_y_pred))
    temp2 = r_(RMSE(d_y_true_last, d_y_pred_last))
    temp_dict['RMSE'] = [temp1, temp1-temp2,'root mean squared error (корень из среднеквадратичной ошибки)']    
    
    temp1 = r_(r2_score(d_y_true, d_y_pred))
    temp2 = r_(r2_score(d_y_true_last, d_y_pred_last))
    temp_dict['R2'] = [temp1, temp1-temp2,'coefficient of determination (коэффициент детерминации)']    
    
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['Значение','Дельта с предыдущим','Описание'])
    display(temp_df)

    last_pred[0], last_pred[1] = d_y_true, d_y_pred

    return


def all_metrics(d_y_true, d_y_pred, d_y_pred_prob):
        
    d_y_true_last, d_y_pred_last, d_y_pred_prob_last =  last_pred[0], last_pred[1], last_pred[2]
    temp_dict = {}
    temp1 = accuracy_score(d_y_true, d_y_pred)
    temp2 = accuracy_score(d_y_true_last, d_y_pred_last)
    temp_dict['accuracy'] = [temp1, temp2-temp1,'(TP+TN)/(P+N)']

    temp1 = balanced_accuracy_score(d_y_true, d_y_pred)
    temp2 = balanced_accuracy_score(d_y_true_last, d_y_pred_last)
    temp_dict['balanced accuracy'] = [temp1, temp2-temp1,'сбалансированная accuracy']
    
    temp1 = precision_score(d_y_true, d_y_pred)
    temp2 = precision_score(d_y_true_last, d_y_pred_last)
    temp_dict['precision'] = [temp1, temp2-temp1,'точность = TP/(TP+FP)']
    
    temp1 = recall_score(d_y_true, d_y_pred)
    temp2 = recall_score(d_y_true_last, d_y_pred_last)
    temp_dict['recall'] = [temp1, temp2-temp1,'полнота = TP/P']
    
    temp1 = f1_score(d_y_true, d_y_pred)
    temp2 = f1_score(d_y_true_last, d_y_pred_last)
    temp_dict['f1_score'] = [temp1, temp2-temp1,'среднее гармоническое точности и полноты']
    
    temp1 = roc_auc_score(d_y_true, d_y_pred_prob)
    temp2 = roc_auc_score(d_y_true_last, d_y_pred_prob_last)
    temp_dict['roc_auc'] = [temp1, temp2-temp1,'Area Under Curve - Receiver Operating Characteristic']    
    
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['Значение','Дельта с предыдущим','Описание'])
    display(temp_df)

    last_pred[0], last_pred[1], last_pred[2] = d_y_true, d_y_pred, d_y_pred_prob

    return

def plot_confusion_matrix(y_true, y_pred, d_my_font_scale, classes,
                          normalize=False,
                          title=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    list_of_labels = [['TP','FP'],['FN','TN']]
    
    if not title:
        if normalize:
            title = 'Нормализованная матрица ошибок'
        else:
            title = 'Матрица ошибок без нормализации'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    cm[0,0], cm[1,1] = cm[1,1], cm[0,0]

    # # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   

    plt.style.use('seaborn-paper')
    cmap=plt.cm.Blues
    color_text = plt.get_cmap('PuBu')(0.85)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.grid(False)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           
           title=title)
    ax.title.set_fontsize(15)
    ax.set_ylabel('Предсказанные значения', fontsize=14, color = color_text)
    ax.set_xlabel('Целевая переменная', fontsize=14, color = color_text)
    ax.set_xticklabels(classes, fontsize=12, color = 'black')
    ax.set_yticklabels(classes, fontsize=12, color = 'black')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, list_of_labels[i][j]+'\n'+format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def confusion_matrix_f(d_name_columns, d_y, d_y_pred, d_my_font_scale, normalize=False):

    class_names  = np.array(d_name_columns, dtype = 'U10')
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(d_y, d_y_pred, d_my_font_scale, classes=class_names,
                        title='Матрица ошибок без нормализации')

    # Plot normalized confusion matrix
    if normalize:
        plot_confusion_matrix(d_y, d_y_pred, d_my_font_scale, classes=class_names, normalize=True,
                        title='Нормализованная матрица ошибок')

    plt.show()
    return

# функция для стандартизации
def StandardScaler_column(d_df, d_col):
    scaler = StandardScaler()
    scaler.fit(d_df[[d_col]])
    return scaler.transform(d_df[[d_col]])


def StandardScaler_df_and_filna_0(d_df, d_columns):
    # стандартизируем все столбцы кроме целевой и Sample
    for i  in list(d_df[d_columns].columns):
        d_df[i] = StandardScaler_column(d_df, i)
        if len(d_df[d_df[i].isna()]) < len(d_df):
            d_df[i] = d_df[i].fillna(d_df[i].min())
    return

def get_dummies_df(d_df, d_columns):
    star_list_columns = list(d_df.columns)
    # реализуем метод OneHotLabels через get_dummies
    d_df = pd.get_dummies(d_df, columns=d_columns, drop_first=True)
    # мы специально не удаляем первоначальные столбы, чтобы потом можно было провести построчную проверку перед стандартизацией и моделированием 
    end_list_columns = list(d_df.columns)
    new_dumm_cat_cols = [x for x in end_list_columns if x  not in star_list_columns]

    temp_dict = {}
    temp_dict['имя НОВОГО добавленного признака'] = new_dumm_cat_cols
    temp_dict['тип признака'] = d_df[new_dumm_cat_cols].dtypes
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
    display(temp_df.T)
    return new_dumm_cat_cols

def scatterplot_with_hist(d_name_column_x, d_name_column_y, d_df):
    temp_df = d_df
    # Create Fig and gridspec
    fig = plt.figure(figsize=(12, 8), dpi= 80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    ax_main.scatter(temp_df[d_name_column_x], temp_df[d_name_column_y], s=1, c=temp_df[d_name_column_y].astype('category').cat.codes)  #, , alpha=.9, data=df, cmap="tab10", edgecolors='gray', linewidths=.5)

    # histogram on the right
    ax_bottom.hist(temp_df[d_name_column_x], 40, histtype='stepfilled', orientation='vertical', color='blue')
    ax_bottom.invert_yaxis()

    # histogram in the bottom
    ax_right.hist(temp_df[d_name_column_y], 40, histtype='stepfilled', orientation='horizontal', color='blue')

    # Decorations
    ax_main.set(title='Scatterplot with Histograms \n '+d_name_column_x+'vs'+ d_name_column_y, xlabel=d_name_column_x, ylabel=d_name_column_y)
    ax_main.title.set_fontsize(20)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(14)

    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    plt.show()
    return


def four_plot_with_log(d_name_plot,d_df):
    
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (12, 3)

    _, axs = plt.subplots(1, 4)
    temp_df = d_df
    axs[0].hist(temp_df,bins=11)
    axs[0].set_title(d_name_plot)
    axs[1].boxplot(temp_df)
    axs[1].set_title('')
    temp_df = d_df.apply(lambda x: math.log(x+1))
    axs[2].hist(temp_df,bins=11)
    axs[2].set_title('log')
    axs[3].boxplot(temp_df)
    axs[3].set_title('')
    return

def four_plot_with_log2(d_name_column,d_df):
    
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (12, 3)
    color_text = plt.get_cmap('PuBu')(0.85)

    
    temp_df=d_df.copy()

    fig = plt.figure()

   
    ax_1 = fig.add_subplot(1, 4, 1)
    ax_2 = fig.add_subplot(1, 4, 2)
    ax_3 = fig.add_subplot(1, 4, 3)
    ax_4 = fig.add_subplot(1, 4, 4)

    plt.suptitle(f'Гистограммы и box-plot для признака \'{d_name_column}\' и log({d_name_column})', fontsize=14, color = color_text, y=-0.02)

    ax_1.hist(temp_df[d_name_column],bins=11)
    ax_1.set_title(f'\'{d_name_column}\'', loc = 'right', fontsize=10, color = color_text)
    ax_2.boxplot(temp_df[d_name_column])
    
    temp_name = 'log_'+d_name_column
    temp_df.loc[:,temp_name] =  temp_df[d_name_column].apply(lambda x: math.log(x+1))

    ax_3.hist(temp_df[temp_name],bins=11)
    ax_3.set_title(f'log({d_name_column})', loc = 'right', fontsize=10, color = color_text)
    ax_4.boxplot(temp_df[temp_name])
    
    plt.show()
    return


def big_hist(d_name_column,d_df):
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (12, 3)

    temp_df = d_df
    temp_df[d_name_column].hist(bins=50)

    return


def big_hist_log(d_name_column,d_df):
    
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (12, 3)

    temp_df = d_df.copy()
    temp_df['log_'+d_name_column] = temp_df[d_name_column].apply(lambda x: math.log(x+1))

    temp_df['log_'+d_name_column].hist(bins=50)

    return


def borders_of_outliers(d_name_column,d_df, log = False):
    
    if log:
        temp_df = d_df[d_name_column].apply(lambda x: math.log(x+1))
    else:
        temp_df = d_df[d_name_column]
    IQR = temp_df.quantile(0.75) - temp_df.quantile(0.25)
    perc25 = temp_df.quantile(0.25)
    perc75 = temp_df.quantile(0.75)
    left_border = perc25 - 1.5*IQR
    right_border = perc75 + 1.5*IQR

    temp_dict = {}
    if log:
        temp_dict['границы выбросов с логарифмом'] = [left_border, right_border]
        temp_dict['границы выбросов без логарифма'] = [math.exp(left_border)-1, math.exp(right_border)-1]
        count_values_left = (temp_df<left_border).sum()
        count_values_right = (temp_df>right_border).sum()
        temp_dict['кол-во значений за границей'] = [count_values_left, count_values_right]

    else:
        temp_dict['границы выбросов'] = [left_border, right_border]
        count_values_left = (temp_df<left_border).sum()
        count_values_right = (temp_df>right_border).sum()
        temp_dict['кол-во значений за границей'] = [count_values_left, count_values_right]
        
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['левая','правая'])
    display(temp_df)
    return 


def describe_with_hist(d_name_plot,d_df):
    temp_describe = d_df.describe()
    temp_dict = {}
    temp_dict['кол-во строк'] = len(d_df)
    temp_dict['тип значений'] = d_df.dtype
    temp_dict['кол-во значений'] = temp_describe[0]
    temp_dict['кол-во NaN'] = (d_df.isna()).sum()
    temp_dict['среднее'] = temp_describe[1]
    temp_dict['медиана'] = temp_describe[5]
    temp_dict['мин'] = temp_describe[3]
    temp_dict['макс'] = temp_describe[7]

    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=[d_name_plot])
    display(temp_df)

    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (4, 3)
    color_text = plt.get_cmap('PuBu')(0.85)
    
    plt.title(f'Гистограмма признака \'{d_name_plot}\' ', fontsize=12, color = color_text)
    n_bins = d_df.nunique()
    if n_bins >15:
        n_bins = 15
    d_df.hist(bins=n_bins)
    return


def describe_without_plots(d_name_plot,d_df):
    temp_describe = d_df.describe().copy()
    temp_dict = {}
    temp_dict['кол-во строк'] = len(d_df)
    temp_dict['тип значений'] = d_df.dtype
    temp_dict['кол-во значений'] = temp_describe[0]
    temp_dict['кол-во NaN'] = (d_df.isna()).sum()
    temp_dict['среднее'] = temp_describe[1]
    temp_dict['медиана'] = temp_describe[5]
    temp_dict['мин'] = temp_describe[3]
    temp_dict['макс'] = temp_describe[7]

    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=[d_name_plot])
    display(temp_df)

    return


def describe_without_plots_all_collumns(d_df, full=True, short=False):
    list_of_names = list(d_df.columns)
    temp_dict = {}
    temp_dict['имя признака'] = list_of_names
    temp_dict['тип'] = d_df.dtypes
    temp_dict['# значений'] = d_df.describe(include='all').loc['count']
    temp_dict['# пропусков(NaN)'] = d_df.isnull().sum().values 
    temp_dict['# уникальных'] = d_df.nunique().values
    if not short:
        temp_dict['в первой строке'] =d_df.loc[0].values
        temp_dict['во второй строке'] = d_df.loc[1].values
        temp_dict['в третьей строке'] = d_df.loc[2].values
    if full :
        temp_dict['минимум'] = d_df.describe(include='all').loc['min']
        temp_dict['среднее'] = d_df.describe(include='all').loc['mean']
        temp_dict['макс'] = d_df.describe(include='all').loc['max']
        temp_dict['медиана'] = d_df.describe(include='all').loc['50%']
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
    display(temp_df.T)

    return



def classic_round(d_num):
    return int(d_num + (0.5 if d_num > 0 else -0.5))


def my_round(d_pred):
    result = classic_round(d_pred*2)/2
    if result <=5:
        return result
    else:
        return 5


def test_model(d_df,d_list_remove_columns, d_RS):
    train_data = d_df.query('Sample == 1').drop(['Sample']+d_list_remove_columns, axis=1, errors='ignore')
    test_data = d_df.query('Sample == 0').drop(['Sample']+d_list_remove_columns, axis=1, errors='ignore')

    y = train_data.Rating.values
    X = train_data.drop(['Rating'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=d_RS)
    print(test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape)
    model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=d_RS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    my_vec_round = np.vectorize(my_round)
    y_pred = my_vec_round(y_pred)
    temp_MAE = metrics.mean_absolute_error(y_test, y_pred)

    print(temp_MAE)
    return


def simple_plot_barh_procent(d_name_title, d_category_names, d_name_column, d_df):
    """
    
    """
    list_values = list(d_df[d_name_column].unique())
    if len(d_category_names) != len(list_values):
        return print('Кол-во категорий не совпадает с кол-вом значений')
    else:
        temp_df = d_df[d_name_column].value_counts(normalize=True)*100

        plt.style.use('seaborn-paper')
        _, ax = plt.subplots(figsize=(10, 2))
        
        category_colors = plt.get_cmap('PuBu')(
                np.linspace(0.85, 0.35, len(list_values)))

        widths = temp_df.values
        starts = temp_df.cumsum() - widths
        ax.barh(0, width = widths, left=starts, height=0.3, color=category_colors)
        xcenters = starts + widths / 2
        text_color = 'white'
        for (x, w, c_n, k) in zip(xcenters, widths, d_category_names, temp_df.keys()):
            ax.text(x,-0.05, str(int(w))+'%', fontsize=18, weight = 'bold', ha='center', va='center',
                    color=text_color)
            ax.text(x,0.05, c_n +'  ('+str(k)+')', fontsize=14, weight = 'bold', ha='center', va='center',
                    color=text_color)
        ax.set_title(d_name_title+' (критерий '+d_name_column+')', loc ='center', fontsize=12, color = category_colors[0])

        plt.show()
    return


def simple_balalayka(d_category_names, 
                     d_name_column_base_x, 
                     d_name_column_group_y, 
                     d_df, 
                     d_my_font_scale):
    """
    
    """
    list_values = list(d_df[d_name_column_base_x].unique())
    if len(d_category_names) != len(list_values):
        return print('Кол-во категорий не совпадает с кол-вом значений')
    else:
        plt.style.use('seaborn-paper')
        plt.subplots(figsize=(12, 4))
        color_text = plt.get_cmap('PuBu')(0.85)

        sns.set(font_scale=d_my_font_scale, style='whitegrid')
        plt.subplot(111)
        b = sns.swarmplot(x=d_name_column_base_x, y=d_name_column_group_y, data=d_df, label= d_name_column_base_x[0], palette="PuBu")
        b.set_title(f'Распределение {d_name_column_base_x} and {d_name_column_group_y}', 
                        fontsize=12, color = color_text)
        b.set_ylabel(d_name_column_group_y, fontsize=14, color = color_text)
        b.set_xlabel(d_name_column_base_x, fontsize=14, color = color_text)

        b.legend(labels=d_category_names, ncol=2, fancybox=True, framealpha=0.75, shadow=True, bbox_to_anchor=(0.5, -0.3), loc='center')
    return


def simple_boxplot(d_category_names, 
                     d_name_column_base_x, 
                     d_name_column_group_y, 
                     d_df, 
                     d_my_font_scale,
                     log = False):
    """
    
    """
    
    list_values = list(d_df[d_name_column_base_x].unique())
    if len(d_category_names) != len(list_values):
        return print('Кол-во категорий не совпадает с кол-вом значений')
    else:
        temp_df = d_df
        if log:
            
            temp_df[d_name_column_group_y] = temp_df[d_name_column_group_y].apply(lambda x: math.log(x+1))
        
        plt.style.use('seaborn-paper')
        plt.subplots(figsize=(6, 4))
        color_text = plt.get_cmap('PuBu')(0.85)

        sns.set(font_scale=d_my_font_scale, style='whitegrid')
        plt.subplot(111)
        b = sns.boxplot(x=d_name_column_base_x, y=d_name_column_group_y, data=temp_df, palette="PuBu")
        b.set_title(f'boxplot распределения значений {d_name_column_base_x} по {d_name_column_group_y}', 
                        fontsize=12, color = color_text)
        b.set_ylabel(d_name_column_group_y, fontsize=14, color = color_text)
        b.set_xlabel(d_name_column_base_x, fontsize=14, color = color_text)

        b.legend(labels=d_category_names, ncol=2, fancybox=True, framealpha=0.75, shadow=True, bbox_to_anchor=(0.5, -0.3), loc='center')
    return


def plot_filter_df_kde(d_category_names, 
                       d_name_column_filter, 
                       d_name_column_group_x, 
                       d_df,
                       d_my_font_scale):
    """
    
    """
    list_values = list(d_df[d_name_column_filter].unique())
    list_values = sorted(list_values)

    if len(d_category_names) != len(list_values):
        return print('Кол-во категорий не совпадает с кол-вом значений')
    else:
        for i in range(len(d_category_names)):
            d_category_names[i] = d_category_names[i] + ' ('+str(list_values[i])+')'
        plt.style.use('seaborn-paper')
        plt.subplots(figsize=(6, 4))

        category_colors = plt.get_cmap('PuBu')(np.linspace(0.85, 0.35, len(list_values)))
        color_text = plt.get_cmap('PuBu')(0.85)

        sns.set(font_scale=d_my_font_scale, style='whitegrid')
        plt.subplot(111)
        for x,i in zip(list_values, range(len(list_values))):
            temp_df = d_df.loc[d_df[d_name_column_filter] == x, d_name_column_group_x]
            k=sns.kdeplot(temp_df, color=category_colors[i], label=x)

        k.set_title(f'Плотность распределений {d_name_column_group_x} с фильтрами по  {d_name_column_filter}', 
                        fontsize=12, color = color_text)
        k.set_xlabel(d_name_column_group_x, fontsize=14, color = color_text)

        k.legend(labels=d_category_names, ncol=len(list_values), fancybox=True, framealpha=0.75, shadow=True, bbox_to_anchor=(0.5, -0.3), loc='center')
    return


def group_plot_hbar_count(d_category_names, 
                         d_name_column_base_x, 
                         d_name_column_group_y, 
                         d_df,
                         d_my_font_scale):
    """
    
    """
    list_values = list(d_df[d_name_column_base_x].unique())
    if len(d_category_names) != len(list_values):
        return print('Кол-во категорий не совпадает с кол-вом значений')
    else:
        temp_df = d_df.copy()
        plt.style.use('seaborn-paper')
        plt.subplots(figsize=(6, 4))
        color_text = plt.get_cmap('PuBu')(0.85)
        sns.set(font_scale=d_my_font_scale, style='whitegrid')

        plt.subplot(111)
        b = sns.barplot(y=d_name_column_base_x, 
                        x=d_name_column_group_y, 
                        data=temp_df, 
                        palette="PuBu", 
                        ci=None, 
                        orient ='h',
                        hue=d_name_column_base_x)

        b.set_title(f'vplot распределения сред. знач. {d_name_column_group_y} сгруп-ные по {d_name_column_base_x}', 
                    fontsize=12, color = color_text)
        b.set_ylabel(d_name_column_group_y, fontsize=14, color = color_text)
        b.set_xlabel(d_name_column_base_x, fontsize=14, color = color_text)

        b.legend(labels=d_category_names, ncol=2, fancybox=True, framealpha=0.75, shadow=True, bbox_to_anchor=(0.5, -0.3), loc='center')
        
    return

def group_plot_barv_mean(d_category_names, 
                         d_name_column_base_x, 
                         d_name_column_group_y, 
                         d_df,
                         d_my_font_scale):
    """
    
    """
    list_values = list(d_df[d_name_column_base_x].unique())
    if len(d_category_names) != len(list_values):
        return print('Кол-во категорий не совпадает с кол-вом значений')
    else:
        temp_df = d_df.copy()
        plt.style.use('seaborn-paper')
        plt.subplots(figsize=(6, 4))
        color_text = plt.get_cmap('PuBu')(0.85)
        sns.set(font_scale=d_my_font_scale, style='whitegrid')

        plt.subplot(111)
        b = sns.barplot(x=d_name_column_base_x, y=d_name_column_group_y, data=temp_df, palette="PuBu", ci=None, hue=d_name_column_base_x)

        b.set_title(f'vplot распределения сред. знач. {d_name_column_group_y} сгруп-ные по {d_name_column_base_x}', 
                    fontsize=12, color = color_text)
        b.set_ylabel(d_name_column_group_y, fontsize=14, color = color_text)
        b.set_xlabel(d_name_column_base_x, fontsize=14, color = color_text)

        b.legend(labels=d_category_names, ncol=2, fancybox=True, framealpha=0.75, shadow=True, bbox_to_anchor=(0.5, -0.3), loc='center')
        
    return



def simple_heatmap(d_title, d_df, d_list_of_columns, d_my_font_scale, d_g, d_size):
    """
    
    """
    temp_df = d_df[d_list_of_columns].copy()

    plt.style.use('seaborn-paper')
    plt.subplots(figsize=(d_size, 6))
    color_text = plt.get_cmap('PuBu')(0.85)
    sns.set(font_scale=d_my_font_scale, style='whitegrid')

    plt.subplot(111)
    h = sns.heatmap(temp_df.corr(), annot = True, fmt=f'.{d_g}g', cmap= "PuBu", center= 0)
    h.set_title(d_title,  fontsize=14, color = color_text)

    return

def binned(df, col_name, bins_no=11):
    # df - имя датафрейма, col_name - наименование признака, который надо разбить на интервалы, 
    # bins - количество интервалов разбиения
    # пример: data['age_binned'] = binned(data,'age',18)

    if not pd.api.types.is_numeric_dtype(df[col_name]):
        print(f'Признак {col_name} не численный, разбиение невозможно')
        return
    else:
        # Вычисляем минимум и максимум разбиения
        bottom = df[col_name].min()
        top = df[col_name].max()
        # Возвращаем признак, разбитый на интервалы
        return pd.cut(df[col_name], bins = np.linspace(bottom, top, num = bins_no))

def StandardScaler_FillNa_0(d_df):
    return




if __name__ == "__main__":
    print('What do you do?')