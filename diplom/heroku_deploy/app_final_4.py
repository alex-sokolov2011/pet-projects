import streamlit as st
# import numpy as np
import pandas as pd
import pickle
# import lightgbm as lgb

def descr_apply_index(row):
    global dict_descr_feats
    return dict_descr_feats[row.name]

# Title
st.title('Кредитный скорринг')
st.markdown("""
Автор: Соколов Александр ([код на github](https://github.com/alex-sokolov2011))\n
* Это веб приложение иллюстрирует модель прогноза выхода клиента банка в дефолт по транзакционным данным.
* Транзакционные данные клиентов использованы с соревнования AlfaBattle2.0 (декабрь 2020 года).
* ***Если возникнут вопросы, пишите в [Telegram](https://t.me/aleks_2011).***
""")

#Задаем глобальные контанты
min_proba_non_defolt = 0.000301
max_proba_defolt = 0.652771
border_class_proba = 0.11540058494271957

#Загружаем данные
PATH_to_file = './data/'
with open(PATH_to_file + 'list_defolt.pickle', 'rb') as f:
    list_defolt = pickle.load(f)
with open(PATH_to_file + 'list_no_defolt.pickle', 'rb') as f:
    list_no_defolt = pickle.load(f)
with open(PATH_to_file + 'top_selected_feats2.pickle', 'rb') as f:
    list_feats = pickle.load(f)
with open(PATH_to_file + 'dict_descr_feats.pickle', 'rb') as f:
    dict_descr_feats = pickle.load(f)
with open(PATH_to_file + 'df_descr_data.pickle', 'rb') as f:
    df_descr_data = pickle.load(f)

df_y_proba_defolt = pd.read_csv(PATH_to_file+'df_y_proba_defolt.csv')
df_y_proba_no_defolt = pd.read_csv(PATH_to_file+'df_y_proba_no_defolt.csv')

data_train = pd.read_csv(PATH_to_file+'data_train.csv')

# lgb_model1_1 = lgb.Booster(model_file='./data/model1/mode_1.txt')
# lgb_model1_2 = lgb.Booster(model_file='./data/model1/mode_2.txt')
# lgb_model1_3 = lgb.Booster(model_file='./data/model1/mode_3.txt')
# lgb_model1_4 = lgb.Booster(model_file='./data/model1/mode_4.txt')
# lgb_model1_5 = lgb.Booster(model_file='./data/model1/mode_5.txt')

# lgb_model2_1 = lgb.Booster(model_file='./data/model2/mode_1.txt')
# lgb_model2_2 = lgb.Booster(model_file='./data/model2/mode_2.txt')
# lgb_model2_3 = lgb.Booster(model_file='./data/model2/mode_3.txt')
# lgb_model2_4 = lgb.Booster(model_file='./data/model2/mode_4.txt')
# lgb_model2_5 = lgb.Booster(model_file='./data/model2/mode_5.txt')



# df_train_proto = pd.read_csv(PATH_to_file+'df_train_proto.csv')
# features = [x for x in df_train_proto.columns if x not in ['app_id', 'flag']]
# train_preds1 = lgb_model1_1.predict(df_train_proto[features])
# train_preds2 = lgb_model1_2.predict(df_train_proto[features])
# train_preds3 = lgb_model1_3.predict(df_train_proto[features])
# train_preds4 = lgb_model1_4.predict(df_train_proto[features])
# train_preds5 = lgb_model1_5.predict(df_train_proto[features])
# temp_blend_train = (train_preds1 + train_preds2 + train_preds3 + train_preds4 + train_preds5)/5 
# df_train_proto['flag_pred']=temp_blend_train

# df_y_true_pred_model1 = pd.read_csv(PATH_to_file+'df_y_true_pred_model1.csv')
# y_true_model1 = df_y_true_pred_model1['y_true']
# y_pred_model1 = df_y_true_pred_model1['y_pred']
# y_pred_proba_model1 = df_y_true_pred_model1['y_pred_proba']

# df_y_true_pred_model2 = pd.read_csv(PATH_to_file+'df_y_true_pred_model2.csv')
# y_true_model2 = df_y_true_pred_model2['y_true']
# y_pred_model2 = df_y_true_pred_model2['y_pred']
# y_pred_proba_model2 = df_y_true_pred_model2['y_pred_proba']




# Filters
st.sidebar.header('Окно выбора опций')
dict_steps = {
    'прогноз модели':1,
    'интерпретация модели':2,
    'описание исходных данных':3
}
list_steps = list(dict_steps.keys())
selected_model = st.sidebar.selectbox(label='Выберите опцию:', options=list_steps)
ch_step = dict_steps[selected_model]


if ch_step == 1:
    dict_type_customer = {
        'Дефолтный':1,
        'Не дефолтный':2,
    }
    list_steps = list(dict_type_customer.keys())
    selected_type_customer = st.sidebar.selectbox(label='Выберите тип клиента:', options=list_steps)
    ch_type_customer = dict_type_customer[selected_type_customer]

    if ch_type_customer == 1:
        selected_id_customer = st.sidebar.selectbox(label='Выберите id клиента:', options=list_defolt)

        selected_feats = st.sidebar.multiselect(label='Выберите признаки транзакций', options=sorted(list_feats), 
            default=['hour_diff_median', 'days_before_max', 'hour_diff_max', 'product', 'days_before_median', 
                     'count_mcc_category_2', 'mean_operation_type_4', 'hour_diff_var', 'count_mcc_category_7', 
                     'days_before_min'])

        st.header('Прогноз модели')
        'id клиента: ', selected_id_customer
        df_y_proba = df_y_proba_defolt[df_y_proba_defolt['app_id']==selected_id_customer]
        y_proba = df_y_proba.iloc[0]['y_pred_proba']
        'Прогноз модели выхода клиента в дефолт: ', y_proba
        y_proba_proc = (y_proba-border_class_proba)/(max_proba_defolt-border_class_proba)
        y_proba_proc = round(50+(50*y_proba_proc),0)
        'Расчетная вероятность выхода клиента в дефолт: ', y_proba_proc, '%'

        st.header('Агрегированные статистики по транзакциям клиента')
        temp_df = data_train[data_train['app_id']==selected_id_customer][['app_id']+ selected_feats].T
        temp_df.columns = ['Значение']
        temp_df['Описание признака'] = temp_df.apply(descr_apply_index, axis=1)
        st.dataframe(temp_df)
        'По умолчанию установлен список 10 самых важных для модели признаков *(вы можете скролить вниз или открыть на полный экран)*'
        st.image('./pictures/feats_imp.png')

    
    else:
        selected_id_customer = st.sidebar.selectbox(label='Выберите id клиента:', options=list_no_defolt)

        selected_feats = st.sidebar.multiselect(label='Выберите признаки транзакций', options=sorted(list_feats), 
            default=['hour_diff_median', 'days_before_max', 'hour_diff_max', 'product', 'days_before_median', 
                     'count_mcc_category_2', 'mean_operation_type_4', 'hour_diff_var', 'count_mcc_category_7', 
                     'days_before_min'])

        st.header('Прогноз модели')
        'id клиента: ', selected_id_customer
        df_y_proba = df_y_proba_no_defolt[df_y_proba_no_defolt['app_id']==selected_id_customer]
        y_proba = df_y_proba.iloc[0]['y_pred_proba']
        'Прогноз модели выхода клиента в дефолт: ', y_proba
        y_proba_proc = (border_class_proba - y_proba)/(border_class_proba - min_proba_non_defolt)
        y_proba_proc = round(50-(50*y_proba_proc),0)
        'Расчетная вероятность выхода клиента в дефолт: ', y_proba_proc, '%'

        st.header('Агрегированные статистики по транзакциям клиента')
        temp_df = data_train[data_train['app_id']==selected_id_customer][['app_id']+ selected_feats].T
        temp_df.columns = ['Значение']
        temp_df['Описание признака'] = temp_df.apply(descr_apply_index, axis=1)
        st.dataframe(temp_df)
        'По умолчанию установлен список 10 самых важных для модели признаков *(вы можете скролить вниз или открыть на полный экран)*'
        st.image('./pictures/feats_imp.png')
elif ch_step == 2:
    st.header('Интерпретация прогнозов модели')
    st.image('./pictures/shap.png')
    st.markdown("""
    * этот график влияние и разделяющую способность признаков:
      * каждая точка графика это клиент
      * цвет - значение прогноза дефолта, чем краснее - тем прогноз дефолта выше, и наоборот
      * горизонтальное положение точки показывает, приводит ли значение конкретной фичи этого клиента к росту предсказания дефолта, или наоборот. 
      
    **Например можно увидеть следующие тренды:**\n 
    1. уменьшение признака 'hour_diff_median' приводит к росту значения целевой переменной (то есть можно предположить, что клиенту приходится делать транзакции слишком часто - а это свидетельство того, что клиент вероятнее выйдет в дефолт, чем клиент который чувствует себя более уверенно в финансовом плане и может позволить делать крупные оплаты сразу не разбивая их на части)
    2. увеличение признака 'hour_diff_max' приводит к росту значения целевой переменной (то есть можно предположить, что клиент долго не пользовался картой это могло произойти в следствии овердрафта и необходимости перекредитования в другом банке)
    3. уменьшение признака 'count_mcc_category_9' приводит к росту значения целевой переменной (к сожалению категории торговых точек как данные о клиентах были обезличены, остается только предположить, что вероятно это категория 'Кафе,бары и рестораны' и клиенты испытывающие сложности в финансах перестают их посещать или делают это существенно реже.)
    4. увеличение признака 'count_operation_type_5' приводит к росту значения целевой переменной (из-за обезличивания типа операций и дополнительного анализа видно что у большинства клиентов таких операций нет, вероятно речь об операции внесение минимального платежа, вместо гашения очередного платежа. И собственно если это так что увеличение кол-ва таких операций приводит к дефолту потому что нагрузка в этом случае только возрастает)
    5. среднее и максимальное значение признака 'product' приводит к росту значения целевой переменной (предположу что 0 - кредит безналичными на карту, 1 - это кредит наличными, 2 - кредит на крупные покупки, 3 - кредит на покупку авто, 4 - ипотека. Сделать такое предположение можно и-за того что 80% всех кредитов по кол-ву приходится на продукты 0,1. А самый редкий это продукт 4. Среднее значение дефолтов 2-2,5% процента по продуктам 0,1,3. По продукту 2 - 7%. А по продукту 4 - 3%. Вероятно кредиты на крупные покупки проверяются менее чем ипотека и они не имеют обеспечения как авто для погашения кредита.)
    6. уменьшение признака 'days_before_max' приводит к росту значения целевой переменной (низкое значение максимума кол-ва дней перед взятием кредита означает короткая история транзакций в банке. Можно предположить, что клиент открыл карту и через короткое время получил кредит. Это либо мошеническая схема, но этот вариант мы не рассматривает ибо он скорее всего не массовый. А вероятно это случай перекредитования в другом банке, когда банк по соновной карте перестал предоставлять или увеличивать кредитную линию. Что является свидетельством неплатежеспособности клиента.)
    """)
elif ch_step == 3:
    st.header('Описание задания соревнования AlfaBattle2.0')
    st.markdown("""
    Участникам соревнования предстояло оценить вероятность того, что клиент выйдет в дефолт, основываясь на истории потребительского поведения по карточным транзакциям.
    
    Каждая такая транзакция содержит информацию о сумме покупки, месте, дате, mcc-категории, валюте и признаки от платежной системы.

    Обучающая выборка собрана за N дней, тестовая выборка за последующие K дней.
    """)
    st.header('Описание 19 признаков исходных данных')
    st.dataframe(df_descr_data)
    '*примечание: вы можете скролить вниз или открыть на полный экран*'




