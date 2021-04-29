import streamlit as st
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb

def descr_apply_index(row):
    global dict_descr_feats
    return dict_descr_feats[row.name]

# Title
st.title('Кредитный скорринг на транзакциях')
st.markdown("""
Автор: Соколов Александр ([код на github](https://github.com/alex-sokolov2011/skillfactory_rds/tree/master/diplom))\n
* Это веб приложение иллюстрирует модель прогноза выхода клиента банка в дефолт на данных карточных транзакций.
* Транзакционные данные клиентов использованы с соревнования AlfaBattle2.0 (декабрь 2020 года).
* ***Если возникнут вопросы, пишите в [Telegram](https://t.me/aleks_2011).***
""")

#Задаем глобальные контанты
min_proba_non_defolt = 0.0000301
max_proba_defolt = 0.652771
border_class_proba = 0.11540058494271957

#Загружаем данные
PATH_to_file = './data/'
with open(PATH_to_file + 'list_clients.pickle', 'rb') as f:
    list_clients = pickle.load(f)
with open(PATH_to_file + 'model/feats_model3.pickle', 'rb') as f:
    list_feats = pickle.load(f)
with open(PATH_to_file + 'dict_descr_feats.pickle', 'rb') as f:
    dict_descr_feats = pickle.load(f)


data = pd.read_csv(PATH_to_file+'data.csv')

model1 = lgb.Booster(model_file=PATH_to_file + 'model/model3_1.txt')
model2 = lgb.Booster(model_file=PATH_to_file + 'model/model3_2.txt')
model3 = lgb.Booster(model_file=PATH_to_file + 'model/model3_3.txt')
model4 = lgb.Booster(model_file=PATH_to_file + 'model/model3_4.txt')
model5 = lgb.Booster(model_file=PATH_to_file + 'model/model3_5.txt')

# Filters
st.sidebar.header('Окно выбора опций')
dict_steps = {
    'прогноз модели':1,
    'интерпретация модели':2
}
list_steps = list(dict_steps.keys())
selected_model = st.sidebar.selectbox(label='Выберите опцию:', options=list_steps)
ch_step = dict_steps[selected_model]


if ch_step == 1:
    
    selected_id_client = st.sidebar.selectbox(label='Выберите id заявки клиента:', options=list_clients)

    selected_feats = st.sidebar.multiselect(label='Выберите признаки транзакций', options=sorted(list_feats), 
        default=['hour_diff_median', 'days_before_max', 'hour_diff_max', 'product', 'days_before_median', 
                    'count_mcc_category_2', 'mean_operation_type_4', 'hour_diff_var', 'count_mcc_category_7', 
                    'days_before_min'])

    st.header('Прогноз модели')
    'id заявки клиента (app_id): ', selected_id_client

    temp_df=data[data['app_id']==selected_id_client]
    train_preds1 = model1.predict(temp_df[list_feats])
    train_preds2 = model2.predict(temp_df[list_feats])
    train_preds3 = model3.predict(temp_df[list_feats])
    train_preds4 = model4.predict(temp_df[list_feats])
    train_preds5 = model5.predict(temp_df[list_feats])
    y_proba = (train_preds1 + train_preds2 + train_preds3 + train_preds4 + train_preds5)/5 

    
    'Прогноз модели выхода клиента в дефолт: ', y_proba[0]
    if y_proba[0]>=border_class_proba:
        y_proba_proc = (y_proba[0]-border_class_proba)/(max_proba_defolt-border_class_proba)
        y_proba_proc = round(50+(50*y_proba_proc),0)
    else:
        y_proba_proc = (border_class_proba - y_proba[0])/(border_class_proba - min_proba_non_defolt)
        y_proba_proc = round(50-(50*y_proba_proc),0)
    'Расчетная вероятность выхода клиента в дефолт: ', y_proba_proc, '%'

    st.header('Агрегированные статистики по транзакциям клиента')
    temp_df = temp_df[selected_feats].T
    temp_df.columns = ['Значение']
    temp_df['Описание признака'] = temp_df.apply(descr_apply_index, axis=1)
    st.dataframe(temp_df)
    'По умолчанию установлен список 10 самых важных для модели признаков *(вы можете скролить вниз или открыть на полный экран)*'
    st.image('./pictures/feats_imp.png')

elif ch_step == 2:
    st.header('Интерпретация прогнозов модели')
    st.image('./pictures/shap.png')
    st.markdown("""
    * этот график показывает влияние и разделяющую способность признаков:
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





