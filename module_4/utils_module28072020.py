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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
np.warnings.filterwarnings('ignore')

import os

global last_pred


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

# %% [code]
