# %% [code]
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
np.warnings.filterwarnings('ignore')

import os


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
    epsilon = 10**(-7)
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (12, 3)

    _, axs = plt.subplots(1, 4)
    temp_df = d_df
    axs[0].hist(temp_df,bins=11)
    axs[0].set_title(d_name_plot)
    axs[1].boxplot(temp_df)
    axs[1].set_title('')
    temp_df = d_df.apply(lambda x: math.log(x+epsilon))
    axs[2].hist(temp_df,bins=11)
    axs[2].set_title('log')
    axs[3].boxplot(temp_df)
    axs[3].set_title('')
    return

def four_plot_with_log2(d_name_column,d_df):
    epsilon = 10**(-7)
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (12, 3)
    
    temp_df=d_df

    plt.subplot2grid((1, 4), (0, 0))
    temp_df[d_name_column].hist(bins=11)

    plt.subplot2grid((1, 4), (0, 1))
    temp_df.boxplot([d_name_column])
    
    temp_name = 'log_'+d_name_column

    d_df.loc[:,temp_name] = d_df[d_name_column].apply(lambda x: math.log(x+epsilon))
    plt.subplot2grid((1, 4), (0, 2))
    temp_df[temp_name].hist(bins=11)

    plt.subplot2grid((1, 4), (0, 3))
    d_df.boxplot([temp_name])
    plt.show()
    return


def big_hist(d_name_column,d_df):
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (12, 3)

    temp_df = d_df
    temp_df[d_name_column].hist(bins=50)

    return


def big_hist_log(d_name_column,d_df):
    epsilon = 10**(-7)
    plt.style.use('seaborn-paper')
    plt.rcParams['figure.figsize'] = (12, 3)

    temp_df = d_df
    temp_df['log_'+d_name_column] = temp_df[d_name_column].apply(lambda x: math.log(x+epsilon))

    temp_df['log_'+d_name_column].hist(bins=50)

    return


def borders_of_outliers(d_name_column,d_df, log = False):
    epsilon = 10**(-3)
    if log:
        temp_df = d_df[d_name_column].apply(lambda x: math.log(x+epsilon))
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
        temp_dict['границы выбросов без логарифма'] = [math.exp(left_border)-epsilon, math.exp(right_border)-epsilon]
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
    n_bins = d_df.nunique()
    if n_bins >15:
        n_bins = 15
    d_df.hist(bins=n_bins)
    return


def describe_without_plots(d_name_plot,d_df):
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

    return


def describe_without_plots_all_collumns(d_df, full=True):
    list_of_names = list(d_df.columns)
    temp_dict = {}
    temp_dict['имя критерия'] = list_of_names
    temp_dict['тип'] = d_df.dtypes
    temp_dict['# значений'] = d_df.describe(include='all').loc['count']
    temp_dict['# пропусков(NaN)'] = d_df.isnull().sum().values 
    temp_dict['# уникальных'] = d_df.nunique().values
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
            epsilon = 10**(-7)
            temp_df[d_name_column_group_y] = temp_df[d_name_column_group_y].apply(lambda x: math.log(x+epsilon))
        
        plt.style.use('seaborn-paper')
        plt.subplots(figsize=(8, 4))
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


def group_plot_barh_procent(d_name_title, d_category_names, d_name_column_group, d_name_column_base, d_df):
    """
    
    """

    return

def StandardScaler_FillNa_0(d_df):
    return




if __name__ == "__main__":
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_train['Train'] = 1 # помечаем где у нас трейн
    df_test['Train'] = 0 # помечаем где у нас тест
    df_test['Survived'] = 2 # в тесте у нас нет значения Survived, мы его должны предсказать, по этому пока просто заполняем двойками
    df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
    simple_plot_barh_procent('Распределение выживших и не выживших', ['Не выжили','Выжили'], 'Survived',df[df['Train']==1])
    simple_balalayka(['Не выжили','Выжили'], 'Survived', 'Age',df[df['Train']==1], 1.5)

# %%
