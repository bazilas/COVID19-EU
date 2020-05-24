import numpy as np 
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


confirmed_data = pd.read_csv('/Users/vb/Code/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_data = pd.read_csv('/Users/vb/Code/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

fields = confirmed_data.keys()
print(fields)

EU_Countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France',
'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland',
'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']

# Per day infections for a country (function)
def get_per_day_data(data, fields):
    days = []
    data_per_day = []
    for i in data.loc[:, fields[4]:fields[-1]]:
        days.append(i[0:-3])
        data_per_day.append(data[i].sum())
    #convert to numpy
    data_per_day = np.array(data_per_day).reshape(-1,1)
    return days, data_per_day

# Forecting for next seven days
def get_forecact(data):
    model = SARIMAX(data, trend='c', order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False, transparams=False)
    return(model_fit.predict(len(data), len(data)+6, typ='levels'))

# Per day infections for a country (function)
def get_per_day_forecast(data_loaded, fields, EU_Countries, data_per_day_EU, data_EU_sum, data_week_forecast):
    for c in EU_Countries:
        instances = data_loaded[data_loaded['Country/Region'] == c]
        days, data_per_day = get_per_day_data (instances, fields)

        # sum up the facts for plotting later sorted results
        data_EU_sum.append(int(data_per_day.sum()))
        data_per_day_EU.append(data_per_day)

        data = data_per_day.flatten()
        data_week_forecast.append(get_forecact(data))
    return days

        #iso_reg = IsotonicRegression().fit(x.flatten(), confirmed_per_day.flatten())

        #kernel = ['rbf']
        #c = [0.1, 1, 10, 100]
        #gamma = [0.1, 1, 10, 100]
        #epsilon = [0.1, 1, 10, 100]
        #shrinking = [True, False]
        #svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}
        #svm = svm.SVR()
        #svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_absolute_error', cv=2, return_train_score=True, n_jobs=-1, n_iter=20, verbose=1)
        #svm_search.fit(x, confirmed_per_day.flatten())
        #svm_best_model = svm_search.best_estimator_
        #svm_pred = svm_best_model.predict(x[-1])


# Per day infections and forecasting
confirmed_per_day_EU = []
confirmed_EU_sum = []
confirmed_week_forecast = []
days = get_per_day_forecast(confirmed_data, fields, EU_Countries, confirmed_per_day_EU, confirmed_EU_sum, confirmed_week_forecast)

# Per day deaths and forecasting
deaths_per_day_EU = []
deaths_EU_sum = []
deaths_week_forecast = []
days = get_per_day_forecast(deaths_data, fields, EU_Countries, deaths_per_day_EU, deaths_EU_sum, deaths_week_forecast)

# Per day ratio of deaths / infections
deaths_infec_per_day_EU = []
deaths_infec_EU_sum = []
deaths_infec_week_forecast = []
for i,j in zip(deaths_per_day_EU, confirmed_per_day_EU):
    ratio = np.nan_to_num(np.divide(i,j))
    deaths_infec_per_day_EU.append(ratio)
    deaths_infec_EU_sum.append(int(ratio.sum()))
    deaths_infec_week_forecast.append(get_forecact(ratio.flatten()))

# Sorted for visualization from most confirmed / deaths to less confirmed /deaths - 
confirmed_per_day_EU_idx = sorted(range(len(confirmed_EU_sum)), key=lambda k: confirmed_EU_sum[k])
deaths_per_day_EU_idx = sorted(range(len(deaths_EU_sum)), key=lambda k: deaths_EU_sum[k])
deaths_infec_per_day_EU_idx = sorted(range(len(deaths_infec_EU_sum)), key=lambda k: deaths_infec_EU_sum[k])

# forecasting days
days_with_forecast = ['+' + str(i+1) for i in range(confirmed_week_forecast[0].shape[0])]

sns.set()
np.random.seed(55)
colors = plt.cm.hsv(np.random.rand(30,))

def plot_statistics(x_label_text, y_label_text, plt_title, fig_dim1, fig_dim2, colors, legend_font_size, 
EU_Countries, sorted_index, data_per_day, days, days_with_forecast, forecast_data, top_n, gradient_flg=False):
    plt.rcParams['figure.figsize'] = [fig_dim1, fig_dim2]
    for cnt, i in enumerate(reversed(sorted_index)):
        if gradient_flg:
            vals = np.append([0], np.gradient(data_per_day[i].flatten()))
            vals = np.gradient(data_per_day[i].flatten())
            plt.plot(days, vals , label=EU_Countries[i], color=colors[cnt])
        else:
            plt.plot(days, data_per_day[i], label=EU_Countries[i], color=colors[cnt])
            plt.scatter(days_with_forecast, forecast_data[i], s=10, marker='o', color=colors[cnt])

        if cnt == top_n:
            break
    plt.ylabel(y_label_text)
    plt.xlabel(x_label_text)
    plt.title(plt_title)
    plt.legend(loc='upper left')
    plt.xticks(fontsize=legend_font_size, rotation=90)
    plt.tight_layout()
    plt.show()

# Infected top-10 rate of change (gradient)
plot_statistics('Date', 'COVID-19 Infected Rate of Change', 'The 10 most infected EU countries and 7 days forecasting (Johns Hopkins CSSE data source)',
 10, 5, colors, 8, EU_Countries, confirmed_per_day_EU_idx, confirmed_per_day_EU, days, days_with_forecast, confirmed_week_forecast, 10, True)

# Infected top-10
plot_statistics('Date', 'COVID-19 Infected (Per Day)', 'The 10 most infected EU countries and 7 days forecasting (Johns Hopkins CSSE data source)',
 10, 5, colors, 8, EU_Countries, confirmed_per_day_EU_idx, confirmed_per_day_EU, days, days_with_forecast, confirmed_week_forecast, 10)

# Deaths top-10
plot_statistics('Date', 'COVID-19 Deaths (Per Day)', 'The first 10 EU countries with most deaths and 7 days forecasting (Johns Hopkins CSSE data source)',
 10, 5, colors, 8, EU_Countries, deaths_per_day_EU_idx, deaths_per_day_EU, days, days_with_forecast, deaths_week_forecast, 10)

# Deaths / Infections top-10
plot_statistics('Date', 'COVID-19 Deaths / Infections (Per Day)', 'The first 10 death / infection rate of the EU countries and 7 days forecasting (Johns Hopkins CSSE data source)',
 10, 5, colors, 8, EU_Countries, deaths_infec_per_day_EU_idx, deaths_infec_per_day_EU, days, days_with_forecast, deaths_infec_week_forecast, 10)

# Infected all EU
plot_statistics('Date', 'COVID-19 Infected (Per Day)', 'The infected EU countries and 7 days forecasting (Johns Hopkins CSSE data source)',
 15, 8, colors, 8, EU_Countries, confirmed_per_day_EU_idx, confirmed_per_day_EU, days, days_with_forecast, confirmed_week_forecast, len(confirmed_per_day_EU_idx))

# Deaths all EU
plot_statistics('Date', 'COVID-19 Deaths (Per Day)', 'The deaths of the EU countries and 7 days forecasting (Johns Hopkins CSSE data source)',
 15, 8, colors, 8, EU_Countries, deaths_per_day_EU_idx, deaths_per_day_EU, days, days_with_forecast, deaths_week_forecast, len(deaths_per_day_EU_idx))

# Deaths / Infections all EU
plot_statistics('Date', 'COVID-19 Deaths / Infections (Per Day)', 'The death / infection rate of the EU countries and 7 days forecasting (Johns Hopkins CSSE data source)',
 15, 8, colors, 8, EU_Countries, deaths_infec_per_day_EU_idx, deaths_infec_per_day_EU, days, days_with_forecast, deaths_infec_week_forecast, len(deaths_infec_per_day_EU_idx))
