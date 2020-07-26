#!/usr/bin/env python

"""
Outliers detection with HP filter, Z-scores and MAD test
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib.pyplot as plt


# Hodrick Prescott filter
def hp_filter(x, lamb=14400):
    w = len(x)
    b = [[1]*w, [-2]*w, [1]*w]
    D = sparse.spdiags(b, [0, 1, 2], w-2, w)
    I = sparse.eye(w)
    B = (I + lamb*(D.transpose()*D))
    return sparse.linalg.dsolve.spsolve(B, x)


def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)

def zscore(data, axis=None):
    if isinstance(data, np.ndarray):
        return stats.zscore(data, axis=1, ddof=1)


def anomaly_detection(x, alpha=0.05, lamb=1600):
    """
    x         : pd.Series
    alpha     : The level of statistical significance with which to
                accept or reject anomalies. (expon distribution)
    lamb      : penalize parameter for hp filter
                Appropriate values of the smoothing parameter depend upon
                the periodicity of the data. Suggested values are:
                    Yearly — 100
                    Quarterly — 1600
                    Monthly — 14400
    return r  : Data frame containing the index of anomaly
    """
    # calculate residual
    xhat = hp_filter(x, lamb=lamb)
    resid = x - xhat

    # drop NA values
    ds = pd.Series(resid)
    ds = ds.dropna()

    # Remove the seasonal and trend component,
    # and the median of the data to create the univariate remainder
    md = np.median(x)
    print("Median: {}".format(md))
    data = ds - md

    # process data, using median filter
    ares = (data - data.median()).abs()
    data_sigma = data.mad() + 1e-12
    ares = ares/data_sigma

    # compute significance
    p = 1. - alpha
    R = stats.expon.interval(p, loc=ares.mean(), scale=ares.std())
    threshold = R[1]

    # extract index, np.argwhere(ares > md).ravel()
    r_id = ares.index[ares > threshold]

    return r_id

# demo
def test():
    # fix
    np.random.seed(42)

    # sample signals
    N = 1024  # number of sample points
    t = np.linspace(0, 2*np.pi, N)
    y = np.sin(t) + 0.02*np.random.randn(N)

    # outliers are assumed to be step/jump events at sampling points
    M = 3  # number of outliers
    for ii, vv in zip(np.random.rand(M)*N, np.random.randn(M)):
        y[int(ii):] += vv

    # detect anomaly
    r_idx = anomaly_detection(y, alpha=0.01)

    # plot the result
    plt.figure()
    plt.plot(y, 'b-')
    plt.plot(r_idx, y[r_idx], 'ro')
    plt.savefig('U:\\ts_anomaly.png')

def plot_save(newer_m,y,r_idx):
    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    fig, ax = plt.subplots(figsize=(23,15))

    ax.plot(r_idx, y[r_idx], 'ro',label="Anomaly")
    ax.plot(newer_m.index,y,'b-',label="Negative sentiment",color='royalblue',alpha=0.9)

    ax.axvline(pd.to_datetime('2008-09-15'),color='orange',label="Lehman Brothers", linestyle='--', lw=6,alpha=0.5)
    ax.axvline(pd.to_datetime('2013-05-15'),color='c',label="Taper Tantrum", linestyle='--', lw=6,alpha=0.5)
    ax.axvline(pd.to_datetime('2014-09-15'),color='m',label="Oil Price Shock", linestyle='--', lw=6,alpha=0.5)
    ax.axvline(pd.to_datetime('2016-06-23'),color='y',label="Brexit Vote", linestyle='--', lw=6,alpha=0.5)
    ax.axvline(pd.to_datetime('2016-11-08'),color='olive',label="US Elections", linestyle='--', lw=6,alpha=0.5)
    ax.set_ylabel('Polarity Score')
    ax.legend(ncol=1,loc=1)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt ) 
    ax.xaxis.set_minor_locator(months)

    # round to nearest years
    datemin = np.datetime64(newer_m.index[0], 'Y')
    datemax = np.datetime64(newer_m.index[-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)

    ax.format_xdata = mdates.DateFormatter('%Y-%m')
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    fig.suptitle('Negative sentiment polarity scores, monthly averages\n\n(anomaly detection using Hodrick-Prescott filter and MAD test, alpha=0.5)', fontsize=20)
    #plt.title('Compound sentiment polarity with outliers, monthly', fontsize=20)
    #ax.set_xticklabels([pandas_datetime.strftime("%Y-%m") for pandas_datetime in new_m.index])
    
    fig.savefig('U:\\sentiment_neg_anomaly_monthly_means_alpha05.png')

    
def main():

    ## defining y (has to be a numpy ndarray)
    df = pd.read_pickle("U:\\blogs_w_sentiment.pickle")
    df.sort_values('Date', inplace=True)
    #print(df.tail())
    #print(df.describe)
    #print(df.dtypes)
    new = df[['Date','sentiment_negative']]
    pd.to_datetime(new['Date'])
    print(new.info())
    new.index = new['Date']
    new = new['2005-01-01':'2018-05-31']
    new_m = new.resample('M').mean()
    new_m['dates'] = new_m.index.map(lambda t: t.strftime('%Y-%m'))
    y = new_m['sentiment_negative']
    print(len(y))   
    r_idx = AnomalyDetection(y, alpha=0.01)
    #list1 = list(new_m['dates'])
    plot_save(new_m,y,r_idx)

if __name__ == "__main__":
    #test()
    main()
