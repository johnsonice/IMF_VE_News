from crisis_points import crisis_points
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import frequency_utils as fu




def crisis_plot(data, country=None, roll_avg=True, roll_window=20):
    """
    For any pandas time series, plot it and add annotations for crisis points given the country.
    :param data: pd.Series with PeriodIndex
    :param country: str
    :return: plt
    """
    assert isinstance(data, pd.Series)
    country = country.lower() if isinstance(country, str) else country
    plt.figure()
    if roll_avg:
        plot_data = data.rolling(window=roll_window).mean().values
    else:
        plot_data = data.values
    print('##1##')
    plt.plot(plot_data)
    plt_indices = [i for i in range(len(data)) if i % 2 == 0]
    plt_labs = list(data.index[plt_indices])
    print('##2##')
    plt.xticks(plt_indices, plt_labs, rotation=90, fontsize=6)
    plt.legend()
    crisis_starts = crisis_points[country]['starts']
    crisis_peaks = crisis_points[country]['peaks']
    print('##3##')
    for s, p in zip(crisis_starts, crisis_peaks):
        if s in data.index and p in data.index:
            s_index = data.index.get_loc(s)
            p_index = data.index.get_loc(p)
            plt.axvline(x=s_index, color='grey', linestyle="--", linewidth=2)
            plt.axvline(x=p_index, color='red', linestyle="--", linewidth=2)
            plt.axvspan(s_index, p_index, facecolor='r', alpha=0.1)

    print('##4##')
    return plt.gcf()


def plot_frequency(data, words=(), roll_avg = True, roll_window=20, country=None, aggregate=True,
                   z_score=False, slope=False, crisis_defs='kr', anomalies=True):
    """
    plot frequency of supplied words given supplied freq dataframe,
    including annotation for country crisis points.
    :param data: pd.DataFrame of word frequency data (words as index, PeriodIndex as columns
    :param words: lst of words for which to plot frequency
    :param country: str country name for which crisis points to plot
    :return:
    """
    assert type(data) == pd.DataFrame
    assert not all((z_score, slope)) # can't do both
    plt.figure()
    num_colors = len(words)
    cm = ListedColormap(sns.color_palette("Paired", num_colors).as_hex())
    ax = plt.subplot(111)
    ax.set_prop_cycle('color',[cm(1. * i / num_colors) for i in range(num_colors)])

    if not aggregate:
        word_freqs = [(word, data.loc[word]) for word in words if word in data.index and sum(data.loc[word] != 0)]
    else:
        word_freqs = [data.loc[word] for word in words if
                      word in data.index and sum(data.loc[word] != 0)]
        grp_freq = sum(word_freqs)
        word_freqs = [('aggregate: {}'.format(list(words)), grp_freq)]

    for i, (word, vals) in enumerate(word_freqs):
        try:
            if i == 13:
                cm = ListedColormap(sns.color_palette("bright", num_colors).as_hex())
                ax.set_color_cycle('color',[cm(1. * i / num_colors) for i in range(num_colors)])
            if roll_avg:
                vals = vals.ewm(span=roll_window).mean()
                if z_score:
                    scores = fu.rolling_z_score(vals, window=roll_window)
                    plot_data = scores.values
                elif slope:
                    slopes = fu.rolling_slope(vals, window=roll_window)
                    plot_data = slopes.values
                else:
                    # plot_data = vals.rolling(window=roll_window).mean().values
                    plot_data = vals.values
            else:
                plot_data = vals.values
            if aggregate:
                #ax.plot(vals.values, label='raw values', color='lightblue', linestyle='--')
                ax.plot(plot_data, label='EWMA: ' + word, color='darkblue')
            else:
                ax.plot(plot_data, label=word)
            if anomalies:
                preds = list(fu.signif_change(vals, window=roll_window, direction='incr').index)
                for pred in preds:
                    x = list(vals.index).index(pred)
                    y = vals[pred]
                    ax.plot(x, y, 'ro')
        except KeyError:
            pass

    # Stylize
    axis_indices = [i for i in range(len(data.columns)) if i % 2 == 0]
    axis_labs = list(data.columns[axis_indices])
    ax.set_xticks(axis_indices)
    ax.set_xticklabels(axis_labs, rotation=90, fontsize=12)
    ax.legend(loc=2)
    if z_score:
        ax.set_ylabel('Z Score')
        plt.axhline(y=1.96, color='k', linewidth=1.5)
        plt.axhline(y=-1.96, color='k', linewidth=1.5)
    elif slope:
        ax.set_ylabel('rolling slope (window = {}'.format(roll_window))
        plt.axhline(y=0, color='red', linewidth=1.5)
    else:
        ax.set_ylabel('Freq Per Thousand Words')

    if country:
        if crisis_defs == 'fund':
            crises = pd.read_csv('../data/crises.csv')
            country_crises = crises[crises['country_name'] == country]['years']
            crisis_starts = ['{}-01'.format(year) for year in set(country_crises['years'])]
            crisis_ends = ['{}-01'.format(int(year)+1) for year in set(country_crises['years'])]
        elif crisis_defs == 'kr':
            crisis_starts = crisis_points[country]['starts']
            crisis_ends = crisis_points[country]['peaks']

        for s, e in zip(crisis_starts, crisis_ends):
            try:
                s_index = data.T.index.get_loc(s)
                plt.axvline(x=s_index, color='grey', linestyle="--", linewidth=2)
                e_index = data.T.index.get_loc(e)
                plt.axvline(x=e_index, color='red', linestyle="--", linewidth=2)
                plt.axvspan(s_index, e_index, facecolor='r', alpha=0.1)
            except:
                continue
    return plt.gcf()
