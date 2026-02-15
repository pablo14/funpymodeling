import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


def coord_plot(data, group_var):
    """
    Coordinate plot analysis for clustering models. Also returns the original and the normalized (min-max) variable table. Useful to extract the main features for each cluster according to the variable means.
    Parameters:
    -----------
    data : Pandas DataFrame containing the variables to analyze the mean across each cluster
    group_var : String indicating the clustering variable name
    Returns:
    --------
    A tuple containing two data frames. The first contains the mean for each category across each value of the group_var. The other data set is      similar but it is min-max normalized, range [0-1].
    It also shows the coordinate or parallel plot.
    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    # If species is the cluster variable:
    >> coord_plot(iris, 'species')
    """
    # 1- group by cluster, get the means
    x_grp=data.groupby(group_var).mean()
    x_grp[group_var] = x_grp.index 
    x_grp=x_grp.reset_index(drop=True)
    x_grp # data with the original variables

    # 2- normalizing the data min-max
    x_grp_no_tgt=x_grp.drop(group_var, axis=1)

    mm_scaler = MinMaxScaler()
    mm_scaler.fit(x_grp_no_tgt)
    x_grp_mm=mm_scaler.transform(x_grp_no_tgt)

    # 3- convert to df
    df_grp_mm=pd.DataFrame(x_grp_mm, columns=x_grp_no_tgt.columns)

    df_grp_mm[group_var]=x_grp[group_var] # variables escaladas

    # 4- plot
    parallel_coordinates(df_grp_mm, group_var, colormap=plt.get_cmap("Dark2"))
    plt.xticks(rotation=90)

    return [x_grp, df_grp_mm]


# ---------------------------------------------------------------
# gain_lift
# ---------------------------------------------------------------
def gain_lift(data, score, target, q_segments=10):
    """
    Generates lift and cumulative gain performance table and plot.
    Higher values at the beginning of the population implies a better model.
    Equivalent to funModeling::gain_lift in R.

    Parameters:
    -----------
    data: data frame
    score: string, column name containing the score/probability
    target: string, binary target variable name
    q_segments: number of segments (5, 10 or 20), default 10

    Returns:
    --------
    DataFrame with columns: Population, Gain, Lift, Score.Point.
    Also displays cumulative gain and lift charts.

    Example:
    --------
    >> gain_lift(data=heart_disease, score='score', target='has_heart_disease')
    """
    df = data[[score, target]].dropna().copy()
    pos_class = df[target].value_counts().idxmin()
    df['target_bin'] = (df[target] == pos_class).astype(int)
    df = df.sort_values(score, ascending=False).reset_index(drop=True)
    n, total_pos = len(df), df['target_bin'].sum()

    results = []
    for i in range(1, q_segments + 1):
        idx = int(n * i / q_segments)
        cum = df.iloc[:idx]['target_bin'].sum()
        gain = round(cum / total_pos * 100, 2)
        results.append({
            'Population': round(i * 100 / q_segments, 1),
            'Gain': gain,
            'Lift': round(gain / 100 / (i / q_segments), 2),
            'Score.Point': round(df.iloc[idx - 1][score], 7)
        })

    results_df = pd.DataFrame(results)

    # Plot
    pop = [0] + list(results_df['Population'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    ax1.plot(pop, [0] + list(results_df['Gain']), 'b-o', lw=2, ms=4)
    ax1.plot([0, 100], [0, 100], 'k--')
    ax1.set(xlabel='Population (%)', ylabel='Gain (%)',
            title='Cumulative Gain Chart', xlim=(0, 100), ylim=(0, 105))

    ax2.plot(results_df['Population'], results_df['Lift'], 'r-o', lw=2, ms=4)
    ax2.axhline(y=1, color='k', ls='--')
    ax2.set(xlabel='Population (%)', ylabel='Lift',
            title='Lift Chart', xlim=(0, 105))

    plt.tight_layout()
    plt.show()

    return results_df