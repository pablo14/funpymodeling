import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


def coord_plot(data, group_var,colormap="Dark2",fig_size=(15, 6)):
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
    # x_grp # data with the original variables

    # 2- normalizing the data min-max
    x_grp_no_tgt=x_grp.drop(group_var, axis=1)

    mm_scaler = MinMaxScaler()
    mm_scaler.fit(x_grp_no_tgt)
    x_grp_mm=mm_scaler.transform(x_grp_no_tgt)

    # 3- convert to df
    df_grp_mm=pd.DataFrame(x_grp_mm, columns=x_grp_no_tgt.columns)

    df_grp_mm[group_var]=x_grp[group_var] # variables escaladas

    # 4- plot
    # Calculate minimum and maximum x-coordinates
    min_x = float('inf')
    max_x = float('-inf')
    for i, row in df_grp_mm.iterrows():
        for j, feature in enumerate(df_grp_mm.columns[:-1]):
            x_coordinate = j - 0.02 * i
            min_x = min(min_x, x_coordinate)
            max_x = max(max_x, x_coordinate)

    plt.figure(figsize=fig_size)
    parallel_coordinates(df_grp_mm, group_var, colormap=plt.get_cmap(colormap))
    # Adding points and labels
    for i, row in df_grp_mm.iterrows():
        class_label = row[group_var].astype(int)
        for j, feature in enumerate(df_grp_mm.columns[:-1]):  # Skip the last column (group_var)
            plt.plot(j, row[feature], 'o', color='black')  # Add a point at each feature value
            if i % 2 == 0:
                plt.text(j - 0.02*i, row[feature] + 0.02, str(class_label), ha='right', va='center', 
                        fontsize=8, color='red')  # Label on the left
            else:
                plt.text(j + 0.02*i, row[feature] - 0.02, str(class_label), ha='left', va='center', 
                        fontsize=8, color='black')  # Label on the right
    plt.xlim(min_x-0.1, max_x+0.8)
    plt.xticks(rotation=90)
    plt.show()


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
    
    # Sort by score descending (high scores = positive class)
    df_desc = df.sort_values(score, ascending=False).reset_index(drop=True)
    
    # Check if we need to invert: calculate gain at 50% population
    n = len(df_desc)
    total_pos = df_desc['target_bin'].sum()
    idx_50 = int(n * 0.5)
    gain_50 = df_desc.iloc[:idx_50]['target_bin'].sum() / total_pos * 100
    
    # If gain at 50% < 50%, scores are inverted (high score = negative class)
    if gain_50 < 50:
        df = df.sort_values(score, ascending=True).reset_index(drop=True)
    else:
        df = df_desc

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