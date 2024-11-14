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