from .data_prep import (
    todf,
    equal_freq,
    tukey_outlier,
    hampel_outlier,
    prep_outliers,
    categ_analysis,
    auto_grouping,
    discretize_get_bins,
    discretize_df,
)
from .exploratory import (
    status,
    corr_pair,
    num_vars,
    cat_vars,
    profiling_num,
    freq_tbl,
    freq_plot,
    plot_num,
)
from .target_analysis import (
    cross_plot,
    var_rank_info,
)
from .model_validation import (
    coord_plot,
    gain_lift,
)


__version__ = "0.2.0"
