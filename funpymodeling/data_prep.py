import pandas as pd
import numpy as np

def todf(data):
    """
    Convert almost any object to a pandas DataFrame

    Parameters
    ----------
    data: data to be converted

    Returns
    -------
    A pandas DataFrame

    Raises
    ------
    ValueError
        If the object to be converted has more than 2 dimensions
    TypeError
        If data is None

    """

    if data is None:
        raise TypeError("'data' parameter cannot be None")

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = pd.DataFrame({'var': data})
        elif data.ndim == 2:
            data = pd.DataFrame(data)
        else:
            raise ValueError(
                "The object to be converted has more than 2 dimensions"
            )

    elif isinstance(data, list):
        data = pd.DataFrame(data)

    else:
        data = pd.DataFrame({'var': data}).convert_dtypes()

    return data