## define modules
# import pandas as pd
# import numpy as np
# import streamlit as st

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder

# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


## define constants
FIRST_ITEM = 0
LAST_ITEM = -1
ONE_STEP = 1
TARGET_COLUMN = 'PM2.5'


# to encode wind feature
WIND_DIRECTION = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE',
    'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW',
    'NW', 'NNW', 'N'
]

DEGREE_VALUE = [
    0, 22.5, 45, 67.5, 90, 112.5, 135,
    157.5, 180, 202.5, 225, 247.5, 270, 292.5,
    315, 337.5, 360
]

WD_ENCODE_VALUE = {wd : deg_val for wd, deg_val in zip(WIND_DIRECTION, DEGREE_VALUE)}

stations_name = st.session_state.locations_name
STATION_CODE = {name : n_id + 1 for n_id, name in enumerate(stations_name)}


## define functions
def styled_header(text, emoji, divider_color='grey', level='header', additional_information=False):
    """Creates a styled subheader with an emoji."""

    if level == 'header':
        st.header(f"{emoji} {text}", divider=divider_color)

    if level == 'subheader':
        st.subheader(f"{emoji} {text}", divider=divider_color)

    if additional_information:
        st.write(additional_information)


def highlight_encoded_and_na(df, cols_to_highlight, precision=1):
    """
    Applies highlighting to a DataFrame for:
    1. Specified columns and their headers (e.g., encoded columns).
    2. All NaN (missing) values throughout the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to style.
    cols_to_highlight : list
        A list of column names to be highlighted for encoding.

    Returns:
    --------
    pd.io.formats.style.Styler
        A Styler object with all highlighting rules applied.
    """
    # Make a copy to avoid modifying the original DataFrame's style
    df_styled = df.copy()

    # --- Define Styles ---
    # Style for encoded columns (light green)
    encoded_cell_style = 'background-color: #E8F5E9;'
    encoded_header_style_props = [('background-color', '#C8E6C9')]
    
    # Style for NaN values (light red)
    nan_style_color = '#FFCDD2'

    # Define the format string based on the desired precision
    format_string = f"{{:.{precision}f}}"

    # --- Prepare Styling Rules ---
    # Rule for encoded column headers
    header_rules = [
        {'selector': f'th.col_heading.col{df.columns.get_loc(col) + 1}', 
         'props': encoded_header_style_props}
        for col in cols_to_highlight if col in df.columns
    ]

    # --- Apply All Styles Sequentially ---
    styler = df_styled.style.set_properties(
        subset=cols_to_highlight,
        **{'background-color': '#E8F5E9'}  # 1. Highlight encoded data cells
    ).set_table_styles(
        header_rules  # 2. Highlight encoded header cells
    ).highlight_null(
        color=nan_style_color,  # 3. Highlight all NaN values
        subset=None # Apply to the whole DataFrame
    ).format(
        format_string,
        na_rep="-"  # Optional: representation for NaN values
    )
    
    return styler


def introduction_section():
    """"
    """
    
    styled_header("Let's prepare our data!", '👇', divider_color="grey", level='header')
    st.markdown("Preprocessing is a technique for cleaning and obtaining important information to improve the performance of machine learning models.")
    st.page_link("https://doi.org/10.1186/s40537-021-00548-1", label="A. Bekkar et al. 2021", icon=":material/open_in_new:")
    st.markdown("   ")


### Label encoding
def encode_wd(data):
    """Convert the wind direction values to degree values in multiples of 22.5 each time.

    Parameters
    ----------
        data: the wind direction column

    Returns
    -------
        an array of wind direction degree value: list
    """

    st.write("🎯 Encode all wind direction values to degree values in multiples of 22.5 each time")

    return data.copy().map(WD_ENCODE_VALUE)


def encode_station(data):
    """Convert the station name to number.
    The number start from 1 to 12 based the `STATION_NAMES` variable in order.

    Parameters
    ----------
        data: the station column that contain station name

    Returns
    -------
        an array of number: list
    """

    st.write("🎯 Encode all stations name")

    return data.copy().map(STATION_CODE)


### Missing values
def get_last_idx_after_nan(start, array):
    """Returns the last index after (continoues/single) missing value(s) from start index.

    Parameters
    ----------
        start: The initial index.
        array: The indexes of given column.

    Returns
    -------
        an index after the last NaN index: integer
    """

    idx_after_start = start + ONE_STEP

    if idx_after_start in array:
        end_idx = get_last_idx_after_nan(idx_after_start, array)
    else:
        end_idx = idx_after_start

    return end_idx


def get_idx_between_nan(start, array):
    """Returns a list of start and last index between (continuous/single)
    missing value(s).

    Parameters
    ----------
        start: The initial index.
        array: The indexes of given column.

    Returns
    -------
        array of the start and the last index between missing value(s): list

    """

    idx_before_start = start - ONE_STEP

    return [idx_before_start, get_last_idx_after_nan(start, array)]


def get_linear_imputation_idx(df, col, mid_idx):
    """Returns the before, the middle, & the after index based given (middle) index.

    Parameters
    ----------
        df: The indexes of given column.
        col: The column name
        mid_idx: The given (middle) index

    Returns
    -------
        array of The start, the middle, & the last index based the given index: list

    Examples
    --------
    >>> idx   |   value
        ---------------
        4669  |  1800.0
        4670  |     NaN
        4671  |  1500.0
    >>> idx_before, idx, idx_current = [4669, 4670, 4671]
    """

    idx_after_mid_idx = mid_idx + ONE_STEP
    idx_before_mid_idx = mid_idx - ONE_STEP

    first_imputation_idx = df.loc[
                        idx_before_mid_idx:idx_before_mid_idx,col
                    ].index.values.__getitem__(FIRST_ITEM)

    mid_imputation_idx = df.loc[
                        mid_idx:mid_idx,
                    ].index.values.__getitem__(FIRST_ITEM)

    last_imputation_idx = df.loc[
                        idx_after_mid_idx:idx_after_mid_idx,col
                    ].index.values.__getitem__(FIRST_ITEM)

    return [first_imputation_idx, mid_imputation_idx, last_imputation_idx]


def get_linear_imputation_item(df, col, idx_before, idx, idx_current):
    """Returns the before, the middle, and the after item between based
    the given (middle) index.

    Parameters
    ----------
        df: The indexes of given column.
        col: The column name
        idx_before: index before spline item
        idx: middle item index.
        idx_current: current given index

    Returns
    -------
        array of start, the middle, and the last item based the given index: list

    Examples
    --------
    >>> idx   |   value
        ---------------
        4669  |  1800.0
        4670  |     NaN
        4671  |  1500.0
    >>> idx_before, idx, idx_current = [1800, NaN, 1500]
    """

    val_before = df.loc[idx_before:idx_before, col].values.__getitem__(FIRST_ITEM)
    val_current = df.loc[idx_current:idx_current, col].values.__getitem__(FIRST_ITEM)

    return [val_before, val_current]


def linear_imputation(item_before, item_current, idx_before, idx, idx_current):
    """Returns the imputed value between item(s) based the given index(s).

    Parameters
    ----------
        item_before: an item before a missing value
        item_current: an item after a missing value
        idx_before: an item index before a missing value
        idx: the missing value index
        idx_current: an item index after a missing value

    Returns
    -------
        imputed value: int

    Sources
    --------
        Linear Imputation (Linear): https://doi.org/10.3390/rs13245018
        Linear Spline Interpolation: https://doi.org/10.1186/s40537-021-00548-1
    """

    left = item_before * ((idx - idx_current) / (idx_before - idx_current))
    right = item_current * ((idx - idx_before) / (idx_current - idx_before))

    return left + right


def interpolate_missing_value(df, col):
    """Returns array of imputed values between the given index.

    Parameters
    ----------
        df: dataframe
        col: a dataframe column

    Returns
    -------
        imputed values: numpy array
    """

    # get all index of NaN value(s)
    nan_idxs = df.loc[pd.isna(df[col]), :].index.to_list()

    # get all x(i - 1), x, x(i) as the indexes
    imputation_idxs = []
    continues_nan_idx = []
    start = None
    end = None


    # get before and after index from a NaN index(s)
    for idx in nan_idxs:
        one_step_after_idx = idx + ONE_STEP

        if one_step_after_idx in nan_idxs:
            if start == None:
                start = idx - ONE_STEP

            end = get_idx_between_nan(idx, nan_idxs).__getitem__(FIRST_ITEM + ONE_STEP)
            continues_nan_idx = [start, end]
            imputation_idxs.append([start, idx, end])

        if one_step_after_idx == end:
            imputation_idxs.append([start, idx, one_step_after_idx])

        if one_step_after_idx not in nan_idxs:
            start, end = None, None
            continues_nan_idx.clear()
            imputation_idxs.append(get_linear_imputation_idx(df, col, idx))


    # get all f(x(i - 1)), f(x(i))
    # get before and after item(s) from a NaN item(s)
    imputation_items = []
    for idx in imputation_idxs:

        idx_before, idx_x, idx_current = idx
        imputation_items.append(
            get_linear_imputation_item(df, col, idx_before, idx, idx_current)
        )


    # fill missing value(s)
    all_interpolated_val = []

    for vals, idxs in zip(imputation_items, imputation_idxs):

        row_dimention = 2
        empty = 0

        arr = np.zeros(shape=(row_dimention, ), dtype='object_')
        item_before, item_current = vals
        idx_before, idx, idx_current = idxs

        interpolate_val = linear_imputation(
                                item_before, item_current, idx_before, idx, idx_current
                            )
        empty_interpolated_val = np.isnan(interpolate_val)
        if not empty_interpolated_val:
            arr[FIRST_ITEM] = interpolate_val
            arr[LAST_ITEM] = int(idx)

        n_total_array = np.sum(arr)
        if n_total_array != empty:
            all_interpolated_val.append(arr)

    return np.array(all_interpolated_val)


def fill_missing_value(df, interpolated_val):
    df_with_no_nans = df.copy()

    for imputation, column_name in zip(
                            interpolated_val,  df_with_no_nans.columns[:LAST_ITEM]
    ):
        one_dimention = 1
        if imputation[FIRST_ITEM].shape[FIRST_ITEM] == one_dimention:
            value = imputation[FIRST_ITEM][FIRST_ITEM][FIRST_ITEM]
            idx = imputation[FIRST_ITEM][FIRST_ITEM][LAST_ITEM]
            df_with_no_nans.loc[idx:idx, column_name] = value

        if imputation[FIRST_ITEM].shape[FIRST_ITEM] > one_dimention:
            for imp in imputation[FIRST_ITEM]:
                value = imp[FIRST_ITEM]
                idx = imp[LAST_ITEM]
                df_with_no_nans.loc[idx:idx, column_name] = value

    st.write("🎯 Fill all any missing values (highlighted by red color from **Step #1**)")

    return  df_with_no_nans


### Min-Max Normalisation
def get_max_column_value(df, col):
    """Returns the highest value from a column

    Parameters
    ----------
        df: dataframe
        col: a dataframe column

    Returns
    -------
    highest values: number
    """

    return np.max(df.loc[FIRST_ITEM:, col])


def get_min_column_value(df, col):
    """Returns the lowest value from a column

    Parameters
    ----------
        df: dataframe
        col: a dataframe column

    Returns
    -------
    lowest values: number
    """

    return np.min(df.loc[FIRST_ITEM:, col])


def calc_min_max_normalization(val, min_max_val):
    """Returns the normalized value from a given value with their min-max value

    Parameters
    ----------
        val: the value that will be normalize
        min_max_val: an array contains min and max value from a column

    Returns
    -------
    normalized value: number
    """

    min_val, max_val = min_max_val
    delta_min_max = max_val - min_val

    denominator_is_zero = (max_val - min_val) == 0

    if denominator_is_zero:
        return 0

    return (val - min_val) / (max_val - min_val)


def normalize_with_min_max(df):
    """Returns the normalized value from a given dataframe

    Parameters
    ----------
        df: dataframe that will be normalize

    Returns
    -------
    normalized data: dataframe
    """

    # create container to store normalized values
    container = pd.DataFrame(np.zeros(shape=df.shape), columns=df.columns, index=df.index)
    min_max_value_per_column = []

    # get the highest and th lowest value from every columns
    for column in df.columns:
        min_max_value_per_column.append([
            get_min_column_value(df, column),
            get_max_column_value(df, column),
            column
        ])

    # normalize given data
    for min_val, max_val, col in min_max_value_per_column:
        for idx, val in zip(df.loc[:, col].keys(), df.loc[:, col].values):
            container.loc[idx:idx, col] = calc_min_max_normalization(val, [min_val, max_val])

    st.write("🎯 The **lowest value is highlighted in red color** and the **highest values is highlighted in green color**")

    return container


def highlight_min_max(df, precision=1):
    """
    Highlights the minimum and maximum values in each column of a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to style.

    Returns:
    --------
    pd.io.formats.style.Styler
        A Styler object with the min/max highlighting applied.
    """
    # Define colors for min and max
    min_color = '#FFCDD2'  # Light Red
    max_color = '#C8E6C9'  # Light Green

    # Define the format string based on the desired precision
    format_string = f"{{:.{precision}f}}"

    # The styler functions automatically ignore non-numeric columns
    styler = df.style.highlight_max(
        axis=0,  # axis=0 operates column-wise
        color=max_color
    ).highlight_min(
        axis=0,  # axis=0 operates column-wise
        color=min_color
    ).format(
        format_string,
        na_rep="-"  # Optional: representation for NaN values
    )
    
    return styler


## display contents
introduction_section()

df_for_preprocessing = st.session_state.main_dataframe.copy(deep=True).iloc[4632:4680]

st.warning("For performance and examples, only 48 hours of data [ 4632 : 4680 ] were used in preprocessing.", icon=":material/warning:")
st.info("For full preprocessed data, see my GitHub", icon=":material/info:")
st.markdown("   ")

st.write(f'🎯 Selected location data in: **{st.session_state.data_location}**, here is the dataframe preview!')
st.dataframe(df_for_preprocessing)


### Label Encoding
step_1_preprocessing_explanation = "Change all non-numeric values such as 'dog, cat, bird' to labels with value '1' representing 'animal'"

styled_header("Step #1: Label Encoding", "👇", divider_color='grey', level='subheader', additional_information=step_1_preprocessing_explanation)

df_for_preprocessing.loc[:,'wd'] = encode_wd(df_for_preprocessing['wd'])
df_for_preprocessing.loc[:,'station'] = encode_station(df_for_preprocessing['station'])
after_encoded = highlight_encoded_and_na(df_for_preprocessing, ['wd', 'station'])

st.dataframe(after_encoded)
st.write("❗ The cell with missing values highlighted by red color ❗")


### Fill Missing Values
step_2_preprocessing_explanation = "Missing values are caused by entered values being lost and forgotten."
styled_header("Step #2: Fill Missing Values", "👇", divider_color='grey', level='subheader', additional_information=step_2_preprocessing_explanation)

# df_has_null = df_for_preprocessing.style.highlight_null(color='yellow')  
interpolated_val = [[interpolate_missing_value(df_for_preprocessing, col)] for col in df_for_preprocessing.columns[4:-1]]
df_for_preprocessing = fill_missing_value(df_for_preprocessing.iloc[:, 4:-1], interpolated_val)

st.dataframe(df_for_preprocessing)


### Min-Max Normalization
step_3_preprocessing_explanation = "The purpose of normalization is to ensure that all values have the same range, because some features have larger range numbers (1000 - 10.000) and others only have small range numbers (0.1 - 10)."
styled_header("Step #3: Min-Max Scaling", "👇", divider_color='grey', level='subheader', additional_information=step_3_preprocessing_explanation)

normalized_df_for_preprocessing = normalize_with_min_max(df_for_preprocessing)
normalized_df_for_preprocessing = highlight_min_max(normalized_df_for_preprocessing)

st.dataframe(normalized_df_for_preprocessing)
