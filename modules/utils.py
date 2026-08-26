import scipy.stats as stats
from scipy.optimize import minimize
import numpy
import pandas
import math


def change_type_robust(x, type):
    try:
        return type(x)
    except:
        return x

def find_nth(string, substring, n):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)

def quantile_truncate_series(series, alpha_min=0.001, alpha_max=0.999):
    q_max = series.quantile(alpha_max)
    q_min = series.quantile(alpha_min)
    series = numpy.minimum(series, q_max)
    series = numpy.maximum(series, q_min)
    
    return series

def norm_pdf(x):
    return stats.norm.pdf(x)

def norm_cdf(x): 
    return stats.norm.cdf(x)

def norm_icdf(x): 
    return stats.norm.ppf(x)

def cumulate_quantile(df_year_var, q, axis):
    if axis==1:
        df_ = pandas.concat([df_year_var.loc[:, :i].quantile(q, axis=axis) for i in df_year_var.columns], axis=1)
        df_.columns = df_year_var.columns
    else:
        df_ = pandas.concat([df_year_var.loc[:i].quantile(q, axis=axis) for i in df_year_var.index], axis=1).T
        df_.index = df_year_var.index
    
    return df_
    
def interpolate_series(s, x, method='index'):
    if x in s.index:
        return s[x]
    
    s[x] = numpy.nan
    s = s.sort_index()

    return s.interpolate(method=method)[x]

def interpolate_dataframe(df, x, method='index'):
    if x in df.index:
        return df.sort_index()

    df_ = df.copy()
    df_.loc[x] = numpy.nan

    return df_.interpolate(method=method).sort_index()

def is_inside_circle(x, y, radius):
    """ Check if a point (x, y) is inside a circle of given radius. """
    return x**2 + y**2 <= radius**2

def mask_matrix(M, radius):
    """ Mask the matrix M to keep only the values inside the circle P. """
    N = M.shape[0]
    if N % 2 == 0:
        raise ValueError("N must be odd.")
    
    center = N // 2
    
    for i in range(N):
        for j in range(N):
            # Calculate the coordinates relative to the center
            x = (j - center)
            y = (center - i)  # Invert y-axis for proper coordinate system
            
            if not is_inside_circle(x, y, radius):
                M[i, j] = 0
    
    return M

def degrees_to_km(lon_dist, lat_dist, lat):
    """
    Convert distance from degrees to kilometers for a point at a specific latitude.
    
    Parameters:
    - lon_dist: distance in degrees of longitude
    - lat_dist: distance in degrees of latitude
    - lat: latitude in degrees at which the conversion is made
    
    Returns:
    - (lon_km, lat_km): tuple of distances in kilometers for longitude and latitude
    """
    # Constant for degrees to kilometers conversion
    km_per_lat_degree = 111.32  # km per degree of latitude
    km_per_lon_degree = 111.32 * math.cos(math.radians(lat))  # km per degree of longitude at the given latitude
    
    lon_km = lon_dist * km_per_lon_degree
    lat_km = lat_dist * km_per_lat_degree
    
    return numpy.sqrt(lon_km**2 + lat_km**2)
    
def pd_to_rating(pd_sim, pd_scale, default_plot, n_levels=2):

    """
    Take a pandas.Series of PDs and computes the notch (rating number).
    """

    pd_threshold = pd_scale
    pd_to_rating = lambda p: (pd_threshold <= p).sum()

    df_notch = pandas.concat([pd_sim.xs(i, level=n_levels-1, drop_level=False).apply(pd_to_rating) for i in tm_factorial_function.plots_col[:-1]]).groupby(level=list(range(n_levels))).mean()
    df_notch = df_notch.rename(index=dict_ratings_notch, level=n_levels-1)

    return df_notch.sort_index()

def denotch(pd_sim, pd_scale, default_plot):

    """
    Take a pandas.Series of PDs and compute the denotching. 
    """

    n_levels = len(pd_sim.index.names)
    df_notch = pd_to_rating(pd_sim, pd_scale, default_plot, n_levels)

    df_starting_notch = df_notch.index.get_level_values(n_levels-1)
    df_denotch = df_notch - df_starting_notch
    df_denotch = df_denotch.sort_index()
    df_denotch = df_denotch.rename(index=dict_notch_ratings)

    df_denotch = -df_denotch.unstack()[dict_ratings_notch.keys()].applymap(lambda x: max(x, 0))
    score_card = (df_denotch - df_denotch.quantile(0.5)).round(0)

    return df_denotch

def cumulate_tr(tr):
    tr_cumulated = tr.copy()

    for i, (i_1, i_2) in enumerate(zip(tr.index[:-1], tr.index[1:])):
        tr_1 = tr[i_1].copy()
        tr_1 = numpy.vstack([tr_1, numpy.eye(tr_1.shape[1])[-1]])
        tr_2 = tr[i_2].copy()
        tr_2 = numpy.vstack([tr_2, numpy.eye(tr_2.shape[1])[-1]])
        tr_cumulated.loc[i_2] = tr_1.dot(tr_2)[:tr[i_1].shape[0], :tr[i_1].shape[1]]

    return tr_cumulated.apply(lambda x: x)

