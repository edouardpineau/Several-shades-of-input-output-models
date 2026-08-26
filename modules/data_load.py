import pandas
import numpy
import os

def load_leontief(path, year, sector):
    
    """
    Inputs:
        - year (int): of the data
        - sector (str): sector classification used of the data ('nace', 'nace_1', 'gics')
    
    Outputs:
        - A_gs2gs (pandas.DataFrame): input-output intensity matrix geo-sector to geo-sector (gs2gs)
        - B_gs2gs (pandas.DataFrame): Leontief inverse of A_gs2gs
        - dict_financials (dict): dictionary containing the geo-sectoral gross value-added, production output, production import, final demand, GHG
    """
    
    try:
        df_xyv = pandas.read_csv(os.path.join(path, "df_xyv_cs_{}_{}.csv".format(sector, year)), index_col=[0, 1])
    except:
        df_xyv = pandas.read_parquet(os.path.join(path, "df_xyv_cs_{}_{}.pqt".format(sector, year)))
        
    df_gva_gs = df_xyv["gva"]
    df_gva_gs.replace(0.0, numpy.nan, inplace=True)
    df_gva_gs = df_gva_gs.apply(lambda x: x * 1e6)
    
    df_output_gs = df_xyv["output"]
    df_output_gs.replace(0.0, numpy.nan, inplace=True)
    df_output_gs = df_output_gs.apply(lambda x: x * 1e6)
    
    df_demand_gs = df_xyv["demand"]
    df_demand_gs.replace(0.0, numpy.nan, inplace=True)
    df_demand_gs = df_demand_gs.apply(lambda x: x * 1e6)
    
    A_gs2gs = pandas.read_parquet(os.path.join(path, 'A_cs2cs_{}_{}.pqt'.format(sector, year)))
    L_gs2gs = pandas.read_parquet(os.path.join(path, 'L_cs2cs_{}_{}.pqt'.format(sector, year)))
        
    df_ghg_gs = pandas.read_csv(os.path.join(path, 'df_ghg_cs_{}_{}.csv'.format(sector, year)), index_col=[0, 1]).squeeze()
    df_ghg_gs.replace(0.0, numpy.nan, inplace=True)
    df_ghg_gs = df_ghg_gs.apply(lambda x: x * 1e6)
    
    dict_financials = {'gva': df_gva_gs, 'output': df_output_gs, 'demand': df_demand_gs, 'ghg': df_ghg_gs}
    
    return A_gs2gs, L_gs2gs, dict_financials


def load_ghosh(path, year, sector):
    
    """
    Inputs:
        - year (int): of the data
        - sector (str): sector classification used of the data ('nace', 'nace_1', 'gics')
    
    Outputs:
        - A_gs2gs (pandas.DataFrame): input-output intensity matrix geo-sector to geo-sector (gs2gs)
        - B_gs2gs (pandas.DataFrame): Leontief inverse of A_gs2gs
        - dict_financials (dict): dictionary containing the geo-sectoral gross value-added, production output, production import, final demand, GHG
    """
    
    df_xyv = pandas.read_csv(os.path.join(path, "df_xyv_cs_{}_{}.csv".format(sector, year)), index_col=[0, 1])
    
    df_gva_gs = df_xyv["gva"]
    df_gva_gs.replace(0.0, numpy.nan, inplace=True)
    df_gva_gs = df_gva_gs.apply(lambda x: x * 1e6)
    
    df_output_gs = df_xyv["output"]
    df_output_gs.replace(0.0, numpy.nan, inplace=True)
    df_output_gs = df_output_gs.apply(lambda x: x * 1e6)
    
    df_demand_gs = df_xyv["demand"]
    df_demand_gs.replace(0.0, numpy.nan, inplace=True)
    df_demand_gs = df_demand_gs.apply(lambda x: x * 1e6)
    
    B_gs2gs = pandas.read_parquet(os.path.join(path, 'B_cs2cs_{}_{}.pqt'.format(sector, year)))
    G_gs2gs = pandas.read_parquet(os.path.join(path, 'G_cs2cs_{}_{}.pqt'.format(sector, year)))
        
    df_ghg_gs = pandas.read_csv(os.path.join(path, 'df_ghg_cs_{}_{}.csv'.format(sector, year)), index_col=[0, 1]).squeeze()
    df_ghg_gs.replace(0.0, numpy.nan, inplace=True)
    df_ghg_gs = df_ghg_gs.apply(lambda x: x * 1e6)
    
    dict_financials = {'gva': df_gva_gs, 'output': df_output_gs, 'demand': df_demand_gs, 'ghg': df_ghg_gs}
    
    return B_gs2gs, G_gs2gs, dict_financials


def compute_tms(path, country, ratings, default_plot, cra):
    dict_tms = pandas.read_excel(os.path.join(path, '{}_{}.xlsx'.format(country, cra)), sheet_name=None, index_col=0)
    dict_tms.pop('Paramaters');

    dict_ratings = {'Aaa':'AAA', 'Aa':'AA', 'Baa':'BBB', 'Ba':'BB', 'Caa':'CCC', 'Ca':'CC'}

    keys_ = list(dict_tms.keys())

    for k in keys_:
        dict_tms[int(k)] = dict_tms.pop(k)
        dict_tms[int(k)] = dict_tms[int(k)].rename(columns=dict_ratings, index=dict_ratings)
        dict_tms[int(k)][default_plot] = dict_tms[int(k)]['Number of defaults']
        dict_tms[int(k)]['effectifs_totaux'] = dict_tms[int(k)][ratings + [default_plot]].sum(1)
        dict_tms[int(k)]['effectifs_defaut'] = dict_tms[int(k)][default_plot]

    df_tms = pandas.concat(dict_tms)
    df_tms = df_tms.loc[(slice(None), ratings), ratings + [default_plot, 'effectifs_totaux', 'effectifs_defaut']]

    return df_tms.loc[df_tms.index.get_level_values(0).unique()]
