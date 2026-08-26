import pandas, numpy
import os

from data_load import *
from leontief_model import *

class cost_earnings_VaR:
    def __init__(self, pathways, sector_name='NACE_FIGARO', source_mrio='FIGARO', start_date=2024):
        self.sector_name = sector_name
        self.source_mrio = source_mrio
        self.data_VaR = {}
        self.model_VaR = {}
        
        self.pathways = pathways
        self.start_date = start_date
                
        self.list_carbon_prices = [10, 25, 50, 100, 250]
        self.scenario_based = False
        self.messages, self.dictionaries = {}, {}
        
    def load_static_data(self, constant_ptr=None, constant_elasticity=False, year_mrio=2022, exclude_sectors=[], precision=None):

        self.dictionaries["sectors"] = pandas.read_excel(os.path.join(self.pathways["GENERAL_DATA_PATH"], 'dictionary_sectors.xlsx'))
        self.dictionaries["regions"] = pandas.read_excel(os.path.join(self.pathways["GENERAL_DATA_PATH"], 'dictionary_regions.xlsx'))
        
        # Load MRIO data
        
        if year_mrio is None:
            year_mrio = max([int(f.replace(".pqt", "")[-4:]) for f in os.listdir(self.pathways["MRIO_PATH"]) if ".pqt" in f and self.sector_name in f])
            
        A_gs2gs, L_gs2gs, dict_financials_gs = load_leontief(self.pathways["MRIO_PATH"], year_mrio, self.sector_name)
        A_gs2gs = A_gs2gs.drop(exclude_sectors, level=1, axis=0).drop(exclude_sectors, level=1, axis=1)
        L_gs2gs = L_gs2gs.drop(exclude_sectors, level=1, axis=0).drop(exclude_sectors, level=1, axis=1)
        
        for k in dict_financials_gs:
            dict_financials_gs[k] = dict_financials_gs[k].drop(exclude_sectors, level=1)
        
        for k in dict_financials_gs:
            dict_financials_gs[k] = dict_financials_gs[k] * (dict_financials_gs["output"] > 0.0).astype(float)
        
        if constant_ptr is not None:
            pass_through = pandas.Series(index=A_gs2gs.index).fillna(constant_ptr).apply(lambda x: min(x, 1.0))
        else:
            pass_through = pandas.Series(index=A_gs2gs.index).fillna(0.0)
            dict_nace_ptr = self.dictionaries["sectors"].set_index([self.sector_name])['PTR'].groupby(level=0).mean()
            
            try:
                dict_nace_ptr = dict_nace_ptr.drop(exclude_sectors + ['Unknown'])
            except:
                self.messages['drop unknown'] = "Already dropped"
            
            for c in pass_through.index.get_level_values(0).unique():
                pass_through[c] = dict_nace_ptr
            
        if constant_elasticity:
            vc_elasticity = -1
        else:
            vc_elasticity = 1 - 1 / numpy.clip(pass_through, 0.00001, 0.99999)
        
        # Leontief model initialization
        
        leontief = Leontief(dict_financials_gs, A=A_gs2gs, L=None, pass_through=pass_through, precision=precision)
        
        # Create a carbone price shock. If function load_scenario_data is called, then these default prices and GHG intensities will be replaced. 
        
        self.data_VaR['df_ghg_gs'] = pandas.DataFrame(1, index=[self.start_date], columns=leontief.dict_financials_gs['gva'].index)
        self.data_VaR['df_carbon_price_gs'] = pandas.DataFrame(0, index=[self.start_date], columns=leontief.dict_financials_gs['gva'].index)
        
        for i in range(1, len(self.list_carbon_prices) + 1):
            self.data_VaR['df_carbon_price_gs'].loc[self.start_date + i, :] = self.list_carbon_prices[i-1]
            self.data_VaR['df_ghg_gs'].loc[self.start_date + i, :] = 1
        
        # Centralize data
        
        self.data_VaR['pass_through'] = pass_through
        self.data_VaR['vc_elasticity'] = vc_elasticity        
        self.model_VaR['leontief'] = leontief
        
    def compute_inflation_shocks(self):
        
        final_indirect_shock, final_direct_shock = {}, {}
        final_inflation = {}
        
        for year in self.data_VaR['df_carbon_price_gs'].index:
            tau_carbon = self.data_VaR['df_carbon_price_gs'].loc[year]
            ghg_trajectory = self.data_VaR['df_ghg_gs'].loc[year]
            ghg_trajectory = numpy.maximum(0.0, ghg_trajectory)
            
            x = self.model_VaR['leontief'].dict_financials_gs['output']
            y = self.model_VaR['leontief'].dict_financials_gs['demand']
            A = self.model_VaR['leontief'].A
            
            ghg_intensity = self.model_VaR['leontief'].dict_financials_gs['ghg'] / x
            ghg_intensity = ghg_intensity.fillna(0.0)
            
            v = self.model_VaR['leontief'].dict_financials_gs['gva'] / x
            v = v.fillna(1.0)
            
            delta_c = tau_carbon * ghg_trajectory * ghg_intensity
            delta_p = self.model_VaR['leontief'].inflation_from_va(delta_c)
            
            dv_vc, delta_x, delta_y = self.model_VaR['leontief'].production_loss_from_inflation(delta_p)
            dv_cost = (1 - self.model_VaR['leontief'].pass_through) * delta_c * (x * (1 + delta_x))
            dv_cost = dv_cost / x
            
            final_direct_shock[year] = -dv_cost.fillna(0.0)
            final_indirect_shock[year] = dv_vc.fillna(0.0)
            final_inflation[year] = delta_p
            
        final_indirect_shock = numpy.maximum(pandas.concat(final_indirect_shock), -1.0)
        final_direct_shock = numpy.maximum(pandas.concat(final_direct_shock), -1.0)
        final_inflation = pandas.concat(final_inflation)
        
        return final_direct_shock, final_indirect_shock, final_inflation
