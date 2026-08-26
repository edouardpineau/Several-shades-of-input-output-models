import pandas
import numpy
import scipy

class Leontief:
    
    def __init__(self, dict_financials_gs, A, L=None, pass_through=None, precision=None):
        
        """
        Inputs:
            
            * dict_financials_gs (dict): 
            * A (pandas.DataFrame): matrices containing the intensity of the flows
            * L (None or pandas.DataFrame): Leontief inverse of A
        """
        
        self.dict_financials_gs = dict_financials_gs
        
        self.dict_financials_gs['gva'] = self.dict_financials_gs['gva'].fillna(0.0)
        self.dict_financials_gs['output'] = self.dict_financials_gs['output'].fillna(0.0)
        self.dict_financials_gs['demand'] = self.dict_financials_gs['demand'].fillna(0.0)
        self.dict_financials_gs['ghg'] = self.dict_financials_gs['ghg'].fillna(0.0)
        self.dict_financials_gs['ghg'] = numpy.maximum(self.dict_financials_gs['ghg'], 0.0)
        
        self.A = A
        self.L = L
        self.precision = precision
        
        if L is None:
            self.L = self.leontief_inverse_numerical(A, precision)
        
        if pass_through is None:
            self.pass_through = pandas.Series(index=self.A.index).fillna(1.0)
            self.Phi = numpy.diag(self.pass_through)
            self.L_Phi = self.L
        else:
            self.pass_through = pass_through
            self.Phi = numpy.diag(self.pass_through)
            self.L_Phi = self.leontief_inverse_numerical(self.Phi.dot(self.A.values), precision)
            self.L_Phi = self.Phi.dot(self.L_Phi)
            self.L_Phi = pandas.DataFrame(self.L_Phi, index=self.A.index, columns=self.A.columns)
        
        self.Phi = pandas.DataFrame(self.Phi, index=self.A.index, columns=self.A.columns)
        
        self.vc_elasticity = 1 - 1 / numpy.clip(self.pass_through, 0.00001, 0.99999)
            
    def inflation_from_inflation(self, delta_p):
        
        """ 
        Computes the inflation related to carbon cost, using the tax diffusion method described for example here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4369553
        
        Input:
            
            * delta_p (float or pandas.Series, >=0): delta_p
        
        Output:
        
            * delta_p
        
        """

        self.A_mu = numpy.diag(delta_p * self.pass_through)
        self.A_mu = self.A_mu / (1 + self.A_mu)
        self.A_mu = self.Phi.dot(self.A) + self.A_mu
        
        self.L_mu = self.leontief_inverse_numerical(self.A_mu, self.precision)
        self.L_mu = self.L_mu.T.dot(self.Phi)
        
        v = (self.dict_financials_gs['gva'] / self.dict_financials_gs['output']).fillna(1.0)
        
        P = self.L_Phi.T.dot(v)
        P_carbon = self.L_mu.dot(v)
        
        return P_carbon - P
        
    def inflation_from_va(self, delta_v):
        
        """ 
        Computes the inflation related to carbon cost, using the value-added degradation for example described in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4755259
        
        Input:
            
            * delta_v (float): additional value added related to cost
        
        Output:
        
            * delta_p
        
        """
        
        v = (self.dict_financials_gs['gva'] / self.dict_financials_gs['output']).fillna(0.0)
        v = numpy.clip(v, 0, numpy.inf)
        
        d = pandas.DataFrame(numpy.diag(1 + delta_v), 
                             index=self.A.index, 
                             columns=self.A.columns).fillna(1.0)
        
        self.L_cp = self.L_Phi.T.dot(d)
        P_carbon = self.L_cp.dot(v)
        P = self.L_Phi.T.dot(v)
        
        return P_carbon - P
    
    def inflation_from_intermediary(self, delta_p):
        
        """
        Computes the total loss of production capacities related to direct_prod_loss due to intermediary consumption inflation. 
        """
        
        self.A_ic = numpy.diag(1 + delta_p * self.pass_through)
        self.A_ic = self.A.values.T.dot(self.Phi).dot(self.A_ic)
        self.A_ic = pandas.DataFrame(self.A_ic, 
                                     index=self.A.index, 
                                     columns=self.A.columns)
        self.L_ic = self.leontief_inverse_numerical(self.A_ic.values, self.precision)
        self.L_ic = self.L_ic.dot(self.Phi)
        
        v = (self.dict_financials_gs['gva'] / self.dict_financials_gs['output']).fillna(1.0)
        
        P = self.L_Phi.T.dot(v)
        P_carbon = self.L_ic.dot(v)
        
        return P_carbon - P
        
    def production_loss_from_inflation(self, delta_p):
        
        """
        Computes the total loss of production capacities related to direct_prod_loss, as detailed in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4923237
        """
        
        delta_y = self.dict_financials_gs['demand'] * self.vc_elasticity * delta_p
        delta_x = self.L.dot(delta_y)
        
        x = self.dict_financials_gs['output']
        A = self.A
        
        dv = ((x + delta_x) * (delta_p - A.T.dot(delta_p)) + (delta_x * numpy.ones_like(delta_x) - delta_x * A.T.dot(numpy.ones_like(delta_x))))
        
        return dv / x, delta_x / x, delta_y / x

    def production_loss_from_production(self, delta_x):
        
        """
        Computes the inflation related to carbon cost, using the tax diffusion method described for example here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4369553
        
        Input:
            
            * delta_x (float or pandas.Series, >=0): 
        
        Output:
        
            * delta_p
        
        """
        
        self.A_stress = numpy.eye(*self.L.shape) * delta_x.apply(lambda x: x / (x + 1)).values
        self.A_stress = self.A + self.A_stress
        
        self.L_stress = numpy.eye(*self.A_stress.shape) - self.A_stress
        self.L_stress = pandas.DataFrame(scipy.linalg.pinv(self.L_stress), 
                                         index=self.L_stress.index, 
                                         columns=self.L_stress.columns)
        
        x_stress = self.L_stress.T.dot(self.dict_financials_gs['demand'])
        
        return x_stress / self.dict_financials_gs['output'] - 1
        
    def leontief_inverse_numerical(self, A, precision=None):
        
        """
        Compute numerically the Leontief inverse
        
        Input:
            
            * A (pandas.DataFrame or numpy.array): matrix of exchange intensity
            * precision (int): precision of the computation (higher is more accurate, lower is faster)
            
        Output:
            
            * L (pandas.DataFrame or numpy.array): Leontief inverse of A
        """
            
        if precision is None:
            L = numpy.eye(*A.shape) - A
            L = scipy.linalg.pinv(L)
            if isinstance(A, pandas.DataFrame):
                L = pandas.DataFrame(L, index=A.index, columns=A.columns)
        else:
            L = numpy.stack([numpy.linalg.matrix_power(A, n=k) for k in range(0, precision+1)])
            L = L.sum(0)
            if isinstance(A, pandas.DataFrame):
                L = pandas.DataFrame(L, index=A.index, columns=A.columns)
            
        return L
    
