self.beta = 0.005		#Baseline transmission rate
self.res_ratio = 1		#Max resistance ratio
self.res_avg = 1
self.inf_ratio = 1
self.inf_avg = 1

self.resJ_max = 2*self.res_avg/(self.res_ratio + 1)
self.resA_max = 2*self.res_avg*self.res_ratio/(self.res_ratio + 1)
self.infJ_max = 2*self.inf_avg/(self.inf_ratio + 1)
self.infA_max = 2*self.inf_avg*self.inf_ratio/(self.inf_ratio + 1)

self.resJ = np.linspace(0, self.resJ_max, self.H_alleles)
self.resA = np.linspace(self.resA_max, 0, self.H_alleles)
self.infJ = np.linspace(0, self.infJ_max, self.P_alleles)*self.beta
self.infA = np.linspace(self.infA_max, 0, self.P_alleles)*self.beta