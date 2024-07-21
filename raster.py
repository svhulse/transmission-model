import numpy as np
import tqdm
import sys
import json 
import pickle as pkl
import multiprocessing as mp

from model import Model


output_path = 'coevol.p'			#Name of raster scenario
size = 10					#Raster dimension
cores = 4					#Number of CPU cores

var_1 = 'mat'					#First parameter rastered
var_2 = 'adult_bias'			#Second parameter rastered
mode = 'coevol'

'''
Range of values for raster variables, too vary something other than v, c_g, or c_s, you 
will need to add a new range array and add it to the vars dict below
'''
mat_vals = np.linspace(0.25, 1, size)			#Range of parameters for virulence costs
bias_vals = np.linspace(1, 4, size)			#Range of parameters for general resistance costs

def pass_to_sim(model):
	return model.run_sim()

if __name__ == '__main__':
	coords = []     #x, y coordinates of each simulation in raster
	models = []     #Empty tuple for model classes

	print('Initializing Models...')
	#Create raster of model classes for each parameter combination
	for i in range(size):
		for j in range(size):
			coords.append((i,j))
			params = {'mat': mat_vals[i], 'adult_bias': bias_vals[j], 'mode': mode}
			new_model = Model(**params)

			models.append(new_model)
		
	#Run simluations for 4 core processor
	pool = mp.Pool(processes=cores)	
	
	print('Running Simulations...')
	results = []
	for result in tqdm.tqdm(pool.imap(pass_to_sim, models), total=len(models)):
		results.append(result)

	raster = []
	for i in range(size):
		inds = [j for j in range(len(coords)) if coords[j][1] == i]
		raster.append([results[j] for j in inds])

	with open(output_path, 'wb') as f:
		pkl.dump([models, results], f)