from __future__ import print_function
#%matplotlib inline
import configargparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

import scanpath_projection

'''
Suppress SourceChangeWarning - we have removed comment lines and debug options from the source code.
'''
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

# Import inference module
from inference import basic_inference,basic_inference_fr
import os
from os import listdir
from os.path import isfile, join

# To break the computation into batches
from itertools import islice

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_ref(dir_path,dataset='oiqa'):     
    if dataset == 'oiqa':
        ref_list = [r'/img321.bmp',r'/img322.bmp',r'/img323.bmp',r'/img324.bmp',r'/img325.bmp',r'/img326.bmp',r'/img327.bmp',r'/img328.bmp',r'/img329.bmp',r'/img330.bmp',r'/img331.bmp',r'/img332.bmp',r'/img333.bmp',r'/img334.bmp',r'/img335.bmp',r'/img336.bmp']
    
    ref_path_list = []
    for ref_path in ref_list: 
        ref_path_list.append(path + ref_path)   
    return ref_path_list

def get_hmd(device = 'htc_vive_pro'):
	hmd_dict = {'quest' : {
		"name": "Oculus Quest", 
		"resolution": [1440, 1600], 
		"fov_diagonal": 110.0,
		"viewing_distance_meters": 3,
		"max_luminance": 133.3, 
		"min_luminance": 0.1,    
		"source": "https://arxiv.org/pdf/1912.02913.pdf"},
	'quest2' : {
		"name": "Meta Quest 2", 
		"resolution": [1920, 1832], 
		"fov_diagonal": 110.0,
		"viewing_distance_meters": 3,
		"max_luminance": 133.3, 
		"min_luminance": 0.1,    
		"source": "https://arxiv.org/pdf/1912.02913.pdf"},
  	"htc_vive_pro": {
		"name": "HTC Vive Pro", 
		"resolution": [1440, 1600], 
		"fov_diagonal": 110.0,
		"viewing_distance_meters": 3,
		"max_luminance": 133.3, 
		"min_luminance": 0.1,    
		"source": "https://arxiv.org/pdf/1912.02913.pdf"},
	"low_res": {
		"name": "Low resolution and small FoV for test", 
		"resolution": [800, 400], 
		"fov_diagonal": 60.0,
		"viewing_distance_meters": 3,
		"max_luminance": 133.3, 
		"min_luminance": 0.1}}
	return hmd_dict[device]

if __name__ == "__main__":
	# Parse arguments
	parser = configargparse.ArgumentParser()
	parser.add_argument("--batch", required=False, default=True, help="Computation mode: --batch", action='store_true')
	parser.add_argument("--build", required=False, default=False, help="Build scanpath: --build", action='store_true')
	parser.add_argument("--reference", required=True, type=str, default='fr', help="Build scanpath for no-reference (nr) or full-reference (fr) scenario")
	parser.add_argument("--render", required=False, default=False, help="Render scanpath: --render",action='store_true')
	parser.add_argument("--read_json", required=False, type=str, default='sp_full.json', help="Path of the scanpath generated json file: --json")
	parser.add_argument("--gpu", required=False, type=int, default=-1, help="Render scanpath: --render")
	args = parser.parse_args()

	#Local
	#path_to_read = r'D:/Sendjasni/OIQA_'
	#path_to_save = r'D:/Althoff/Asymptotic/test_ref/'
	#path_to_json = r'D:/Althoff/Asymptotic/test/sp_full.json'
	
	#Server
	path_to_read = r'/media/data/SIC/lalthoff/Dataset/OIQA_'
	path_to_save = r'/media/data/SIC/lalthoff/Projects/asymptotic/Results/macrorun1'
	scanpath_dir = r'/media/data/SIC/lalthoff/Projects/asymptotic/Code/Data
	path_to_json = r'/media/data/SIC/lalthoff/Projects/asymptotic/Code/Data/oiqa_15sp_30points.json'

	# Get paths from all image files
	paths = [(path_to_save + '/'+ f) for f in listdir(path_to_save) if isfile(join(path_to_save, f))]  

    #OIQA
    n_ref = 16
    n_dist = 20 #20 dist per ref where there is 4 dist types, meaning 5 img per dist type
	len_batch = n_dist #for OIQA dataset the number of distorted image correspond to the batch length
	batch_chunk = 2 #subdivision of the batch in chunks of # elements

	batch = True
	build_scanpath = False
	render_scanpaths = True

	print('ARGS',args)
	if args.batch:
		#Build scanpaths
		if args.build:
			# Model currently developed to work on GPU
			#device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
			# Workaround 1 refactor code to work with cpu
			
			if args.gpu >= 0 and torch.cuda.is_available():
				device = torch.device('cuda:' + str(opt.gpu))
			else:
				device = torch.device('cpu')
			print("* Working on " + str(device))
			#assert str(device) == "cuda:0"

			if args.reference == 'nr':
				# Load model_generator   
				generator = torch.load('models/model_generator_217.pth',map_location=device)  

				for i,paths in enumerate(chunks(read_paths, 10)):
					print('Processing image batch > ', i,'/',len(chunks(read_paths, 10)))
					basic_inference(image_paths=paths, n_batch=i, generator=generator, device=device, path_to_save=path_to_save, file_name= f'sp_full.json', n_generated=25)
					# Control the number of loops to batches
					#if i == 3:
					#	break
			else: #Full-reference
				read_paths = get_ref(path_to_read,dataset='oiqa')
				generator = torch.load('models/model_generator_217.pth',map_location=device)
				for i,paths in enumerate(chunks(len_batch, batch_chunk)):
					#print('Processing image > ', i,'/',len(chunks(read_paths, 10)))			
					basic_inference_fr(image_paths=paths, n_batch=i, generator=generator, device=device, path_to_save=path_to_save, file_name= f'sp_ref.json', n_generated=100)

		#Render scanpaths
		if args.render:

			if args.reference == 'nr':
				scanpath_path = scanpath_folder+'/'+args.read_json
				save_dir = path_to_save+r'/gen_projections/' 
				for i,chunk in enumerate(chunks(len_batch, batch_chunk)):
					print('Processing batch > ', i,'/',len(chunks(len_batch, batch_chunk)))	
					scanpath_projection.call_main(paths=paths, scanpath_path=scanpath_path, save_dir=save_dir, img_dir=path_to_read, hmd=hmd_res, scanpath_num=15)
					# Control the number of loops to batches
					# if i == 2:
					# 	break
			else:
				read_paths = get_ref(path_to_read,dataset='oiqa')
				scanpath_path = scanpath_folder+'/'+args.read_json
				save_dir = path_to_save+r'/gen_projections/' 
				hmd_res = get_hmd('htc_vive_pro')['resolution']
				for i,chunk in enumerate(chunks(len_batch, batch_chunk)):
					print('Processing batch > ', i,'/',len(chunks(len_batch, batch_chunk)))				
					scanpath_projection.call_ref(scanpath_path=scanpath_path, save_dir=save_dir, img_dir=path_to_read, hmd=hmd_res, scanpath_num=15)
					# Control the number of loops to batches
					# if i == 1:
					# 	break
	else:
		# Testing output by controlling img list 
		paths = ["D:/Sendjasni/OIQA_/img10.png", "D:/Sendjasni/OIQA_/img2.jpg"]
		basic_inference(image_paths=paths, generator=generator, device=device, path_to_save=path_to_save, file_name= 'sp_full_test_batch.json', n_generated=25)
		print('Initiating projection rendering from ',paths)
		scanpath_projection.call_main(paths=paths, scanpath_path=path_to_save,file_name='sp_full_test_batch.json', scanpath_num=25)

	print("Done.")

