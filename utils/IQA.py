import pyiqa
import torch
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path


def split_dataset(dir_path):
    Path(dir_path+'').mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':

    # list all available metrics
    print(pyiqa.list_models())   

    device = torch.device('cpu')
    print("* Working on " + str(device))
    
    #For all metrics available
    #iqa_metric_names = ['brisque', 'ckdn', 'cw_ssim', 'dbcnn', 'dists', 'fid', 'fsim', 'gmsd', 'ilniqe', 'lpips', 'lpips-vgg', 'mad', 'ms_ssim', 'musiq', 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'niqe', 'nlpd', 'nrqm', 'paq2piq', 'pi', 'pieapp', 'psnr', 'ssim', 'vif', 'vsi']
    #for iqa_name in iqa_metric_names:
    #    iqa_metrics.append(pyiqa.create_metric(iqa_name, device=device))

    #Testing with two metrics
    iqa_fr_metric = [pyiqa.create_metric('psnr', device=device),pyiqa.create_metric('ssim', device=device)]
    iqa_nr_metric = [pyiqa.create_metric('brisque', device=device),pyiqa.create_metric('pieapp', device=device)]

    path_to_read = r'D:/Althoff/Asymptotic/macrotest1/gen_projections'
	path_to_save = r'D:/Althoff/Asymptotic/macrotest1/IQA_results'
    read_paths = [(path_to_read + '/'+ f) for f in listdir(path_to_read) if isfile(join(path_to_read, f))] 

    ref_paths = []
    for f in listdir(path_to_read):
        if f.split('_')[0] == 'ref':
            ref_paths.append(path_to_read + '\\'+ f)

    os.path.splitext(image_path)[1]

    #score_fr = iqa_fr_metric(dist_paths, ref_paths)
    for iqm in iqa_nr_metric:
        for path in paths:
            score_nr = iqm(path)

    # Testing for manual selection of images
	#paths = ['D:/Sendjasni/OIQA_/img1.jpg','D:/Sendjasni/OIQA_/img2.jpg']
	#paths = [(path_to_save + '/' + img_path) for img_path in img_paths] 
	
	# Get paths from all folder's files