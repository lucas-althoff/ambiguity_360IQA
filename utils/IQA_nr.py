import pyiqa
import torch
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import numpy as np
import itertools
from torchvision import transforms
from PIL import Image

import pandas as pd

def read_img(path):
    img = Image.open(path).convert('RGB')
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)

if __name__ == '__main__':

    # list all available metrics
    print(pyiqa.list_models())   

    device = torch.device('cpu')
    print("* Working on " + str(device))

    #Testing with two metrics
    #iqa_fr_metrics = [pyiqa.create_metric('psnr', device=device),pyiqa.create_metric('ssim', device=device),pyiqa.create_metric('pieapp', device=device)]
    brisque = pyiqa.create_metric('brisque', device=device)
    nique = pyiqa.create_metric('niqe', device=device)
    ilniqe = pyiqa.create_metric('ilniqe', device=device)
    nrqm = pyiqa.create_metric('nrqm', device=device)
    nima = pyiqa.create_metric('nima', device=device)
    pi = pyiqa.create_metric('pi', device=device)
    paq2piq = pyiqa.create_metric('paq2piq', device=device)
    musiq_ava = pyiqa.create_metric('musiq-ava', device=device)
    musiq_koniq = pyiqa.create_metric('musiq-koniq', device=device)
    dbcnn = pyiqa.create_metric('dbcnn', device=device)

    #Test1 path
    #path_to_read = r'C:\Users\lalthoff\Projects\AsymptoticScan\Code\data'

    #Test 1 Code
    # nique_nr_list=[]
    # brisque_nr_list=[]
    # for path in read_paths:
    #     print(f'Computing for image {path}')
    #     tensor_img = read_img(path)  
    #     nique_nr_list.append(float(nique(tensor_img.unsqueeze(0)))) 
    #     brisque_nr_list.append(float(brisque(tensor_img.unsqueeze(0))))
    
    # metrics_dict = {'nique':nique_nr_list,
    #                 'brisque':brisque_nr_list}

    # df = pd.DataFrame.from_dict(metrics_dict)
    # df['images'] = files
    # df.insert(0, 'images', df.pop('images'))
    # print(metrics_dict)
    # df.to_csv(r'test_result.csv', index=False, header=True)   

    import os
    from os import listdir
    from os.path import isfile, join

    #Test2 path
    #path_to_read = r'C:\Users\lalthoff\Projects\AsymptoticScan\microtest'
    path_to_read = r'D:\Althoff\Asymptotic\macrotest3\gen_projections'
    read_paths = [(path_to_read + '\\'+ f) for f in listdir(path_to_read) if isfile(join(path_to_read, f))] 
    files = [f for f in listdir(path_to_read) if isfile(join(path_to_read, f))]
    
    #Test 2 code
    nique_nr_list = []
    brisque_nr_list = []
    ilniqe_nr_list = []
    nrqm_nr_list = []
    nima_nr_list = []
    pi_nr_list = []
    paq2piq_nr_list = []
    dbcnn_nr_list = []
    musiq_ava_nr_list = []
    musiq_koniq_nr_list = []
    
    #Geting the name of all images inside the folder
    img_names = []
    for i,f in enumerate(files):
        name = f.split('_')[0]
        if not (name in img_names):
            img_names.append(name)

    #Get number of combinations (considering that each image has all 750 samples rendered)
    sc = list(range(25))
    points = list(range(30))
    combination_len = len(list(itertools.product(sc, points)))

    ref_paths = ['img321', 'img322', 'img323', 'img324', 'img325', 'img326', 'img327', 'img328', 'img329', 'img330', 'img331', 'img332', 'img333', 'img334', 'img335', 'img336']
    
    _,f, filenames = next(walk(mypath+ref_list[1]), (None, None, [])) 
    # Initialize csv header
    f = open(save_dir+'nr_scores.csv','a')
    f.write('distImg,scanpath,point,brisque,niqe,ilniqe,nrqm,nima,pi,paq2piq,music-ava,musiq-koniq,dbcnn')
    f.close()
    for f,path in zip(read_paths,files):
        print(f'Computing for image {path}')
        tensor_img = read_img(path_to_read+r'/'+path)  
        nique_nr_list.append(float(nique(tensor_img.unsqueeze(0)))) 
        brisque_nr_list.append(float(brisque(tensor_img.unsqueeze(0))))
        ilniqe_nr_list.append(float(ilniqe(tensor_img.unsqueeze(0))))
        nrqm_nr_list.append(float(nrqm(tensor_img.unsqueeze(0))))
        nima_nr_list.append(float(nima(tensor_img.unsqueeze(0))))
        pi_nr_list.append(float(pi(tensor_img.unsqueeze(0))))
        paq2piq_nr_list.append(float(paq2piq(tensor_img.unsqueeze(0))))
        dbcnn_nr_list.append(float(dbcnn(tensor_img.unsqueeze(0))))
        musiq_ava_nr_list.append(float(musiq_ava(tensor_img.unsqueeze(0))))
        musiq_koniq_nr_list.append(float(musiq_koniq(tensor_img.unsqueeze(0))))
        
        #Parsing the file name
        img_name = f.split('_')[0]
        scanpath = f.split('_')[1]
        point = f.split('_')[2].split('.')[0]
        #print(f'name = {img_name} \n scanpath = {scanpath} \n point = {point}')

        metrics_dict = {'nique':nique_nr_list,
                        'brisque':brisque_nr_list,
                        'ilniqe': ilniqe_nr_list,
                        'nrqm': nrqm_nr_list,
                        'nima': nima_nr_list,
                        'pi': pi_nr_list,
                        'paq2piq': paq2piq_nr_list,
                        'musiq-ava': musiq_ava_nr_list,
                        'musiq-koniq': musiq_koniq_nr_list,
                        'dbcnn': dbcnn_nr_list
                        }
        
        f.write(f'{path},{list(metrics_dict.keys())[0]},
                {list(metrics_dict.keys())[1]},
                {list(metrics_dict.keys())[2]},
                {list(metrics_dict.keys())[3]},
                {list(metrics_dict.keys())[4]},
                {list(metrics_dict.keys())[5]},
                {list(metrics_dict.keys())[6]},
                {list(metrics_dict.keys())[7]},
                {list(metrics_dict.keys())[8]},
                {list(metrics_dict.keys())[9]}
                +'\n') #csv line
        f.close()

    df = pd.DataFrame.from_dict(metrics_dict)
    df['images'] = files
    df.insert(0, 'images', df.pop('images'))
    df.to_csv(r'result_nr_metrics_macrotest3.csv', index=False, header=True)