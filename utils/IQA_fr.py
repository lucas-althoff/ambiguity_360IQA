import pyiqa
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from natsort import natsorted

def read_img_as_tensor(path):
    #print(path, os.path.isfile(path))
    img = Image.open(path).convert('RGB')
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img).unsqueeze(0)

def test_4metrics():
    ref=325
    dist=90
    scanpath=9
    example_path = f'D:\Althoff\Asymptotic\macrorun1\gen_projections\img{ref}\img{dist}\scanpath_{scanpath}'
    save_dir = r'D:\Althoff\Asymptotic\macrorun1\IQA_results'
    f = open(save_dir+r'\fr_scores_test.csv','a')
    f.write('ref,dist,scanpath,point,psnr,ssim,vif,brisque')
    f.close()

    for point in range(29):
        print(f'Computing for {point}')
        #print(f'Computing for image {example_path}'+f'\ref_{point}.bmp')
        ref_img = read_img_as_tensor(example_path+'\\ref_'+str(point)+'.bmp')  
        
        #if point < 5:
        dist_img = read_img_as_tensor(example_path+f'\{point}.png')
        #else:
        #    dist_img = read_img_as_tensor(example_path+f'\{point}.png')  

        f = open(save_dir+r'\fr_scores_test.csv','a')
        if point == 0:
            f.write('\n')
        f.write(f'{ref},{dist},{scanpath},{point},'\
                f'{float(psnr(ref_img,dist_img))},'
                f'{float(ssim(ref_img,dist_img))},'
                f'{float(vif(ref_img,dist_img))},'
                f'{float(brisque(ref_img,dist_img))}'+'\n')
        f.close()


def test():
    example_path = r'D:\Althoff\Asymptotic\test_ref2\gen_projections\img321\img1\scanpath_0'
    f = open(save_dir+'\fr_scores.csv','a')
    f.write('distImg,scanpath,point,psnr,ssim,ms_ssim,cw_ssim,fsim,vif,lpips,dists,niqe,brisque,nrqm,musiq,musiq-koniq')
    f.close()

    for point in range(30):
        #print(example_path)
        #print(f'Computing for image {example_path}'+f'\ref_{point}.bmp')
        ref_img = read_img_as_tensor(example_path+'\\ref_'+str(point)+'.bmp')  
        
        #if point < 5:
        dist_img = read_img_as_tensor(example_path+f'\{point}.jpg')
        #else:
        #    dist_img = read_img_as_tensor(example_path+f'\{point}.png')  

        f = open(save_dir+'\fr_scores.csv','a')
        f.write(#f'{ref},{dist},{scanpath},{point},'\
                f'{float(psnr(ref_img,dist_img))},'
                f'{float(ssim(ref_img,dist_img))},'
                f'{float(ms_ssim(ref_img,dist_img))},'
                f'{float(cw_ssim(ref_img,dist_img))},'
                f'{float(fsim(ref_img,dist_img))},'
                f'{float(vif(ref_img,dist_img))},'
                f'{float(lpips(ref_img,dist_img))},'
                f'{float(dists(ref_img,dist_img))},'
                f'{float(nique(ref_img,dist_img))},'
                f'{float(brisque(ref_img,dist_img))},'
                f'{float(nrqm(ref_img,dist_img))},'
                f'{float(musiq(ref_img,dist_img))},'
                f'{float(musiq_koniq(ref_img,dist_img))}'+'\n')
        f.close()

if __name__ == '__main__':
    # list all available metrics
    print(pyiqa.list_models())   

    device = torch.device('cpu')
    print("* Working on " + str(device))

    # metrics_fr = [
    # 'psnr',
    # 'ssim',
    # 'ms_ssim',
    # 'cw_ssim',
    # 'fsim',
    # 'vif',
    # 'lpips',
    # 'dists',
    # 'niqe',
    # 'brisque',
    # 'nrqm',
    # 'musiq',
    # 'musiq-koniq']
    metrics_fr = ['psnr','ssim','vif','fsim','vsi','brisque']

    psnr = pyiqa.create_metric('psnr', device=device)
    ssim = pyiqa.create_metric('ssim', device=device)
    #ms_ssim = pyiqa.create_metric('ms_ssim', device=device)
    #cw_ssim = pyiqa.create_metric('cw_ssim', device=device)
    fsim = pyiqa.create_metric('fsim', device=device)
    vif = pyiqa.create_metric('vif', device=device)
    vsi = pyiqa.create_metric('vsi', device=device)
    #lpips = pyiqa.create_metric('lpips', device=device)
    #dists = pyiqa.create_metric('dists', device=device)
    #nique = pyiqa.create_metric('niqe', device=device)
    brisque = pyiqa.create_metric('brisque', device=device)
    #nrqm = pyiqa.create_metric('nrqm', device=device)
    #musiq = pyiqa.create_metric('musiq', device=device)
    #musiq_koniq = pyiqa.create_metric('musiq-koniq', device=device)

    test=False
    compute_full = False
    if test:
        st = time.time()
        test_4metrics()
        et = time.time()
        elapsed = (et - st)/60
        print('Execution time:', elapsed, 'minutes')
        print('Execution time for 9 reference images:', 2700*elapsed/(60*24), 'days')

    else:
        root_path = r'D:\Althoff\Asymptotic\macrorun1\gen_projections'
        #root_path = r'D:\Althoff\Asymptotic\test_ref2\gen_projections'
        save_dir = r'D:\Althoff\Asymptotic\macrorun1\IQA_results'
        
        f = open(save_dir+r'\fr_scores_img321.csv','a')
        f.write('ref,dist,scanpath,point,psnr,ssim,vif,vsi,fsim,brisque \n')
        f.close()

        if compute_full:
            refs = next(os.walk(root_path), (None, None, []))[1]
        else:
            refs = ['img321']

        for r,ref in enumerate(refs):
            #dists = os.listdir(root_path+'\\'+ref) 
            dists = next(os.walk(root_path+'\\'+ref), (None, None, []))[1]
            dists = natsorted(dists)
            for d,dist in enumerate(dists):
                #scanpaths = os.listdir(root_path+'\\'+ref+'\\'+dist)
                scanpaths = next(os.walk(root_path+'\\'+ref+'\\'+dist), (None, None, []))[1]
                scanpaths = natsorted(scanpaths)
                for scanpath in scanpaths:
                    #os.listdir(root_path+'\\'+ref+'\\'+dist+'\\'+scanpath)
                    #next(os.walk(root_path+'\\'+ref), (None, None, []))[1]
                    for point in range(30):
                        #print(f'{dist}/{scanpath}/{point}')
                        example_path = root_path+f'\{ref}'+f'\{dist}\{scanpath}'
                        #print(example_path)
                        ref_img = read_img_as_tensor(example_path+'\\ref_'+str(point)+'.bmp')  
            
                        if d < 5:
                            dist_img = read_img_as_tensor(example_path+f'\{point}.jpg')
                        else:
                            dist_img = read_img_as_tensor(example_path+f'\{point}.png')  

                        f = open(save_dir+r'\fr_scores_img321.csv','a')

                        if compute_full:
                            f.write(f'{float(psnr(ref_img,dist_img))},'
                                    f'{float(ssim(ref_img,dist_img))},'
                                    f'{float(ms_ssim(ref_img,dist_img))},'
                                    f'{float(cw_ssim(ref_img,dist_img))},'
                                    f'{float(fsim(ref_img,dist_img))},'
                                    f'{float(vif(ref_img,dist_img))},'
                                    f'{float(lpips(ref_img,dist_img))},'
                                    f'{float(dists(ref_img,dist_img))},'
                                    f'{float(nique(ref_img,dist_img))},'
                                    f'{float(brisque(ref_img,dist_img))},'
                                    f'{float(nrqm(ref_img,dist_img))},'
                                    f'{float(musiq(ref_img,dist_img))},'
                                    f'{float(musiq_koniq(ref_img,dist_img))}'+'\n')
                        else: 
                            f.write(f'{ref},{dist},{scanpath},{point},'\
                                    f'{float(psnr(ref_img,dist_img))},'
                                    f'{float(ssim(ref_img,dist_img))},'
                                    f'{float(vif(ref_img,dist_img))},'
                                    f'{float(vsi(ref_img,dist_img))},'
                                    f'{float(fsim(ref_img,dist_img))},'
                                    f'{float(brisque(ref_img,dist_img))}'+'\n')
                        f.close()
                print('Computed for: ' + f'{dist} > {d}/20')