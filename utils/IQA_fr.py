import pyiqa
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from natsort import natsorted
  
def get_ref(dir_path,dataset='oiqa'):     
    
    if dataset == 'oiqa':
        ref_list = [r'/img321.bmp',r'/img322.bmp',r'/img323.bmp',r'/img324.bmp',r'/img325.bmp',r'/img326.bmp',r'/img327.bmp',r'/img328.bmp',r'/img329.bmp',r'/img330.bmp',r'/img331.bmp',r'/img332.bmp',r'/img333.bmp',r'/img334.bmp',r'/img335.bmp',r'/img336.bmp']
    
    ref_path_list = []
    for ref_path in ref_list: 
        ref_path_list.append(dir_path+ref_path)   
    
    return ref_path_list

def get_dist_list(ref_path,dir_path,dataset='oiqa'):
    ref_list = get_ref(dir_path,dataset)

    if (os.path.isfile(ref_path)):
            ref = ref_path 
    else:
        raise Exception(f"File {ref_path} not found")
        
    # Search for the reference image in the list of references
    dist_list = []
    for i,refs in enumerate(ref_list):
        #print(refs , '\n', ref)
        if refs == ref: 
            # Construct the list of dist images
            for j in range(1,21,1):
                if j>5: 
                    dist_list.append(dir_path+'/'+f'img{20*i+j}.png')
                else:
                    dist_list.append(dir_path+'/'+f'img{20*i+j}.jpg')
    return dist_list

def read_img_as_tensor(path):
    #print(path, os.path.isfile(path))
    img = Image.open(path).convert('RGB')
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img).unsqueeze(0)

def test_metrics():
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

    test2 = False
    test= False
    compute_full_dataset = True
    compute_all_metrics = False
    compute_erp = True
    
    if test:
        st = time.time()
        test_metrics()
        et = time.time()
        elapsed = (et - st)/60
        print('Execution time:', elapsed, 'minutes')
        print('Execution time for 9 reference images:', 2700*elapsed/(60*24), 'days')
    elif test2:
        root_oiqa = r'/media/data/SIC/lalthoff/Projects/asymptotic/Data/OIQA'
        root_path = r'/media/data/SIC/lalthoff/Projects/asymptotic/Results/macrorun1/gen_projections'
        save_dir = r'/media/data/SIC/lalthoff/Projects/asymptotic/Results/macrorun1/'
        
        if compute_full_dataset:
            refs = next(os.walk(root_path), (None, None, []))[1]
        else:
            refs = ['img322','img323','img324','img326','img327','img328','img329','img330','img331','img332','img333','img334','img335','img336']
            #refs = ['img322']

        if compute_erp:
            f = open(os.path.join(save_dir,'oiqa_fr_iqa_erp.csv'),'a')
            f.write('ref,dist,psnr,ssim,vif,vsi,fsim,brisque \n')
            f.close()
            ref_img_path = get_ref(root_oiqa,dataset='oiqa')

            for r,ref in enumerate(refs):
                if r == 1:
                    ref_img = read_img_as_tensor(ref_img_path[r])
                    dist_list = get_dist_list(ref_img_path[r],root_oiqa)
                    files = os.listdir(root_oiqa)             
                    dists1=[f for f in files if '.jpg' in f]
                    dists2=[f for f in files if '.png' in f]
                    dists = dists1+dists2
                    dists = natsorted(dists)
                    print(ref_img_path[r])
                    for j in range(20):
                        print(dists[20*r+j])                    
                        if j == 0:
                            dist_img = read_img_as_tensor(os.path.join(root_oiqa,dists[20*r+j]))

                            f = open(os.path.join(root_oiqa,f'oiqa_fr_iqa_erp.csv'),'a')

                            if compute_all_metrics:
                                f.write(f'{ref},{dist},'\
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
                            else: 
                                f.write(f'{ref},{dists[20*r+j]},'\
                                        f'{float(psnr(ref_img,dist_img))},'
                                        f'{float(ssim(ref_img,dist_img))},'
                                        f'{float(vif(ref_img,dist_img))},'
                                        f'{float(vsi(ref_img,dist_img))},'
                                        f'{float(fsim(ref_img,dist_img))},'
                                        f'{float(brisque(ref_img,dist_img))}'+'\n')
                            f.close()
                            print(f'Metrics Computed {20*r+j}/320')
                            break 
    else:
        #root_path = r'D:/Althoff/Asymptotic/macrorun1/gen_projections'
        #root_path = r'D:\Althoff\Asymptotic\test_ref2\gen_projections'
        #save_dir = r'D:\Althoff\Asymptotic\macrorun1\IQA_results'

        root_oiqa = r'/media/data/SIC/lalthoff/Projects/asymptotic/Data/OIQA/'
        root_path = r'/media/data/SIC/lalthoff/Projects/asymptotic/Results/macrorun1/gen_projections'
        save_dir = r'/media/data/SIC/lalthoff/Projects/asymptotic/Results/macrorun1/'

        #/media/data/SIC/lalthoff/Projects/asymptotic/remote_code/ambiguity_360IQA/utils
        #/media/data/SIC/lalthoff/Projects/asymptotic/Code/utils/IQA_fr.py
        
        f = open(os.path.join(save_dir,'fr_scores_img322_img336'),'a')
        f.write('ref,dist,scanpath,point,psnr,ssim,vif,vsi,fsim,brisque \n')
        f.close()

        if compute_full_dataset:
            refs = next(os.walk(root_path), (None, None, []))[1]
        else:
            refs = ['img322','img323','img324','img326','img327','img328','img329','img330','img331','img332','img333','img334','img335','img336']
            #refs = ['img322']

        if compute_erp:
            f = open(os.path.join(save_dir,'oiqa_fr_iqa_erp.csv'),'a')
            f.write('ref,dist,psnr,ssim,vif,vsi,fsim,brisque \n')
            f.close()
            ref_img_path = get_ref(root_oiqa,dataset='oiqa')
            st = time.time()
            for r,ref in enumerate(refs):
                ref_img = read_img_as_tensor(ref_img_path[r])
                dist_list = get_dist_list(ref_img_path[r],root_oiqa)
                files = os.listdir(root_oiqa)             
                dists1=[f for f in files if '.jpg' in f]
                dists2=[f for f in files if '.png' in f]
                dists = dists1+dists2
                dists = natsorted(dists)
                #print(ref_img_path[r])
                for j in range(20):
                    #print(dists[20*r+j])                    
                    dist_img = read_img_as_tensor(os.path.join(root_oiqa,dists[20*r+j]))

                    f = open(os.path.join(save_dir,f'oiqa_fr_iqa_erp.csv'),'a')

                    if compute_all_metrics:
                        f.write(f'{ref},{dist},'\
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
                    else: 
                        f.write(f'{ref},{dists[20*r+j]},'\
                                f'{float(psnr(ref_img,dist_img))},'
                                f'{float(ssim(ref_img,dist_img))},'
                                f'{float(vif(ref_img,dist_img))},'
                                f'{float(vsi(ref_img,dist_img))},'
                                f'{float(fsim(ref_img,dist_img))},'
                                f'{float(brisque(ref_img,dist_img))}'+'\n')
                    f.close()
                    et = time.time()
                    elapsed = (et - st)/60          
                    print(f'Metrics Computed {20*r+j}/320  time consumption:{elapsed}')          
        else:    
            for r,ref in enumerate(refs):
                #dists = os.listdir(root_path+'\\'+ref)
                
                ref_dir = os.path.join(root_path, ref) 
                dists = next(os.walk(ref_dir), (None, None, []))[1]
            
                #print(f'{dists,root_path,ref,a}')
                dists = natsorted(dists)
                for d,dist in enumerate(dists):
                    #scanpaths = os.listdir(root_path+'\\'+ref+'\\'+dist)
                    dist_dir = os.path.join(ref_dir,dist)
                    scanpaths = next(os.walk(dist_dir), (None, None, []))[1]
                    scanpaths = natsorted(scanpaths)
                    for scanpath in scanpaths:
                        #os.listdir(root_path+'\\'+ref+'\\'+dist+'\\'+scanpath)
                        #next(os.walk(root_path+'\\'+ref), (None, None, []))[1]
                        for point in range(30):
                            #print(f'{dist}/{scanpath}/{point}')
                            scanpath_dir = os.path.join(dist_dir,scanpath)
                            #example_path = root_path+f'\{ref}'+f'\{dist}\{scanpath}'
                            #print(example_path)
                            ref_img = read_img_as_tensor(os.path.join(scanpath_dir,f'ref_{point}.bmp'))  
                            #ref_img = read_img_as_tensor(example_path+'\\ref_'+str(point)+'.bmp')  
                
                            if d < 5:
                                dist_img = read_img_as_tensor(os.path.join(scanpath_dir,f'{point}.jpg'))
                            else:
                                dist_img = read_img_as_tensor(os.path.join(scanpath_dir,f'{point}.png'))  

                            f = open(os.path.join(save_dir,r'fr_scores_img322_img336.csv'),'a')

                            if compute_all_metrics:
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
                    print('Computed for: ' + f'{ref} >'+ f'{dist} > {d}/19')