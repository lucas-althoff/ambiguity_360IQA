import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path

'''
Framework to compute the ambiguity of IQA metrics for 360 images
Pre-processed: Scan path projections for ref and distorted images
Input: Pair of dist-ref images
Output: Ambiguity interval for the IQA metric
'''

def get_scanpath_scores(iqm_df_path,iqm_list,sp_num):
    '''
    input: list of objective metrics names,
    output: dict with quality scores for each metric sorted in sequence
    '''
    df = pd.read_csv(iqm_df_path,sep=';')
    scores = {}
    for d,dist in enumerate(dist_list):
        for sp in range(15):
            #if d == 0 and sp == 2:
               #print(df[(df.dist == dist) & (df.scanpath == f'scanpath_{sp}')])
            for iqm in iqm_list:
                print(df[(df.dist == dist) & (df.scanpath == f'scanpath_{sp}')][iqm].values.tolist())
                #scores = df_sp[iqm].values.tolist()
                #print(score)
    
    #print(df)

def include_sp(mos,q_list,eps=0.01):
    d = mos - mean(q_list)
    mae = d_old - d

    if mae < eps:
        include = true
    else:
        include = False
    return include

def equivalence_pixel(path_vp1,path_vp2):
    ''' Flag function indicating if the successesive viewports are equivalent in pixel wise manner
    Input: Path to the perceivebleness map in supra-threshold JOD scale
    Output: Boolean flag  (True: equivalent, False: non-equivalent)
    '''
    img = Image.open(path_vp1)
    img1 = np.array(img, dtype="float32") * 1.0/255.0
    a = img1[:,:,0].flatten()
    img = Image.open(path_vp2)
    img2 = np.array(img, dtype="float32") * 1.0/255.0
    b = img2[:,:,0].flatten()
    res1 = np.sum(a > .666)/a.shape
    res2 = np.sum(b > .666)/b.shape
    print(res1,'\n',res2)
    np.count_nonzero(img2 > 0.5)/(np.shape(img2)[0]*np.shape(img2)[1])

def get_dist_list(ref=1,dist_type='jpeg'):
    
    if ref!=1:
        ref_list = ['img321','img322','img323','img324','img325','img326','img327','img328','img329','img330','img331','img332','img333','img334','img335','img336']
        # Search for the reference image in the list of references
        dist_list = []
        for i,refs in enumerate(ref_list):
            # Construct the list of dist images
            for j in range(1,6,1): #OIQA has 5 distorted images per reference for each distortion type
                if dist_type=='jpeg': 
                    dist_list.append(dir_path+f'img{20*i+j}.jpg')
                elif dist_type=='jpeg2000':
                    dist_list.append(dir_path+f'img{5+20*i+j}.png')
                elif dist_type=='blur':
                    dist_list.append(dir_path+f'img{9+20*i+j}.png')
                elif dist_type=='noise':
                    dist_list.append(dir_path+f'img{14+20*i+j}.png')
    else: 
        i = ref_list.index(ref)
        # Search for the reference image in the list of references
        dist_list = []

        # Construct the list of dist images
        for j in range(1,6,1): #OIQA has 5 distorted images per reference for each distortion type
            if dist_type=='jpeg': 
                dist_list.append(dir_path+f'img{20*i+j}.jpg')
            elif dist_type=='jpeg2000':
                dist_list.append(dir_path+f'img{5+20*i+j}.png')
            elif dist_type=='blur':
                dist_list.append(dir_path+f'img{9+20*i+j}.png')
            elif dist_type=='noise':
                dist_list.append(dir_path+f'img{14+20*i+j}.png')
    return dist_list

def get_scanpath_meanscores(iqm_df_path,iqm_list,sp_num=15):
    '''
    Create a list of mean scores for a single scanpath and set of IQM
    input: list of objective metrics names,
    output: dict with quality scores for each metric sorted in sequence
    '''
    df = pd.read_csv(iqm_df_path,sep=';')
    mscores = []
    #zip(iqm,mscore)
    mscores_iqm1 = []
    mscores_iqm2 = []
    for d,dist in enumerate(dist_list): 
        for iqm in iqm_list:
            scores_list = []
            for sp in range(1,sp_num):
                #Unit test 
                # if d == 0 and sp == 2 and iqm == 'psnr':
                #     print(df[(df.dist == dist) & (df.scanpath == f'scanpath_{sp}')][iqm].values.tolist())
                #     scores = df[(df.dist == dist) & (df.scanpath == f'scanpath_{sp}')][iqm].values.tolist()
                #     scores = [float(score) for score in scores]
                #     mean = sum(scores) / len(scores)
                #     print(mean)
                
                print(dist,sp, f'scanpath_{sp}',sp_num-1)
                print(df[(df.dist == dist) & (df.scanpath == f'scanpath_{sp}')][iqm].values.tolist())
                scores = df[(df.dist == dist) & (df.scanpath == f'scanpath_{sp}')][iqm].values.tolist()
                scores = [float(score) for score in scores]
                mean = sum(scores) / len(scores)
                scores_list.append(mean)
                #scores = df_sp[iqm].values.tolist()
                #mscores_iqm1 = df_sp[iqm].values.tolist().mean()
                #mscores_iqm2 = df_sp[iqm].values.tolist().mean()
                #mscores.append()
                #print(score)
            scores_dict = {
            'iqm':iqm,
            'img':dist,
            'scanpath': [f'scanpath_{sp}' for sp in range(sp_num)],
            'scores': scores_list
            }
            mscores.append(scores_dict)
    #print(mscores)
    return mscores 

if __name__ == '__main__':

    #IQM quality scores
    iqm_df_path = Path('D:/Althoff/Asymptotic/macrorun1/IQA_results/fr_scores_img321_img325.csv')
    iqm_list = ['psnr','ssim']
    dist_list = ['img81','img82','img83','img84','img85'] #sorted with respect to the amount of distortion
    get_scanpath_scores(iqm_df_path,iqm_list,sp_num=30)

    dataset = True

    if dataset:
        for 
            include_sp(mos,q_list,eps=0.1)
            num_sp += 1
    else:
        print('This package only compute ambiguity interval for a dataset')
    #Perceibleness sequence
    
    #Ambiguity interval
    #get_ai(increase = True)

    #Perceibleness map
    #path = r'D:\Althoff\Asymptotic\test_ref3\gen_projections\img326\img101\scanpath_7\heat_maps'
    #img = Image.open(path+r'\5_diff_map.png')#.convert("L") 
    
    #img1 = np.array(img, dtype="float32") * 1.0/255.0
    #a = img1[:,:,0].flatten()

    #path = r'D:\Althoff\Asymptotic\test_ref3\gen_projections\img326\img115\scanpath_9\heat_maps'
    #img = Image.open(path+r'\5_diff_map.png')#.convert("L") 

    #img2 = np.array(img, dtype="float32") * 1.0/255.0
    #b = img2[:,:,0].flatten()
    #res1 = np.sum(a > .666)/a.shape
    #res2 = np.sum(b > .666)/b.shape
    #print(res1,'\n',res2)
    #np.count_nonzero(img2 > 0.5)/(np.shape(img2)[0]*np.shape(img2)[1])