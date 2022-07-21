import os
from os import listdir
from os.path import isfile, join

def get_ref(dataset='oiqa'):
    if dataset == 'oiqa':
        ref_list = [r'D:/Sendjasni/OIQA_/img321.bmp',r'D:/Sendjasni/OIQA_/img322.bmp',r'D:/Sendjasni/OIQA_/img323.bmp',r'D:/Sendjasni/OIQA_/img324.bmp',r'D:/Sendjasni/OIQA_/img325.bmp',r'D:/Sendjasni/OIQA_/img326.bmp',r'D:/Sendjasni/OIQA_/img327.bmp',r'D:/Sendjasni/OIQA_/img328.bmp',r'D:/Sendjasni/OIQA_/img329.bmp',r'D:/Sendjasni/OIQA_/img330.bmp',r'D:/Sendjasni/OIQA_/img331.bmp',r'D:/Sendjasni/OIQA_/img332.bmp',r'D:/Sendjasni/OIQA_/img333.bmp',r'D:/Sendjasni/OIQA_/img334.bmp',r'D:/Sendjasni/OIQA_/img335.bmp',r'D:/Sendjasni/OIQA_/img336.bmp']
    return ref_list

def get_dist_list(ref,dir_path='D:/Sendjasni/OIQA_/',dataset='oiqa'):
    ref_list = get_ref(dataset)

    if (os.path.isfile(ref)):
            ref = ref 
    else:
        raise Exception(f"File {dir_path+ref} not found")
        
    # Search for the reference image in the list of references
    dist_list = []
    for i,refs in enumerate(ref_list):
        #print(refs , '\n', ref)
        if refs == ref: 
            # Construct the list of dist images
            for j in range(1,21,1):
                if j<6: 
                    dist_list.append(dir_path+f'img{20*i+j}.png')
                else:
                    dist_list.append(dir_path+f'img{20*i+j}.jpg')
    return dist_list

if __name__ == "__main__":
    for n in range(16):
        a=get_dist_list(get_ref('oiqa')[n])
        print('PARA N = ',n)
        print(a)
# path_to_read = r'C:\Users\lalthoff\Projects\AsymptoticScan\Code\data'
# read_paths = [(path_to_read + '/'+ f) for f in listdir(path_to_read) if isfile(join(path_to_read, f))] 

# # ref_paths = []
# # for f in listdir(path_to_read):
# #     if f.split('_')[0] == 'ref':
# #         ref_paths.append(path_to_read + '\\'+f)

# ref_dir = r'C:\Users\lalthoff\Projects\AsymptoticScan\Code\data\ref'
# ref_paths = listdir(ref_dir)

# dist_dir = r'C:\Users\lalthoff\Projects\AsymptoticScan\Code\data\dist'
# dist_paths = listdir(dist_dir)

# ref_q = []
# for ref_path in ref_paths:
#     # Getting coded names
#     a_ref = os.path.splitext(ref_path)[0].split('.')[0].split('_')[1]
#     sc_ref = os.path.splitext(ref_path)[0].split('.')[1].split('_')[1]
#     point_ref = os.path.splitext(ref_path)[0].split('.')[1].split('_')[2]
    
#     for dist_path in dist_paths:
#         # Getting coded names
#         dist = os.path.splitext(dist_path)[0].split('.')[0]
#         sc_dist = os.path.splitext(dist_path)[0].split('.')[1].split('_')[1]
#         point_dist = os.path.splitext(dist_path)[0].split('.')[1].split('_')[2]
#         print(dist,sc_dist,point_dist)

    # #Search for ref-dist pairs
    # if b == a:

    #     #Get all scanpaths pairs
    #     for sc in range(25):
    #         for point in range(30):
# if a == b:
#     dist_paths, ref_paths

#print(ref_paths)