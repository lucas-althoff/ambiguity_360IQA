import json 

def read_json_file(filename):
    data = []
    with open(filename, 'r') as f:
        data = [json.loads(_.replace('}]}"},', '}]}"}')) for _ in f.readlines()]
    return data

if __name__ == '__main__':

    scanpath_path = r'C:\Users\lalthoff\Projects\AsymptoticScan\Code\results\sp_full.json'
    dataset = read_json_file(scanpath_path)
    img = 'img10.jpg'
    sc = 1
    #Opening my json
    for entry in dataset:
        #print('TESTE \n','\n',entry[0].keys(),len(entry))
        
        #entry has 8400 items, the first 25 are dicts and the last 8375 are lists of dicts 
        for i,edict in enumerate(entry):
            if isinstance(edict, dict): # First image
                if edict['file'] == img and edict['scanpath_id'] == sc:
                    print('theta =', edict['theta'])
            else: # The whole dataset except first image
                # Problema identificado: as listas de dicionarios foram iteradas de forma errada
                # Solucao: Gerar novamente sem utilizar a funcao append_json  
                if i == 245:
                    print(len(edict))
                if edict[0]['file'] == img and edict[0]['scanpath_id'] == sc:
                    print('theta =', edict['theta'])
                
                #     print(i,'\n',edict)
        #for sc in range(scanpath_num):
        #    for entry in entry_list:
        #        if entry['file'] == img and entry['scanpath_id'] == sc:
        #            theta = entry['theta']
        #            phi = entry['phi']
