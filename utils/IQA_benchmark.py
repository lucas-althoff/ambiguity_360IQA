import pandas as pd    
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

#MOS
def oiqa_tidy():
    path_oiqa = r'C:\Users\Usuário\Videos\Projetos\UnivPoitiers\QoEMEX2022\VQA_360Videos_Database\360IQA_ds1_OIQA\scores\OIQA_tidy.csv'
    oiqa = pd.read_csv(path_oiqa,sep=';')
    return oiqa 

def compare(path):
    df = pd.read_csv(path,sep=';')
    print(df)

    print(df.columns.tolist())

    stats_numeric1 = df['psnr'].describe()#.astype(int)
    stats_numeric2 = df['ssim'].describe()#.astype(int)
    stats_numeric3 = df['vif'].describe()#.astype(int)
    stats_numeric4 = df['vsi'].describe()#.astype(int)
    stats_numeric5 = df['fsim'].describe()#.astype(int)
    stats_numeric6 = df['brisque'].describe()#.astype(int)

    print(stats_numeric1, '\n',stats_numeric2, '\n',stats_numeric3, '\n',stats_numeric4, '\n',stats_numeric5, '\n',stats_numeric6)


def get_stat(path,iqa_metrics):
    df = pd.read_csv(path,sep=';')
    #stats_numeric1 = df['nique'].describe()#.astype(int)
    #stats_numeric2 = df['brisque'].describe()#.astype(int)

    for metric in iqa_metrics:
        m = df.groupby(['dist','scanpath'])[metric].mean()
        print(m)

    # df['ilniqe'].describe()
    # df['nrqm'].describe()
    # df['nima'].describe()
    # df['pi'].describe()
    # df['paq2piq'].describe()
    # df['dbcnn'].describe()
    
    #print(stats_numeric1)
    #print(stats_numeric2)

def plot_scatter(path, iqa_metrics, mos, img_sp):
    df = pd.read_csv(path,sep=';')

    plt.figure(num=1, figsize=(5, 3), dpi=200, facecolor='w', edgecolor='k')

    df_mean = df.groupby(['ref','dist','scanpath'])['brisque'].mean()
    df_mean = pd.DataFrame(df_mean).reset_index(inplace=False)
    df_mean_img = df_mean.groupby('dist').mean()
    #print(df_mean_img.reset_index(inplace=False))
    #print(pd.DataFrame(mos))
    
    mos = pd.DataFrame(mos)
 
    a = []
    for ind in range(1,337):
        row ='img'+str(ind)
        a.append(row)
    
    mos['dist'] = a
    mos = mos.reset_index()
    df_mean_img = df_mean_img.reset_index()
    cross_mos = pd.merge(df_mean_img,mos)
    #print(mos,type(mos))
    #print(df_mean_img,type(df_mean_img))
    #df_mean = df_mean.plot(legend=True)
    #sns.relplot(kind='line', data=cross_mos, x='brisque', y='score', aspect=1.75)
    sns.scatterplot(data=cross_mos, x="brisque", y="score")
    #print(df_mean,df_mean_img)
    #print(df_mean[df_mean.dist =='img100'])
    #print(df_mean['dist']=='img100')
    #for metric in iqa_metrics:
        #m = df.groupby(['dist','scanpath'])[metric].mean()
        #sns.scatterplot(data=m, x="MOS", y="")
        #print(m)

    #plt.legend(prop={'size': 6}, title = 'IQA Metric')
    plt.show()
     
def plot_line():     
    df = pd.read_csv(path,sep=';')


def plot_distribution(path,iqa_metrics = False,box=False,dist=False):
    df = pd.read_csv(path,sep=';')
    
    if not iqa_metrics:
        iqa_metrics = ['brisque',
        'nique',
        'ilniqe',
        'nrqm',
        'nima',
        'pi',
        'paq2piq',
        'dbcnn',
        'musiq-ava',
        'musiq-koniq']

    if dist and box:
        fig, axs = plt.subplots(ncols=2)
    else:
        plt.figure(num=1, figsize=(5, 3), dpi=200, facecolor='w', edgecolor='k')

    for x in iqa_metrics:
        # Subset to the species
        print(df[x],x)
        subset = df[x]/df[x].max()
        
        # Draw the density plot
        if dist:
            sns.distplot(subset, hist = False, kde = True,
                        kde_kws = {'linewidth': .5, 'shade' : True},
                        bins=10,
                        label = x,
                        ax = axs[0])

        if box:
            #print(df.drop(['brisque', 'psnr'], axis=1),'\n',pd.melt(df.drop(['brisque', 'psnr'], axis=1)),'\n')
            #df2 = pd.DataFrame(data=df, columns=['vsi','vif','ssim','fsim'])
            df2 = pd.DataFrame(data=df, columns=iqa_metrics)
            sns.boxplot(x="variable", y="value", data=pd.melt(df2), ax=axs[1])
    #plt.axvline(df[iqa_metrics[1]].median(), color='orange', linestyle='solid', linewidth=.8, alpha = 0.5)
    #plt.axvline(df[iqa_metrics[0]].median(), color='blue', linestyle='solid', linewidth=.8, alpha = 0.5)
    #for i,iqa_metric in enumerate(iqa_metrics):
    #    plt.axvline(df[iqa_metric].median(),color = 'k', linestyle='solid', linewidth=.8, alpha = 0.5)
    
    axs[0].legend(prop={'size': 6}, title = 'IQA Metric')
    axs[1].legend(prop={'size': 6}, title = 'IQA Metric')

    axs[0].set_xlabel('Normalized score')
    axs[0].set_ylabel('Density')
    #df.boxplot(column = "nique", figsize= (5,4))
    plt.show()   

def parse_args():
    parser = argparse.ArgumentParser(description="IQA benchmark app")
    parser.add_argument("--plot",  required=False,type=bool, default=False, help="Control function to run")
    parser.add_argument("--stats", required=False, type=bool, default=False, help="Control function to run")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    #path = r'C:\Users\lalthoff\Projects\AsymptoticScan\Code\asymptotic_IQA\result_nr_metrics.csv'
    path = r'D:\Althoff\Asymptotic\macrorun1\IQA_results\fr_scores_img325.csv'
    iqa_metrics = ['psnr',
    'ssim',
    'vif',
    'vsi',
    'fsim',
    'brisque'] 
    print(f'Computing mean for {path}')

    oiqa = oiqa_tidy()
    mos = oiqa.groupby('image')['score'].mean()
    scores = oiqa[oiqa['image']==325]
    #print(oiqa[oiqa['image']==325])
    #if args.stats:
    #    get_stat(path,iqa_metrics)
        #compare(path)

    if args.plot:
        plot_distribution(path,iqa_metrics,box=True,dist=True)
        plot_scatter(path,iqa_metrics,mos,scores)
    if args.vdp:
        import subprocess
        command = 'python3 fvvdp_run.py --ref D:\Sendjasni\OIQA_\img321.bmp --test D:\Althoff\Asymptotic\macrotest2\gen_projections\img1.jpg_0_*.jpg --display "htc_vive_pro" --heatmap threshold'
    