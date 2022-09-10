#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import missingno as msno

class EDA():
    
    def __init__(self, df, filename):    
        plt.style.use('ggplot')
        self.df = df
        self.filename = filename

    ## 데이터 분포 check
    def check_distributions(self):
        columns = self.df.columns

        fig, axes  = plt.subplots(3,
                                6,
                                figsize=(20,10)
        )

        axes = axes.ravel()

        print('Checking Distributions...')

        for idx, i in tqdm(enumerate(columns)) :
            if self.df[i].dtype == 'int64' :
                sns.distplot(self.df[i], ax = axes[idx])

            elif self.df[i].dtype == 'float' :
                sns.distplot(self.df[i], ax=axes[idx])

            elif self.df[i].dtype == 'object' :
                sns.countplot(self.df[i], ax = axes[idx])
        
        print('\nDone!')

        for i in range(1, 3*6 - len(columns) + 1) :
            axes[-i].remove()

        plt.subplots_adjust(left=0.1, bottom=0.1, 
                            right=0.9, top=0.9, 
                            wspace=0.3, hspace=0.4)

        plt.tight_layout() 

        ## 파일의 이름을 입력하면 'distribution of ~~'로 저장
        plt.savefig(f'./src/figs/Distributions Of {self.filename}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()

    ## 데이터 히트맵 check
    def check_corr(self) :
        print('변수간 상관 관계를 확인합니다...')
        mat = self.df.corr()
        fig, ax = plt.subplots(figsize = (20,10))
        sns.heatmap(mat, annot=True, fmt='.2f',
        cmap = 'coolwarm', mask=np.triu(mat, 1))

        print('Done!!')

        plt.tight_layout() 

        plt.savefig(f'./src/figs/Heat Map Of {self.filename}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()

    '''
    pip install missingno 
    '''
    def check_missing_values (self) :
        
        print('결측치를 확인합니다...')

        fig, axes = plt.subplots(2, 1, figsize=(20,10))

        axes[0].msno.matrix(self.df)
        axes[1].msno.bar(self.df, sort='ascending')

        print('Done!!')

        plt.tight_layout() 

        plt.savefig(f'./src/figs/Missing Values of {self.filename}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()


    ## 이상치 check
    def check_outliers(self) :
        print('이상치를 확인합니다...')

        

        columns = self.df.columns

        con_cols = [i for i in columns if (self.df[i].dtype == 'int64') or (self.df[i].dtype == 'float')]

        print('파악된 연속형 변수\n',con_cols)

        fig, axes  = plt.subplots(3,
                                6,
                                figsize=(20,10)
        )

        axes = axes.ravel()

        for idx, i in enumerate(con_cols) :

            sns.boxplot(self.df[i], ax=axes[idx], whis=1.5)

        print('Done!!')

        
        for i in range(1, 3*6 - len(con_cols) + 1) :
            axes[-i].remove()

        plt.subplots_adjust(left=0.1, bottom=0.1, 
                            right=0.9, top=0.9, 
                            wspace=0.3, hspace=0.4)

        plt.tight_layout() 

        plt.savefig(f'./src/figs/Outliers of {self.filename}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()