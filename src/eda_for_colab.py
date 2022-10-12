#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings

import time

warnings.filterwarnings('ignore')
import missingno as msno

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
            if self.df[i].dtype == 'int32' :
                sns.distplot(self.df[i], ax = axes[idx])

            elif self.df[i].dtype == 'float32' :
                sns.distplot(self.df[i], ax=axes[idx])

            elif self.df[i].dtype == 'category' :
                if len(self.df[i].unique()) > 6 and len(self.df[i].unique()) <= 50 :
                    sns.countplot(y = self.df[i], ax= axes[idx])

                elif len(self.df[i].unique()) > 50 :
                    print(f'{i} 칼럼은 너무 많은 범주를 가지고 있습니다. 다른 전처리를 추천드립니다.')
                    continue

                
                else :    
                    sns.countplot(self.df[i], ax = axes[idx])
                    axes[idx].set_xticklabels(axes[idx].get_xticklabels(),rotation = 30)
        


        print('\nDone!')

        for i in range(1, 3*6 - len(columns) + 1 ) :
            axes[-i].remove()

        plt.subplots_adjust(left=0.1, bottom=0.1, 
                            right=0.9, top=0.9, 
                            wspace=0.3, hspace=0.4)

        plt.tight_layout() 

        ## 파일의 이름을 입력하면 'distribution of ~~'로 저장
        plt.savefig(f'/content/Distributions_Of_{self.filename}.png',
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

        plt.savefig(f'/content/Heat_Map_Of_{self.filename}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()

    '''
    pip install missingno 
    '''
    def check_missing_values (self) :
        
        print('결측치를 확인합니다...')

        print('👀칼럼 별 결측치 수')
        for col in self.df.columns :
            print(f'\t ❗ {col} : {self.df[col].isnull().sum()}')


        msno.bar(self.df)

        print('Done!!')

        plt.tight_layout() 

        plt.savefig(f'/content/Missing_Values_of_{self.filename}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()


    ## 이상치 check
    def check_outliers(self) :
        print('이상치를 확인합니다...')

        

        columns = self.df.columns

        con_cols = [i for i in columns if self.df[i].dtype != 'object' and self.df[i].dtype != 'datetime']

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

        plt.savefig(f'/content/Outliers_of_{self.filename}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()



class EDAPreprocessing :
    '''
    This class is for the EFFICIENT USING of dataframe..! 
    Because our data sometimes takes 10G usage of memory... 
    I think it is very dangerous to our computers..!
    '''
    def __init__(self, df) :
        self.df = df

    def _check_times (original_fn) :
        def wrapper_fn(*args, **kwargs) :
            start = time.time()
            result = original_fn(*args, **kwargs)
            end = time.time()

            print(f'{original_fn.__name__} 함수의 소요시간\n: {end-start:.2f}초')
            
            return result
        return wrapper_fn


    @_check_times
    ## 효율적인 메모리 사용을 위한 다이어트
    def diet_dataframe(self) :
        cols = self.df.columns 

        for i in cols :
            if self.df[i].dtype =='int64' or self.df[i].dtype == 'float64' :
                ## int와 float은 8, 16 으로 나타내면 나타낼 수 있는 수가 우리 데이터의 값을 담지 못함
                ## 64보단 적은 32로 바꿈 (그래도 Memory Useage는 절반이나 줄어든다.)

                if self.df[i].dtype =='int64' :
                    self.df = self.df.astype({i : 'int32'})
                    

                else : 
                    self.df = self.df.astype({i : 'float32'})

            elif self.df[i].dtype == 'object' :
                if 'time' in i or 'date' in i : ## 시간이나 날짜와 관련된 거면 그냥 넘김 (어떻게 처리할 것인지 논의 필요!)
                    continue
                
                else :
                    self.df = self.df.astype({ i : 'category'}) ## 범주가 너무 많지만 않다면 category가 더 효율적

        return self.df  

        '''
        사용법 예시
        trainer = EDAPreprocessing(df)
        dieted_df = trainer.diet_dataframe()
        '''