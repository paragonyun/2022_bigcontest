from time import strftime
import pingouin as pg
import pandas as pd

'''
1. 두 군집을 비교할 때 

ttest = TTEST(clus_df, 'Cluster_col') 

ttest_df = ttest.check_ttest()
ttest_df # p-value 해석 : 두 군집의 해당 칼럼은 통계적으로 유의미하게 다른 칼럼입니다. (유의수준 95%)


2. 2개 이상의 군집을 비교할 때
근데 군집간 분산이 다를 것으로 예상되기에 Welch-Anova 실시

anova = ANOVA()
anova_df = anova.check_anova()
anova_df # F-value True의 해석 : 세개 이상 군집의 해당 칼럼은 "모두 같은 것은 아니다 (= 적어도 하나는 다르다)" (유의수준 95%) 

'''

class TTEST :
    def __init__ (self, clus_df, clus_col : str) :
        self.clus_df = clus_df
        self.clus_col = clus_col
        
        self.ttest_objs = [i for i in self.clus_df.columns if i != self.clus_col]

    def check_ttest(self) :
        
        obj_df = self.clus_df.copy()
        
        values = []

        clus_unique = obj_df[self.clus_col].unique()
        clus_dfs = [  obj_df[obj_df[self.clus_col] == i] for i in clus_unique  ] ## 군집별 데이터프레임이 담긴 list 생성


        if len(clus_dfs) > 2 :
            raise 't-test는 2개의 군집을 대상으로 합니다. ANOVA를 진행하세요'
        
        cols = self.ttest_objs
        p_values = []

        for i in cols :

            result = pg.ttest(clus_dfs[0][i], clus_dfs[1][i], correction = 'auto')

            
            p_values.append(result['p-val'].values[0])

        TF = [True if p_val < 0.05 else False for p_val in p_values]

        ttest_df = pd.DataFrame({'P-Value' : p_values ,
                                '유의함_여부' : TF},
                                index = cols)

        return ttest_df

class ANOVA :
    def __init__ (self,  clus_df, clus_col : str) :
        self.clus_df = clus_df
        self.clus_col = clus_col 

        self.anova_objs = [i for i in self.clus_df.columns if i != self.clus_col]


    def check_anova(self, ) :
        obj_df = self.clus_df.copy()
        
        values = []

        clus_unique = obj_df[self.clus_col].unique()
        clus_dfs = [  obj_df[obj_df[self.clus_col] == i] for i in clus_unique  ] ## 군집별 데이터프레임이 담긴 list 생성


        if len(clus_dfs) <= 2 :
            raise 'ANOVA는 3개 이상의 군집을 대상으로 합니다. T-Test를 진행하세요'
        
        cols = self.anova_objs
        p_values = []

        for i in cols :
            result = pg.welch_anova(dv = self.clus_col, 
                                between = i ,
                                data = obj_df )

            
            p_values.append(result['p-unc'].values[0])
            

        TF = [True if p_val < 0.05 else False for p_val in p_values]

        anova_df = pd.DataFrame({'P-Value' : p_values ,
                                '유의함_여부' : TF},
                                index = cols)

        return anova_df
