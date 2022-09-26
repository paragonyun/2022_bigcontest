import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import warnings 
warnings.filterwarnings('ignore')

## Emcoders
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder 

## datetime
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

from sklearn.decomposition import PCA


class  ClusteringPreprocessor() :
    def __init__ (self, dataset : pd.DataFrame) :
        self.df = dataset
        self.scalers = [StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer]
        
        self.drop_cols = ['application_id', 'user_id', 'insert_time']

        self.onehot_cols = ['gender','income_type','employment_type','houseown_type','purpose',
                            'personal_rehabilitation_yn','personal_rehabilitation_complete_yn']

        self.continuous_cols = ['age', 'credit_score', 'yearly_income','service_year',
                                'desired_amount', 'existing_loan_cnt', 'existing_loan_amt',
                                'income_per_credit', 'existing_loan_percent']
    
        self.prep_scaled_dfs = [] ## DataFrame을 담는 List

    def _basic_preprocessor(self, df) -> pd.DataFrame : 
        print('전처리 시작...')
        output_df  = df.copy()

        ## 필요 없는 칼럼 제거
        output_df = output_df.drop(self.drop_cols, axis=1)


        ## 나이대로 바꾸기
        output_df['birth_year'] = pd.to_datetime(output_df['birth_year'], format='%Y', errors='ignore')
        def _calculate_age(x) :
            this_yaer = datetime.now().year
            return this_yaer - x.year
        output_df['age'] = output_df['birth_year'].apply(lambda x : _calculate_age(x))
        output_df.drop(['birth_year'], axis=1, inplace=True)

        ## 근속년수로 바꾸기
        def _cuting (x) :
            return str(x)[:6]
        output_df['company_enter_month'] =  output_df['company_enter_month'].apply(lambda x : _cuting(x))
        output_df['company_enter_month'] = pd.to_datetime(output_df['company_enter_month'], format='%Y%m')
        output_df['service_year'] = output_df['company_enter_month'].apply(lambda x : _calculate_age(x))
        output_df.drop(['company_enter_month'], axis=1, inplace=True)

        ## 파생변수 만들기
        # 신용점수 대비 연소득 : 연소득/신용점수
        output_df['income_per_credit'] = output_df['yearly_income'] / output_df['credit_score']

        # 기대출비율 : 기대출금액 / 연소득
        output_df['existing_loan_percent'] = output_df['existing_loan_amt'] / output_df['yearly_income']

        ## onehot 인코딩
        for i in self.onehot_cols :
            ohe = OneHotEncoder(sparse=False)
            ohe_df = pd.DataFrame(ohe.fit_transform(output_df[[i]]),
                                columns = [f'{i}_'+ str(col) for col in ohe.categories_[0]])
            output_df = pd.concat([output_df.drop(columns=[i]),
                                ohe_df], axis=1)

        ## 결측치 전부 drop
        print('결측치 제거 전 : ', output_df.isnull().sum().sum())
        output_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        output_df.dropna(inplace=True)
        print('결측치 제거 후 : ', output_df.isnull().sum().sum())

        output_df = output_df.reset_index(drop=True)

        return output_df

    def _scaling(self, df : pd.DataFrame) :
        scale_obj = df.copy()


        for i in self.scalers :
            scaler_name = str(i).split('.')[-1][:-2]
            scaler = i()
            scaled_values = scaler.fit_transform(scale_obj[self.continuous_cols])
            scaled_df = pd.DataFrame(scaled_values, columns = self.continuous_cols)

            ori_df = scale_obj.drop(self.continuous_cols, axis=1)


            scaler_name = pd.concat([ori_df, scaled_df], axis=1)


            scaler_name.reset_index(inplace=True, drop=True)

            self.prep_scaled_dfs.append(scaler_name)
        

        return self.prep_scaled_dfs



    def test_run(self) :
        '''
        전처리 잘 되는지 확인하는 함수입니다.
        '''
        prep_df = self._basic_preprocessor(self.df)

        prep_scaled_df_lst = self._scaling(prep_df)

        print('Scaling 완료')
        print('사용된 Scalers\n',self.scalers)
        print(f'총 {len(self.prep_scaled_dfs)}개의 DataFrame이 나왔습니다.')
        print('\n\n예시\n')

        return self.prep_scaled_dfs[0]


    def _visualize(self, df_lst : list) :
        print('시각화를 시작합니다...')
        viz_dfs = self.prep_scaled_dfs

        for viz_df, scaler in zip(viz_dfs, self.scalers) :
            scaler_name = str(scaler).split('.')[-1][:-2]

            print('현재 Scaler : ',  scaler_name)
            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection='3d')

            pca_2 = PCA(n_components=2)
            pca_2_transformed = pca_2.fit_transform(viz_df)
            pca_2_df = pd.DataFrame({'x_axis' : pca_2_transformed[: , 0],
                                    'y_axis' : pca_2_transformed[: , 1]})

            pca_3 = PCA(n_components=3)
            pca_3_transformed = pca_3.fit_transform(viz_df)
            pca_3_df = pd.DataFrame({'x_axis' : pca_3_transformed[:, 0],
                                    'y_axis' : pca_3_transformed[:, 1],
                                    'z_axis' : pca_3_transformed[:, 2],})

            ## 2차원 시각화 결과
            ax1.scatter(x = pca_2_df.loc[:, 'x_axis'], y = pca_2_df.loc[:, 'y_axis'],
                                alpha=0.4)
            ax1.set_title(f'2D Scatter Plot of {scaler_name}')


            ## 3차원 시각화 결과
            ax2.scatter(xs = pca_3_df.loc[:, 'x_axis'], ys = pca_3_df.loc[:, 'y_axis'], zs = pca_3_df.loc[:, 'z_axis'],
                                alpha=0.4)
            ax2.set_title(f'3D Scatter Plot of  {scaler_name}')

            plt.savefig(f'./data/PCA_Result_of_{scaler_name}.png')

            plt.show()
        

    def run(self) :
        prep_df = self._basic_preprocessor(self.df)

        prep_scaled_df_lst = self._scaling(prep_df)

        print('Scaling 완료')

        print(f'총 {len(self.prep_scaled_dfs)}개의 DataFrame이 나왔습니다. shape : {self.prep_scaled_dfs[0].shape}')
        print(self.prep_scaled_dfs[0].info())

        self._visualize(prep_scaled_df_lst)

        ## 메모리 절약을 위해... 지워줌..!
        del prep_scaled_df_lst

        print('Done')