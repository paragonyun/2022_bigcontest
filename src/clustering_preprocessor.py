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

## Feature Selection
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
# from sklearn.ensemble import ExtraTreesClassifier


## Feature Selection
from FRUFS import FRUFS
from lightgbm import LGBMClassifier, LGBMRegressor


class  ClusteringPreprocessor :

    def __init__ (self, dataset : pd.DataFrame, selection=False, extraction=False) :
        self.df = dataset
        self.scalers = [StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer]
        
        self.drop_cols = ['application_id', 'user_id', 'insert_time']

        self.onehot_cols = ['gender','income_type','employment_type','houseown_type','purpose',
                            'personal_rehabilitation_yn','personal_rehabilitation_complete_yn']

        self.continuous_cols = ['age', 'credit_score', 'yearly_income','service_year',
                                'desired_amount', 'existing_loan_cnt', 'existing_loan_amt',
                                'income_per_credit', 'existing_loan_percent']
    
        self.prep_scaled_dfs = [] ## DataFrame을 담는 List


        self.selection = selection ## True or False로 갈 거임
        self.extraction = extraction ## True or False로 갈 거임


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

        ## 결측치 전부 drop
        print('결측치 제거 전 : ', output_df.isnull().sum().sum())
        output_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        output_df.dropna(inplace=True)
        print('결측치 제거 후 : ', output_df.isnull().sum().sum())

        output_df = output_df.reset_index(drop=True)


        ## onehot 인코딩
        for i in self.onehot_cols :
            ohe = OneHotEncoder(sparse=False)
            ohe_df = pd.DataFrame(ohe.fit_transform(output_df[[i]]),
                                columns = [f'{i}_'+ str(col) for col in ohe.categories_[0]])
            output_df = pd.concat([output_df.drop(columns=[i]),
                                ohe_df], axis=1)

        # ## 결측치 전부 drop
        # print('결측치 제거 전 : ', output_df.isnull().sum().sum())
        # output_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # output_df.dropna(inplace=True)
        # print('결측치 제거 후 : ', output_df.isnull().sum().sum())

        output_df = output_df.reset_index(drop=True)

        return output_df

    def _scaling(self, df : pd.DataFrame) :
        scale_obj = df.copy()

        fitted_scalers = []


        for i in self.scalers :
            scaler_name = str(i).split('.')[-1][:-2]
            scaler = i()
            scaled_values = scaler.fit_transform(scale_obj[self.continuous_cols])
            scaled_df = pd.DataFrame(scaled_values, columns = self.continuous_cols)

            ori_df = scale_obj.drop(self.continuous_cols, axis=1)


            scaler_name = pd.concat([ori_df, scaled_df], axis=1)


            scaler_name.reset_index(inplace=True, drop=True)

            self.prep_scaled_dfs.append(scaler_name)
            fitted_scalers.append(scaler)
        

        return self.prep_scaled_dfs, fitted_scalers


    def _feature_selection(self, df_lst) : ## scaled된 DF List 를 가지고 Selection 측정
        '''
        기존의 Selection 기법들은 모두 Supervised 기법이기 때문에, 
        Unsupervised 기법인 FRUFS를 실험적으로 사용합니다.
        참고링크 : https://www.deepwizai.com/projects/how-to-perform-unsupervised-feature-selection-using-supervised-algorithms
        [PyPi] https://pypi.org/project/FRUFS/
        [Kaggle Notebook] https://www.kaggle.com/code/marketneutral/frufs-unsupervised-selection
        '''
        df = df_lst[0]

        select_dfs = []

        cat_cols_after_prep = []

        ## Categorical Data 지정
        for i in self.onehot_cols :
            for prep_cols in df.columns :
                if prep_cols.startswith(i) :
                    cat_cols_after_prep.append(prep_cols)

        for df in df_lst :
            model = FRUFS(model_r = LGBMRegressor(random_state=42),
                            model_c = LGBMClassifier(random_state=42),
                            categorical_features = cat_cols_after_prep,
                            random_state = 42,
                            k = 10,   ## k개까지 측정합니다.
                            n_jobs = -1)
            
            # fit transform은 k개로 이루어진 df를 return 합니다.
            selected_df = model.fit_transform(df)

            print('Selected Columns\n', selected_df.columns)

            # 이거 하면 XGBoost 스타일로 그려줍니다. 지금은 필요 없을 거 같아서 Pass
            # model.feature_importance()

            select_dfs.append(selected_df)

        return select_dfs

    def _feature_extraction(self, df_lst) :## scaled된 DF List 를 가지고 Extraction 수행
        extract_dfs = []

        df = df_lst[0]

        cat_1 = ['age', 'service_year']

        cat_2 = [i for i in df.columns if i.startswith('employment_type_')]

        cat_3 = ['yearly_income']

        cat_4 = ['credit_score','existing_loan_cnt','existing_loan_amt']
        
        cat_5 = ['desired_amount']


        for i in df.columns :
            if i.startswith('gender') :
                cat_1.append(i)
            
            elif i.startswith('income_type') or i.startswith('houseown_type'):
                cat_3.append(i)
            
            elif i.startswith('personal_rehabilitation_') :
                cat_4.append(i)

            elif i.startswith('purpose') :
                cat_5.append(i)
        
        pca_cols = [cat_1, cat_2, cat_3, cat_4, cat_5]

        for df in df_lst :
            extract_df = pd.DataFrame()

            for idx, obj_col in enumerate(pca_cols) :
                pca = PCA(n_components=0.8)
                transformed_idx = pca.fit_transform(df[obj_col])
                pca_idx_df = pd.DataFrame(transformed_idx, 
                                        columns = [str(obj_col) + str(idx) + '_' + str(i) for i in range(len(transformed_idx[0]))])
            
                
                extract_df = pd.concat([extract_df, pca_idx_df], axis=1)

            extract_dfs.append(extract_df)
        
        return extract_dfs

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
        viz_dfs = df_lst

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
            
            if self.extraction :
                plt.savefig(f'./data/Extraction_Result_of_{scaler_name}.png')

            elif self.selection :
                plt.savefig(f'./data/Selection_result_of_{scaler_name}.png')

            else : 
                plt.savefig(f'./data/Normal_Result_of_{scaler_name}.png')

            plt.show()

    def _reduce_size(self, df) :
        print('Data Size 줄이는 중...')
        output_df = df.copy()

        cols = output_df.columns 

        for i in cols :
            if output_df[i].dtype == 'int64' :
                output_df = output_df.astype({i : 'int8'})

            elif output_df[i].dtype == 'float64' :
                ## 얘는 혹시 몰라 32로 
                output_df = output_df.astype({i : 'float32'})

        return output_df
        

    def run(self)  :
        prep_df = self._basic_preprocessor(self.df)

        prep_df = self._reduce_size(prep_df)

        prep_scaled_df_lst, fitted_scalers = self._scaling(prep_df)

        print('Scaling 완료')

        print(f'총 {len(self.prep_scaled_dfs)}개의 DataFrame이 나왔습니다. shape : {self.prep_scaled_dfs[0].shape}')
        print(self.prep_scaled_dfs[0].info(),'\n\n')

        if self.extraction :
            print('Extraction을 시작합니다...')
            extract_dfs = self._feature_extraction(prep_scaled_df_lst)
            self._visualize(extract_dfs)

            print('Extract 결과 DF List를 반환합니다.')
    
            del prep_scaled_df_lst

            return extract_dfs, fitted_scalers

        elif self.selection :
            print('Selection을 시작합니다...')
            select_dfs = self._feature_selection(prep_scaled_df_lst) 
            self._visualize(select_dfs)

            print('Selection 결과 DF List를 반환합니다.')

            del prep_scaled_df_lst

            return select_dfs, fitted_scalers, prep_df

        else :
            self._visualize(prep_scaled_df_lst)


            print('일반 전처리 결과 DF List를 반환합니다.')
            return prep_scaled_df_lst, fitted_scalers


    '''
    Example
    1. 그냥 전처리 후의 시각화
    clus = ClusteringPreprocessor(df)
    clus.run()

    2. Extracction 이후의 시각화
    clus = ClusteringPreprocessor(df, extraction=True)
    clus.run()

    3. Selection 이후의 시각화
    clus = ClusteringPreprocessor(df, selection=True)
    clus.run()
    '''