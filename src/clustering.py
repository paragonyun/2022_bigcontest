from os import link
from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings 
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_samples, silhouette_score

from yellowbrick.cluster import KElbow

from sklearn.decomposition import PCA

from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, MinMaxScaler

import gower
from datetime import datetime

import gc

from kmodes.kprototypes import KPrototypes




class Clustering :
    def __init__ (self, df : pd.DataFrame, scaled = False, num_clus = None) :
        self.df = df
        self.cluster_models = {'KM' : self._KMeans_clustering,
                                'MS' : self._MeanShift_clustering,
                                'DB' : self._DBSCAN_clustering,
                                'GM' : self._GaussianMixture_clustering,
                                'HI' : self._Hierachical_clustering}

        self.fin_df = df.copy()

        self.scaled = scaled
        self.num_clus = num_clus

        self.drop_cols = ['application_id','user_id','insert_time','company_enter_month']


    def _scaling(self, input_df) :
        print('스케일링 진행 중...')

        def clean_dataset(df):
            assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
            return df[indices_to_keep].astype(np.float64)

        scale_obj = input_df.copy()

        print('스케일링 전 : ', scale_obj.shape)

        scale_obj = scale_obj.drop(self.drop_cols, axis=1)

        scale_obj = clean_dataset(scale_obj).reset_index()

        ss = StandardScaler()

        transed = ss.fit_transform(scale_obj)
        
        print('스케일링 후 : ', scale_obj.shape)

        scaled_df = pd.DataFrame(transed, columns = scale_obj.columns)

        return scaled_df
        

    def _checking_elbows(self, input_df) :
        global km_best_elbow
        global gm_best_elbow

        print('최적의 군집 수 찾는 중...')
        elbow_df = input_df.copy()

        km = KMeans()
        gm = GaussianMixture()

        km_viz = KElbow(km, k = (2, 8), metric='silhouette')
        gm_viz = KElbow(gm, k = (2, 8), metric='calinski_harabasz', force_model = True)

        km_viz.fit(elbow_df)
        gm_viz.fit(elbow_df)

        km_best_elbow, gm_best_elbow = km_viz.elbow_value_, gm_viz.elbow_value_


    def _KMeans_clustering(self, input_df) :
        print('K-Means로 군집화 수행 중...')
        output_df = input_df.copy()
        
        model = KMeans(n_clusters = self.num_clus, init = 'k-means++')
        model.fit_predict(output_df)

        output_df['KMeans'] = model.labels_

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        pca_2 = PCA(n_components=2)
        pca_2_transformed = pca_2.fit_transform(output_df.drop(['KMeans'], axis=1))

        pca_2_df = pd.DataFrame({'x_axis' : pca_2_transformed[:,0],
                                'y_axis' : pca_2_transformed[:,1],
                                'Cluster' : output_df['KMeans']})

        pca_3 = PCA(n_components=3)
        pca_3_transformed = pca_3.fit_transform(output_df.drop(['KMeans'], axis=1))
        pca_3_df = pd.DataFrame({'x_axis' : pca_3_transformed[:, 0],
                        'y_axis' : pca_3_transformed[:, 1],
                        'z_axis' : pca_3_transformed[:, 2],
                        'Cluster' : output_df['KMeans']})


        for i in range(len(pca_2_df['Cluster'].unique())) :
            marker_i = pca_2_df[pca_2_df['Cluster'] == i].index
            ax1.scatter(x = pca_2_df.loc[marker_i, 'x_axis'],
                        y = pca_2_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax1.set_title('K-Means Clustering 2D Visualization')
        ax1.legend()

        for i in range(len(pca_3_df['Cluster'].unique())) :
            marker_i = pca_3_df[pca_3_df['Cluster'] == i].index
            ax2.scatter(xs = pca_3_df.loc[marker_i, 'x_axis'],
                        ys= pca_3_df.loc[marker_i, 'y_axis'],
                        zs =  pca_3_df.loc[marker_i, 'z_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax2.set_title('K-Means Clustering 3D Visualization')
        ax2.legend()

        plt.savefig(f'./data/Clustering_of{sys._getframe(0).f_code.co_name}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()

        self.fin_df['KM'] = model.labels_


    def _MeanShift_clustering(self, input_df) :
        print('Mean Shift로 군집화 수행 중...')
        output_df = input_df.copy()

        bandwidth = estimate_bandwidth(output_df)
        model = MeanShift(bandwidth = bandwidth)

        labels = model.fit_predict(output_df)
        output_df['MeanShift'] = labels

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        pca_2 = PCA(n_components=2)
        pca_2_transformed = pca_2.fit_transform(output_df.drop(['MeanShift'], axis=1))

        pca_2_df = pd.DataFrame({'x_axis' : pca_2_transformed[:,0],
                                'y_axis' : pca_2_transformed[:,1],
                                'Cluster' : output_df['MeanShift']})

        pca_3 = PCA(n_components=3)
        pca_3_transformed = pca_3.fit_transform(output_df.drop(['MeanShift'], axis=1))
        pca_3_df = pd.DataFrame({'x_axis' : pca_3_transformed[:, 0],
                        'y_axis' : pca_3_transformed[:, 1],
                        'z_axis' : pca_3_transformed[:, 2],
                        'Cluster' : output_df['MeanShift']})


        for i in range(len(pca_2_df['Cluster'].unique())) :
            marker_i = pca_2_df[pca_2_df['Cluster'] == i].index
            ax1.scatter(x = pca_2_df.loc[marker_i, 'x_axis'],
                        y = pca_2_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax1.set_title('Mean Shift Clustering 2D Visualization')
        ax1.legend()

        for i in range(len(pca_3_df['Cluster'].unique())) :
            marker_i = pca_3_df[pca_3_df['Cluster'] == i].index
            ax2.scatter(xs = pca_3_df.loc[marker_i, 'x_axis'],
                        ys= pca_3_df.loc[marker_i, 'y_axis'],
                        zs =  pca_3_df.loc[marker_i, 'z_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax2.set_title('Mean Shift Clustering 3D Visualization')
        ax2.legend()

        plt.savefig(f'./data/Clustering_of{sys._getframe(0).f_code.co_name}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()

        self.fin_df['MS'] = labels

        

    def _GaussianMixture_clustering(self, input_df) :
        print('Gaussian Mixture로 군집화 중...')
        output_df = input_df.copy()

        model = GaussianMixture(n_components = self.num_clus)
        labels = model.fit_predict(output_df)

        output_df['GM'] = labels

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        pca_2 = PCA(n_components=2)
        pca_2_transformed = pca_2.fit_transform(output_df.drop(['GM'], axis=1))

        pca_2_df = pd.DataFrame({'x_axis' : pca_2_transformed[:,0],
                                'y_axis' : pca_2_transformed[:,1],
                                'Cluster' : output_df['GM']})

        pca_3 = PCA(n_components=3)
        pca_3_transformed = pca_3.fit_transform(output_df.drop(['GM'], axis=1))
        pca_3_df = pd.DataFrame({'x_axis' : pca_3_transformed[:, 0],
                        'y_axis' : pca_3_transformed[:, 1],
                        'z_axis' : pca_3_transformed[:, 2],
                        'Cluster' : output_df['GM']})


        for i in range(len(pca_2_df['Cluster'].unique())) :
            marker_i = pca_2_df[pca_2_df['Cluster'] == i].index
            ax1.scatter(x = pca_2_df.loc[marker_i, 'x_axis'],
                        y = pca_2_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax1.set_title('Gaussian Mixture Clustering 2D Visualization')
        ax1.legend()

        for i in range(len(pca_3_df['Cluster'].unique())) :
            marker_i = pca_3_df[pca_3_df['Cluster'] == i].index
            ax2.scatter(xs = pca_3_df.loc[marker_i, 'x_axis'],
                        ys= pca_3_df.loc[marker_i, 'y_axis'],
                        zs =  pca_3_df.loc[marker_i, 'z_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax2.set_title('Gaussian Mixture Clustering 3D Visualization')
        ax2.legend()

        plt.savefig(f'./data/Clustering_of{sys._getframe(0).f_code.co_name}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()

        self.fin_df['GM'] = labels


    def _DBSCAN_clustering(self, input_df) :
        print('DBSCAN으로 군집화 중...')

        output_df = input_df.copy()

        model = DBSCAN()
        labels = model.fit_predict(output_df)


        output_df['DBSCAN'] = labels

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        pca_2 = PCA(n_components=2)
        pca_2_transformed = pca_2.fit_transform(output_df.drop(['DBSCAN'], axis=1))

        pca_2_df = pd.DataFrame({'x_axis' : pca_2_transformed[:,0],
                                'y_axis' : pca_2_transformed[:,1],
                                'Cluster' : output_df['DBSCAN']})

        pca_3 = PCA(n_components=3)
        pca_3_transformed = pca_3.fit_transform(output_df.drop(['DBSCAN'], axis=1))
        pca_3_df = pd.DataFrame({'x_axis' : pca_3_transformed[:, 0],
                        'y_axis' : pca_3_transformed[:, 1],
                        'z_axis' : pca_3_transformed[:, 2],
                        'Cluster' : output_df['DBSCAN']})


        for i in range(len(pca_2_df['Cluster'].unique())) :
            marker_i = pca_2_df[pca_2_df['Cluster'] == i].index
            ax1.scatter(x = pca_2_df.loc[marker_i, 'x_axis'],
                        y = pca_2_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax1.set_title('DBSCAN Clustering 2D Visualization')
        ax1.legend()

        for i in range(len(pca_3_df['Cluster'].unique())) :
            marker_i = pca_3_df[pca_3_df['Cluster'] == i].index
            ax2.scatter(xs = pca_3_df.loc[marker_i, 'x_axis'],
                        ys= pca_3_df.loc[marker_i, 'y_axis'],
                        zs =  pca_3_df.loc[marker_i, 'z_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax2.set_title('DBSCAN Clustering 3D Visualization')
        ax2.legend()

        plt.savefig(f'./data/Clustering_of{sys._getframe(0).f_code.co_name}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()

        self.fin_df['DB'] = labels

    def _Hierachical_clustering(self, input_df) :
        print('계층적 군집화 중...')

        output_df = input_df.copy()

        model = linkage(y = output_df, method='complete' ,metric='euclidean')

        labels = fcluster(model, t = 3, criterion='distance')

        plt.figure(figsize=(12,6))
        dendrogram(model, leaf_rotation = 90)
        plt.title('Hierachical Clustering Visualization')
        plt.savefig(f'./data/Clustering_of{sys._getframe(0).f_code.co_name}.png')
        
        plt.show()
        

        self.fin_df['HI'] = labels


    def _calculate_scores(self) :
        for i in self.cluster_models :
            score_ = silhouette_score(self.df, self.fin_df[i])
            print(f'Silhouette Score of {i} : {score_:.2f}')
        

    def run(self) :
        if self.scaled :
            self.cluster_models['KM'](self.df)
            self.cluster_models['MS'](self.df)
            self.cluster_models['DB'](self.df)
            self.cluster_models['GM'](self.df)
            self.cluster_models['HI'](self.df)
            
            self._calculate_scores()

            return self.fin_df

        else :
            scaled_df = self._scaling(self.df)
            
            # self._checking_elbows(scaled_df)
            
            self.cluster_models['KM'](scaled_df)
            self.cluster_models['MS'](scaled_df)
            self.cluster_models['DB'](scaled_df)
            self.cluster_models['GM'](scaled_df)
            self.cluster_models['HI'](scaled_df)
            
            self._calculate_scores()

            return self.fin_df


class GowerDistance :
    '''
    I think Gower Distance is fitted to our datas 
    because output dataset is mixed with continuous values and categorical values..!
    So I found some distance metric named "gower" and I test it with "DBSCAN"

    But this class was not used the visualization result was not good... ㅠㅠ
    '''

    def __init__ (self, raw_df, my_eps, pre_selection=False) :
        gc.collect()

        self.df = raw_df
        self.drop_cols = ['application_id', 'insert_time']
        self.continuous_cols = ['age', 'credit_score', 'yearly_income','service_year',
                                'desired_amount', 'existing_loan_cnt', 'existing_loan_amt',
                                ]
    
        self.selected_cols = ['mice_credit_score', 'mice_existing_loan_amt',
                            'income_type_EARNEDINCOME','houseown_type_자가']

        self.pre_selection = pre_selection

        self.my_eps=my_eps


    def _selection_preprocessing(self, df) :
        '''
        input : mice df
        '''

        output_df = df.copy()

        output_df = output_df.loc[:, self.selected_cols]

        output_df.reset_index(drop=True, inplace=True)

        income_type_dict = {
            1 : 'Y',
            0 : 'N'
        }

        houseown_type_dict = {
            1 : 'Y',
            0 : 'N'
        }

        rehabilitaion_dict = {
            0.0 : "N",
            1.0 : "Y"
        }

        output_df['personal_rehabilitation_yn'] = output_df['personal_rehabilitation_yn'].replace(rehabilitaion_dict)
        output_df['personal_rehabilitation_complete_yn'] = output_df['personal_rehabilitation_complete_yn'].replace(rehabilitaion_dict)
        output_df['income_type_EARNEDINCOME'] = output_df['income_type_EARNEDINCOME'].replace(income_type_dict)
        output_df['houseown_type_자가'] = output_df['houseown_type_자가'].replace(houseown_type_dict)

        con_cols = [i for i in output_df.columns if i != 'income_type_EARNEDINCOME' and i != 'houseown_type_자가']

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(output_df[con_cols])
        scaled_df = pd.DataFrame(scaled_values, columns = con_cols)

        ori_df = output_df.drop(con_cols, axis=1)

        output_df_scaled = pd.concat([ori_df, scaled_df], axis=1)    

        output_df_scaled.reset_index(drop=True)

        return output_df_scaled



    def _preprocessing(self, df):
        output_df = df.copy()

        output_df = output_df.drop(self.drop_cols, axis=1)

        ## birth_year와 company_enter_month 전처리 (continuous로 바꿔줌)
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

        purpose_dict = {
            'LIVING' : '생활비',
            'SWITCHLOAN' : '대환대출',
            'BUSINESS' : '사업자금',
            'ETC' : '기타',
            'HOUSEDEPOSIT' : '전월세보증금',
            'BUYHOUSE' : '주택구입',
            'INVEST' : '투자',
            'BUYCAR' : '자동차구입'
        }

        gender_dict = {
            1.0 : 'M',
            0.0 : 'F'
        }

        rehabilitaion_dict = {
            0.0 : "N",
            1.0 : "Y"
        }

        output_df['purpose'] = output_df['purpose'].replace(purpose_dict)
        output_df['gender'] = output_df['gender'].replace(gender_dict)
        output_df['personal_rehabilitation_yn'] = output_df['personal_rehabilitation_yn'].replace(rehabilitaion_dict)
        output_df['personal_rehabilitation_complete_yn'] = output_df['personal_rehabilitation_complete_yn'].replace(rehabilitaion_dict)

        output_df.replace([np.inf, -np.inf], np.nan)

        ## delete missing values and drop cols
        output_df = output_df.dropna(axis = 0)
        output_df.reset_index(drop=True, inplace=True)

        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(output_df[self.continuous_cols])
        scaled_df = pd.DataFrame(scaled_values, columns = self.continuous_cols)

        ori_df = output_df.drop(self.continuous_cols, axis=1)
        ori_df = ori_df.drop(['user_id'], axis=1)
        ori_df.reset_index(drop=True, inplace=True)

        output_df_scaled = pd.concat([ori_df, scaled_df], axis=1)    

        output_df_scaled.reset_index(drop=True)

        return output_df_scaled, output_df

    def _reduce_size(self, df):
        output_df = df.copy()

        cols = output_df.columns 

        for i in cols :
            if output_df[i].dtype == 'int64' :
                output_df = output_df.astype({i : 'int32'})
            
            elif output_df[i].dtype == 'float64' :
                output_df = output_df.astype({i : 'float32'})
            
            # elif output_df[i].dtype == 'object' :
            #     output_df = output_df.astype({i : 'category'})

        return output_df

    def _calculate_distance(self, df) :
        '''
        gower distance seems to be not needed to scale the continous values..
        (I read so many examples.. but all of them did not do scaling..!)
        '''

        print('Gower Distance 계산중...')
        output_df = df.copy()

        cat_TF = [True if output_df[i].dtype == 'object' else False for i in output_df.columns]

        print('Plz Check they are categorical data..!')
        print(output_df.columns)
        print(cat_TF)

        gower_distance_mat = gower.gower_matrix(output_df, cat_features=cat_TF)

        cols = [f'DataPoint_{i}' for i in range(len(gower_distance_mat))]


        gower_df = pd.DataFrame(gower_distance_mat, index=cols, columns= cols)

        return gower_df, output_df

    def _DBSCAN (self, mat, ori_df) :
        print('DBSCAN으로 Clustering 중...')

        db = DBSCAN(metric = 'precomputed', eps=self.my_eps)

        db.fit(mat)

        ori_df['Gower_DB'] = db.labels_

        print('✔파악된 군집')
        print(ori_df['Gower_DB'].unique())

        return ori_df
        
    def _check_clus_result(self, gower_mat : np.matrix, clus_df) : ## input_df should be a gower matrix

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        pca_2 = PCA(n_components=2)
        pca_2_transformed = pca_2.fit_transform(gower_mat)

        pca_2_df = pd.DataFrame({'x_axis' : pca_2_transformed[:,0],
                                'y_axis' : pca_2_transformed[:,1],
                                'Cluster' : clus_df['Gower_DB']})

        pca_3 = PCA(n_components=3)
        pca_3_transformed = pca_3.fit_transform(gower_mat)
        pca_3_df = pd.DataFrame({'x_axis' : pca_3_transformed[:, 0],
                        'y_axis' : pca_3_transformed[:, 1],
                        'z_axis' : pca_3_transformed[:, 2],
                        'Cluster' : clus_df['Gower_DB']})


        for i in range(len(pca_2_df['Cluster'].unique())) :
            marker_i = pca_2_df[pca_2_df['Cluster'] == i].index
            ax1.scatter(x = pca_2_df.loc[marker_i, 'x_axis'],
                        y = pca_2_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax1.set_title('Gower DBSCAN Clustering 2D Visualization')
        ax1.legend()

        for i in range(len(pca_3_df['Cluster'].unique())) :
            marker_i = pca_3_df[pca_3_df['Cluster'] == i].index
            ax2.scatter(xs = pca_3_df.loc[marker_i, 'x_axis'],
                        ys= pca_3_df.loc[marker_i, 'y_axis'],
                        zs =  pca_3_df.loc[marker_i, 'z_axis'],
                        label = f'Cluster {i}',
                        alpha = 0.3)

        ax2.set_title('Gower DBSCAN Clustering 3D Visualization')
        ax2.legend()

        plt.savefig(f'./data/Gower Clustering_of{sys._getframe(0).f_code.co_name}.png',
                    bbox_inches='tight', pad_inches=0)

        plt.show()

        # pd.set_option('max_rows', None)
        return clus_df.groupby('Gower_DB').agg(['median', 'mean']).T

    def run(self) :
        df = self.df

        if self.pre_selection :
            print('전처리 중...')
            prep_df = self._selection_preprocessing(df)

        else :
            print('전처리 중...')
            prep_df, origin_df = self._preprocessing(df)

        
        gower_mat = self._calculate_distance(prep_df)
        clus_df = self._DBSCAN(gower_mat, origin_df)

        group_by = self._check_clus_result(gower_mat, clus_df)

        return clus_df, group_by

class KPrototype :
    '''
    Second Way to Clustering mixed dataset!!
    Easy to use ! I introduce "KPrototype"
    
    '''

    def __init__ (self, raw_df, pre_preped = False, n_clus = 3) :
        gc.collect()

        self.df = raw_df
        self.drop_cols = ['application_id', 'insert_time']
        self.continuous_cols = ['age', 'credit_score', 'yearly_income','service_year',
                                'desired_amount', 'existing_loan_cnt', 'existing_loan_amt',
                                ]

        self.n_clus = n_clus

        self.pre_preped =pre_preped

    def _preprocessing(self, df) :
        print('전처리중...')
        output_df = df.copy()
        output_df = output_df.drop(self.drop_cols, axis=1)

        ## birth_year와 company_enter_month 전처리 (continuous로 바꿔줌)
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

        purpose_dict = {
            'LIVING' : '생활비',
            'SWITCHLOAN' : '대환대출',
            'BUSINESS' : '사업자금',
            'ETC' : '기타',
            'HOUSEDEPOSIT' : '전월세보증금',
            'BUYHOUSE' : '주택구입',
            'INVEST' : '투자',
            'BUYCAR' : '자동차구입'
        }

        gender_dict = {
            1.0 : 'M',
            0.0 : 'F'
        }

        rehabilitaion_dict = {
            0.0 : "N",
            1.0 : "Y"
        }

        output_df['purpose'] = output_df['purpose'].replace(purpose_dict)
        output_df['gender'] = output_df['gender'].replace(gender_dict)
        output_df['personal_rehabilitation_yn'] = output_df['personal_rehabilitation_yn'].replace(rehabilitaion_dict)
        output_df['personal_rehabilitation_complete_yn'] = output_df['personal_rehabilitation_complete_yn'].replace(rehabilitaion_dict)

        output_df.replace([np.inf, -np.inf], np.nan)


        ## delete missing values and drop cols
        output_df = output_df.dropna(axis = 0)
        output_df.reset_index(drop=True, inplace=True)

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(output_df[self.continuous_cols])
        scaled_df = pd.DataFrame(scaled_values, columns = self.continuous_cols)

        scaled_df.reset_index(drop=True, inplace=True)
        ori_df = output_df.drop(self.continuous_cols, axis=1)
        ori_df = ori_df.drop(['user_id'], axis=1) ## 군집화용에서만 drop
        ori_df.reset_index(drop=True, inplace=True)

        output_df_scaled = pd.concat([ori_df, scaled_df], axis=1)    


        output_df_scaled.reset_index(drop=True, inplace=True)

        return output_df_scaled, output_df


    def _kprototype(self, df) :
        print(f"K-Prototype으로 군집화 중..., 군집 수 : {self.n_clus}")
        output_df = df.copy()
        model = KPrototypes(n_clusters=self.n_clus, random_state=42, n_jobs=-1, verbose=1)

        cat_features_pre = [i for i in output_df.columns if output_df[i].dtype == 'object']

        cat_features_idx = [list(output_df.columns).index(i) for i in cat_features_pre]
        print(f"\nPreprocessed Data Frame's Columns\n{output_df.columns}\n")
        print(f'Detected Categorical Features Index are... {cat_features_idx} \n Is it right? [y/n]')
        
        yn = str(input())

        if yn == 'n' :
            print(yn)
            raise Exception('인덱스 코드를 다시 작성하세요')
        print(yn)
        print('Start K-Prototype Clustering...')
        model.fit_predict(output_df, categorical = cat_features_idx)

        df['KProto'] = model.labels_

        return df

    def _check_clus_result(self, df) :
        ## KProto 는 cat의 원래 형태를 사용하기 때문에 "거리로써 표현"하는 t-sne나 PCA는 적절하지 않다고 판단했습니다.
        ## 이에 groupby로 평균와 중앙값을 비교하는 형태로 바꿨습니다.

        output_df = df.copy()

        return output_df.groupby('KProto').agg(['median', 'mean']).T


    def run(self) :  
        df = self.df

        if self.pre_preped :
            print('이미 전처리 된 데이터입니다.')
            dropped = df.drop(['user_id'], axis=1)
            prep_df = dropped
            

        else :
            prep_df, origin_df = self._preprocessing(df)

        clus_df = self._kprototype(prep_df)

        group_by = self._check_clus_result(clus_df)

        if self.pre_preped :
            prep_df['KProto'] = clus_df['KProto']
            return prep_df, group_by

        else :

            origin_df['KProto'] = clus_df['KProto']

            return origin_df, group_by