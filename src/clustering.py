# TODO
# 1. df 하나만 보는 거 list로 바꾸기

from os import link
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

class Clustering ():
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
        
        '''
        이상하게 inf 값이 섞여있더라구요.. 일단 제외를 시키는 함수입니다.
        무엇이 inf값을 가지는지 파악하고 대체값을 넣고 하겠습니다.
        '''
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

        ## 이 output_df에 저장된 결과로 각 모델별로 시각화도 시키고
        ## 나중에 이 결과들을 모아서 Hard Voting도 시킬 거임
        output_df['KMeans'] = model.labels_

        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(output_df.drop(['KMeans'], axis=1))

        pca_df = pd.DataFrame({'x_axis' : pca_transformed[:,0],
                                'y_axis' : pca_transformed[:,1],
                                'Cluster' : output_df['KMeans']})
        

        for i in range(len(pca_df['Cluster'].unique())) :
            marker_i = pca_df[pca_df['Cluster'] == i].index
            plt.scatter(x = pca_df.loc[marker_i, 'x_axis'],
                        y = pca_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}')

        plt.title('K-Means Clustering Visualization')
        plt.legend()

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

        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(output_df.drop(['MeanShift'], axis=1))

        pca_df = pd.DataFrame({'x_axis' : pca_transformed[:,0],
                                'y_axis' : pca_transformed[:,1],
                                'Cluster' : output_df['MeanShift']})
        

        for i in range(len(pca_df['Cluster'].unique())) :
            marker_i = pca_df[pca_df['Cluster'] == i].index
            plt.scatter(x = pca_df.loc[marker_i, 'x_axis'],
                        y = pca_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}')

        plt.title('MeanShift Clustering Visualization')
        plt.legend()

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

        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(output_df.drop(['GM'], axis=1))

        pca_df = pd.DataFrame({'x_axis' : pca_transformed[:,0],
                                'y_axis' : pca_transformed[:,1],
                                'Cluster' : output_df['GM']})
        

        for i in range(len(pca_df['Cluster'].unique())) :
            marker_i = pca_df[pca_df['Cluster'] == i].index
            plt.scatter(x = pca_df.loc[marker_i, 'x_axis'],
                        y = pca_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}')

        plt.title('Gaussian Mixture Clustering Visualization')
        plt.legend()
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

        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(output_df.drop(['DBSCAN'], axis=1))

        pca_df = pd.DataFrame({'x_axis' : pca_transformed[:,0],
                                'y_axis' : pca_transformed[:,1],
                                'Cluster' : output_df['DBSCAN']})
        

        for i in range(len(pca_df['Cluster'].unique())) :
            marker_i = pca_df[pca_df['Cluster'] == i].index
            plt.scatter(x = pca_df.loc[marker_i, 'x_axis'],
                        y = pca_df.loc[marker_i, 'y_axis'],
                        label = f'Cluster {i}')

        plt.title('DBSCAN Clustering Visualization')
        plt.legend()
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
