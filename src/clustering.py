'''
군집화는 일단 Scaling이 되고 나서 진행됐다는 것을 가정합니다
때문에 여기서 사용하는 DataFrame은 Scaling이 된 상태의 DataFrame 입니다!
데이터마다 다르지만 저희 데이터는 Scaling이 필요한 데이터라고 판단했습니다
참고링크 : https://stats.stackexchange.com/questions/89809/is-it-important-to-scale-data-before-clustering

Clustering 목적 외에도 시각화를 PCA 압축을 통해 하는데, 이를 위해선 Scaling이 필요합니다.

추가적으로, 군집화의 앙상블이 생각보다 그렇게 자주 사용하는 기법은 아니라고 합니다.
하라면 할 수 있겠지만 잘 하지는 않는다고 하네요. 이유를 생각해보면 비지도학습이라

X라는 군집화 알고리즘이 A집단을 0으로 분류하고 B 집단을 1로 분류하고
Y라는 군집화 알고리즘이 A집단을 1로 분류하고 B 집단을 0으로 분류했다고 해서 이게 
의미가 없다는 것입니다. 

군집화의 결과로 뱉은 값은 비슷하다고 판단한 데이터들을 잘 모았다는 것이지 잘 "맞췄다"에 초점을 맞춘 것이 아니기 때문입니다.

군집이 label로 뱉은 값들을 단순히 보팅하면 같은 데이터를 잘 묶었음에도 불구하고
"표시한" 게 다르다는 이유로 Voting이 꼬일 수 있습니다.

해결 방법으로는 어차피 아래의 코드는 KMeans하고 붙이고~ MS하고 붙이고~ 하는 방식이라 
이전 알고리즘의 군집과 Feature를 고려하면서 군집의 Label을 붙이게 하는 방법이 생각나긴 하나
이는 다같이 생각해볼 문제인 것같습니다.
'''

'''
Todo List
1. inf값 추적
2. Processing print
3. 최적의 군집 형태 잡아두기
4. 시간이 많이 걸리는 것 같은 곳 
    (1) 군집 개수 추정
    (2) Bandwidth 추정
'''
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
    def __init__ (self, df : pd.DataFrame) :
        self.df = df
        self.cluster_models = {'KM' : self._KMeans_clustering,
                                'MS' : self._MeanShift_clustering,
                                'DB' : self._DBSCAN_clustering,
                                'GM' : self._GaussianMixture_clustering,
                                'HI' : self._Hierachical_clustering}

        self.fin_df = df.copy()

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
        
        model = KMeans(n_clusters = km_best_elbow, init = 'k-means++')
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

        model = GaussianMixture(n_components = gm_best_elbow)
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
        scaled_df = self._scaling(self.df)
        
        self._checking_elbows(scaled_df)
        
        self.cluster_models['KM'](scaled_df)
        self.cluster_models['MS'](scaled_df)
        self.cluster_models['DB'](scaled_df)
        self.cluster_models['GM'](scaled_df)
        self.cluster_models['HI'](scaled_df)
        
        self._calculate_scores()

        return self.fin_df
