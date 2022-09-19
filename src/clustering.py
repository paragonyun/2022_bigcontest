'''
군집화는 일단 Scaling이 되고 나서 진행됐다는 것을 가정합니다
때문에 여기서 사용하는 DataFrame은 Scaling이 된 상태의 DataFrame 입니다!
데이터마다 다르지만 저희 데이터는 Scaling이 필요한 데이터라고 판단했습니다
참고링크 : https://stats.stackexchange.com/questions/89809/is-it-important-to-scale-data-before-clustering

Clustering 목적 외에도 시각화를 PCA 압축을 통해 하는데, 이를 위해선 Scaling이 필요합니다.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score

# from yellowbrick.cluster import KElbowVisualizer

from sklearn.decomposition import PCA

class Clustering ():
    def __init__ (self, df : pd.DataFrame) :
        self.df = df
        self.cluster_models = {'KM' : self._KMeans_clustering,
                                'MS' : self._MeanShift_clustering,
                                'DB' : self._DBSCAN_clustering,
                                'GM' : self._GaussianMixture_clustering}

    def _checking_elbows(self) :
        return None 

    def _KMeans_clustering(self) :
        # TODO 여기 cluter어떻게 정할지 미리 생각해야됨..!
        output_df = self.df.copy()
        
        model = KMeans(n_clusters = 3, init = 'k-means++')
        model.fit(output_df)

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
        plt.show()

        return output_df

    def _MeanShift_clustering(self) :
        return

    def _DBSCAN_clustering(self) :
        return
    def _GaussianMixture_clustering(self) :
        return