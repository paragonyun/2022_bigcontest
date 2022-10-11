import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Bank_info():
    
    
    def __init__(self, match_df: pd.DataFrame):
        self.bank_cols = [
            'bank_id',
            'product_id',
            'loan_rate',
            'cofix_rate',
            #'credit_score',
            'mice_credit_score',
            #'existing_loan_amt'
            'mice_existing_loan_amt'
        ]
        self.match_df = match_df
        
        
        
    def _cut_match_df(self, match_df: pd.DataFrame) -> pd.DataFrame:
        return match_df[self.bank_cols]
    
    
    
    def _add_loan_rate_per_cofix_rate_cols(self, df:pd.DataFrame) -> pd.DataFrame:
        df['loan_rate_per_cofix_rate'] = df['loan_rate'] / df['cofix_rate']
        return df
    
    
    
    def _create_bank_info_df(self, loan_bank_df:pd.DataFrame) -> pd.DataFrame:
        
        bank_id_list = []
        #num_applied_loan_list = []
        #num_product_list = []
        avg_loan_rate_list = []
        avg_cofix_rate_list = []
        avg_credit_score_list = []
        avg_existing_loan_amt_list = []
        avg_loan_rate_per_cofix_rate_list = []
        for bank_id, bank_df in loan_bank_df.groupby('bank_id'):
            bank_id_list.append(bank_id)
            #num_applied_loan_list.append(len(bank_df))
            #num_product_list.append(len(bank_df.product_id.unique()))
            avg_loan_rate_list.append(bank_df.loan_rate.mean())
            avg_cofix_rate_list.append(bank_df.cofix_rate.mean())
            #avg_credit_score_list.append(bank_df.credit_score.mean())
            avg_credit_score_list.append(bank_df.mice_credit_score.mean())
            #avg_existing_loan_amt_list.append(bank_df.existing_loan_amt.mean())
            avg_existing_loan_amt_list.append(bank_df.mice_existing_loan_amt.mean())
            avg_loan_rate_per_cofix_rate_list.append(bank_df.loan_rate_per_cofix_rate.mean())
            
        bank_info_df = pd.DataFrame({
            'bank_id': bank_id_list,
            #'num_applied_loan': num_applied_loan_list,
            # 'num_product': num_product_list,
            'avg_loan_rate': avg_loan_rate_list,
            'avg_cofix_rate': avg_cofix_rate_list,
            'avg_credit_score': avg_credit_score_list,
            'avg_existing_loan_amt': avg_existing_loan_amt_list,
            'avg_loan_rate_per_cofix_rate': avg_loan_rate_per_cofix_rate_list,
            }).astype(np.float64)
        
        return bank_info_df
    
    
    
    def _scale(self, bank_info_df: pd.DataFrame) -> pd.DataFrame:
        scaled_features = MinMaxScaler().fit_transform(bank_info_df.values) # 이상치를 죽이기 위해서 MinMaxScaler 사용
        scaled_bank_info_df = pd.DataFrame(scaled_features, index=bank_info_df.index, columns=bank_info_df.columns)
        scaled_bank_info_df['bank_id'] = scaled_bank_info_df.index # bank_id 까지 스케일되어서 다시 인덱스화
        return scaled_bank_info_df
    
    
    
    def _get_cluster_labels(self, bank_info_df: pd.DataFrame) -> list:
        clusterer_input_df = bank_info_df.iloc[:,1:-1] # bank_id, label 을 제외하고 fit 하기 위함
        kmeans_clusterer = KMeans(n_clusters=3) # 제 1, 2, 3, 금융권으로 구분하는게 목표이기 때문
        kmeans_clusterer.fit(clusterer_input_df) 
        cluster_labels = kmeans_clusterer.predict(clusterer_input_df)
        return cluster_labels
    
    
    
    def _plot_pca_2d(self, bank_info_df: pd.DataFrame, ori_cluster_labels: list) -> None:
        pca_input_df = bank_info_df.iloc[:,1:-1] # bank_id와 label을 제외하고 pca 하기 위함
        pca = PCA(n_components=2)
        printcipal_components = pca.fit_transform(pca_input_df)
        principal_df = pd.DataFrame(
            data=printcipal_components,
            columns = ['p_component_1', 'p_component_2']
            )
        principal_df['label'] = ori_cluster_labels
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('p_component_1', fontsize = 15)
        ax.set_ylabel('p_component_2', fontsize = 15)
        ax.set_title('Bank PCA result with label', fontsize=20)

        labels = [0, 1, 2]
        colors = ['r', 'g', 'b']
        for label, color in zip(labels, colors):
            label_idx = bank_info_df['label'] == label
            ax.scatter(
                principal_df.loc[label_idx, 'p_component_1'],
                principal_df.loc[label_idx, 'p_component_2'],
                c = color,
                s = 50
                )
        ax.legend(labels)
        ax.grid()
        fig.show()
        
        
        
    def _plot_pca_3d(self, bank_info_df: pd.DataFrame, ori_cluster_labels: list) -> None:
        pca_input_df = bank_info_df.iloc[:,1:-1] # bank_id와 label을 제외하고 pca 하기 위함
        pca = PCA(n_components=3)
        printcipal_components = pca.fit_transform(pca_input_df)
        principal_df = pd.DataFrame(
            data=printcipal_components,
            columns = ['p_component_1', 'p_component_2', 'p_component_3'],
            )
        principal_df['label'] = ori_cluster_labels
        total_var = pca.explained_variance_ratio_.sum() * 100
        fig = px.scatter_3d(
            principal_df,
            x='p_component_1',
            y='p_component_2',
            z='p_component_3',
            color='label',
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'p_component_1', '1': 'p_component_2', '2': 'p_component_3'}
        )
        fig.update_layout(
            autosize=False,
            width=1600,
            height=800
        )
        fig.show()
    
    
    
    def run(self, plot=False):
        loan_bank_df = self._cut_match_df(self.match_df)
        loan_bank_df = self._add_loan_rate_per_cofix_rate_cols(loan_bank_df)
        bank_info_df = self._create_bank_info_df(loan_bank_df)
        scaled_bank_info_df = self._scale(bank_info_df)
        cluster_labels = self._get_cluster_labels(scaled_bank_info_df)
        bank_info_df['bank_label'] = cluster_labels
        scaled_bank_info_df['bank_label'] = cluster_labels

        bank_label_df = pd.DataFrame(bank_info_df.bank_id, bank_info_df.bank_label).reset_index(drop=False)
        bank_label_match_df = self.match_df.merge(bank_label_df, how='inner', on='bank_id')

        if plot:
            self._plot_pca_2d(scaled_bank_info_df, cluster_labels)
            self._plot_pca_3d(scaled_bank_info_df, cluster_labels)

        return bank_label_match_df, bank_info_df