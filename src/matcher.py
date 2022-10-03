import pandas as pd
import os
from typing import List, Tuple
from sklearn.utils import resample
from tqdm import tqdm
import pickle

class Matcher():
    
    def __init__(
        self,
        prep_user_df: pd.DataFrame,
        prep_loan_df: pd.DataFrame,
        prep_log_df: pd.DataFrame,
        prep_cofix_df: pd.DataFrame,
        matcher_config: dict,
        ):
        
        self.user_df = prep_user_df
        self.loan_df = prep_loan_df
        self.log_df = prep_log_df
        self.cofix_df = prep_cofix_df
        
        self.save_dir = matcher_config.get('save_dir')
        
        self.split_timestamp = pd.Timestamp(
            year=matcher_config.get('split_year'),
            month=matcher_config.get('split_month'),
            day=matcher_config.get('split_day')
        )
        
        self.split_col = 'loanapply_insert_time'
        self.label_col = 'is_applied'
        self.match_key_col = 'application_id'
        
        self.num_down_sampling = 10
    
    
    def _match_loan_cofix(self):
        print('Loan과 Cofix 매칭중...')
        temp_df_list = []
        for _, cofix_row in self.cofix_df.iterrows():
            start = cofix_row['대상기간_시작']
            end = cofix_row['대상기간_끝'] + pd.Timedelta(hours=24)
            target_loan_df = self.loan_df[(start <= self.loan_df[self.split_col]) & (self.loan_df[self.split_col] <= end)]
            target_loan_df['cofix_rate'] = cofix_row['cofix_rate']
            temp_df_list.append(target_loan_df)
        return pd.concat(temp_df_list).reset_index(drop=True)
    
    
    def _match_loan_user(self, loan_df: pd.DataFrame) -> List[pd.DataFrame]:
        print('Loan_Cofix와 User를 merge중...')
        matched_df = pd.merge(loan_df, self.user_df, left_on=self.match_key_col, right_on=self.match_key_col)
        return matched_df
    
        
    def _split_dataset(self, input_df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print('Train(Valid)과 Test로 나누는 중...')
        train_valid_df = input_df[input_df[time_col] < self.split_timestamp].reset_index(drop=True)
        test_df = input_df[input_df[time_col] >= self.split_timestamp].reset_index(drop=True)
        return train_valid_df, test_df
    
    
    def _save_matched_df(self, matched_df: pd.DataFrame, save_file_name: str) -> None:
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, save_file_name)
        matched_df.to_feather(save_path)
        print(f'✅ matched dataset saved at ({save_path})')
    
    
    def _save_matched_df_list(self, matched_df_list: List[pd.DataFrame], save_file_name: str) -> None:
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, save_file_name)
        with open(save_path, 'wb') as f:
            pickle.dump(matched_df_list, f, pickle.HIGHEST_PROTOCOL)
        print(f'✅ matched dataset saved at ({save_path})')
    
    
    def run(self, save_mode=True):
        loan_cofix_df = self._match_loan_cofix()
        ml_df = self._match_loan_user(loan_cofix_df)
        ml_train_valid_df, ml_test_df = self._split_dataset(ml_df, self.split_col)
        #ml_train_valid_df_list = self._make_down_sampling_fold(ml_train_valid_df)
        
        if save_mode:
            #self._save_matched_df_list(ml_train_valid_df_list, 'ml_train_valid.pkl')
            #for fold_idx, ml_train_valid_df in enumerate(ml_train_valid_df_list):
            #    ml_train_valid_df.columns = ml_train_valid_df.columns.astype(str)
            #    self._save_matched_df(ml_train_valid_df, f'ml_train_{fold_idx+10}.fth')  
            self._save_matched_df(ml_train_valid_df, 'ml_train_valid.fth')
            self._save_matched_df(ml_test_df, 'ml_test.fth')
            
        return ml_train_valid_df, ml_test_df