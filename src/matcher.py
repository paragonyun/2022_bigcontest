import pandas as pd
import os
from typing import List

class Matcher():
    
    def __init__(
        self,
        prep_user_df: pd.DataFrame,
        prep_loan_df: pd.DataFrame,
        prep_log_df: pd.DataFrame,
        prep_cofix_df: pd.DataFrame,
        matcher_config: dict,
        ):
        self.save_dir = matcher_config.get('save_dir')
        self.split_timestamp = pd.Timestamp(
            year=matcher_config.get('split_year'),
            month=matcher_config.get('split_month'),
            day=matcher_config.get('split_day')
        )
        self.split_col = 'loanapply_insert_time'
        self.user_df = prep_user_df
        self.loan_df = prep_loan_df
        self.log_df = prep_log_df
        self.cofix_df = prep_cofix_df
    
    
    def _match_loan_cofix(self):
        print('Loan과 Cofix 매칭중...')
        temp_df_list = []
        for _, cofix_row in self.cofix_df.iterrows():
            start = cofix_row['대상기간_시작']
            end = cofix_row['대상기간_끝'] + pd.Timedelta(hours=24)
            target_loan_df = self.loan_df[(start <= self.loan_df['loanapply_insert_time']) & (self.loan_df['loanapply_insert_time'] <= end)]
            target_loan_df['cofix_rate'] = cofix_row['cofix_rate']
            temp_df_list.append(target_loan_df)
        return pd.concat(temp_df_list).reset_index(drop=True)
    
    
    def _match_loan_user(self, loan_df: pd.DataFrame) -> List[pd.DataFrame]:
        print('Loan_Cofix와 User를 merge중...')
        matched_df = pd.merge(loan_df, self.user_df, left_on='application_id', right_on='application_id')
        return matched_df
    
        
    def _split_dataset(self, input_df: pd.DataFrame, time_col: str) -> tuple:
        print('Train(Valid)과 Test로 나누는 중...')
        train_valid_df = input_df[input_df[time_col] < self.split_timestamp].reset_index(drop=True)
        test_df = input_df[input_df[time_col] >= self.split_timestamp].reset_index(drop=True)
        return train_valid_df, test_df
    
    
    def _save_matched_df(self, matched_df:pd.DataFrame, save_file_name: str) -> None:
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, save_file_name)
        matched_df.to_feather(save_path)
        print(f'✅ matched dataset saved at ({save_path})')
    
    
    def run(self, save_mode=True):
        loan_cofix_df = self._match_loan_cofix()
        ml_df = self._match_loan_user(loan_cofix_df)
        ml_train_valid_df, ml_test_df = self._split_dataset(ml_df, self.split_col)
        if save_mode:
            self._save_matched_df(ml_train_valid_df, 'ml_train_valid.fth')
            self._save_matched_df(ml_test_df, 'ml_test.fth')
        return ml_train_valid_df, ml_test_df