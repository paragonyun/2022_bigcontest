from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np

class mice():
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def mice_1(self):
        df2 = self.df
        
        amt_df = df2[['existing_loan_cnt', 'existing_loan_amt']]
        
        def change_amt(cnt,amt):
            if (cnt == 0) & pd.isna(amt):
                return 0
            else:
                return amt
                
        df2['new_existing_loan_amt'] = amt_df.apply(lambda x : change_amt(x['existing_loan_cnt'],x['existing_loan_amt']), axis=1)
        df2 = df2.drop('existing_loan_amt', axis=1)
        
        ##mice
        mice_df = df2[['credit_score','연령대', '근속정도' ,'yearly_income','existing_loan_cnt', 'new_existing_loan_amt']]
        
        imputer = IterativeImputer(imputation_order='ascending', max_iter=5,random_state=42)
        imputed_dataset_train = imputer.fit_transform(mice_df)
        train_mice = pd.DataFrame(imputed_dataset_train, columns=mice_df.columns)
        
        credit_score = list(train_mice['credit_score'].apply(lambda x : min(1000, int(x))))
        existing_loan_amt = list(train_mice['new_existing_loan_amt'])
        age_group = list(train_mice['연령대'])
        
        df2['mice_credit_score'] = credit_score
        df2['mice_existing_loan_amt'] = existing_loan_amt
        df2['mice_age_group'] = np.round(age_group, 0)
        
        df2 = df2.drop(['credit_score','new_existing_loan_amt','연령대','income_per_credit','existing_loan_percent'], axis=1)
        
        df2['loanamt_per_income'] = df2['mice_existing_loan_amt'] / (1 + df2['yearly_income'])
        df2['loanamt_per_income'].fillna(0, inplace=True)
        
        df2['income_per_credit'] = df2['yearly_income'] / (1 + df2['mice_credit_score'])
    
        return df2
