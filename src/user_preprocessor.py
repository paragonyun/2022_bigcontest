from .preprocessor import Preprocessor
import pandas as pd

## Emcoders
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder 

## 

class User_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        ## onehot encoder로 바뀔 cols
        self.onehot_cols = ['gender','income_type','employment_type','houseown_type',
                            'personal_rehabilitation_yn','personal_rehabilitation_complete_yn',
                            'national_health_insurance_type']

        ## ordinal encoder로 바뀔 cols
        self.ordinal_cols = ['birth_year','yearly_income']

        ## datetime type으로 바뀔 cols
        self.time_cols = ['insert_time', 'company_enter_month']

        ## 범주화될 cols
        self.categorical_cols = ['birth_year','yearly_income']

    '''
    전처리 순서 : 
    1. datetime type 대상 cols 전처리
    2. 파생변수 만들기
    3. 범주화 대상 cols 범주화 전처리
    4. OneHot Encoder 대상 cols 전처리
    5. Ordinal Encoder 대상 cols 전처리
    '''
    def __to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        print('datetime으로 바꾸는 중...')
        output_df = df.copy()

        ## datetime 대상이 될 cols를 돌면서
        for i in self.time_cols :
            ## object 가 아니라 int나 뭐 그런 식이면
            if output_df[i].dtype != 'object' :
                ## dtype을 object로 바꾸고
                output_df[i] = output_df[i].astype('object')
            
            ## 해당 column을 datetime type으로 바꿔줌
            output_df[i] = pd.to_datetime(output_df[i])

        ## 그렇게 바뀐 df return 
        return output_df

    def __derived_variable_maker(self, df: pd.DataFrame) :
        print('파생변수 생성 중...')
        output_df = df.copy()
        ## 신용점수 대비 연소득 : 연소득/신용점수
        output_df['income_per_credit'] = output_df['yearly_income'] / output_df['credit_score']

        ## 기대출비율 : 기대출금액 / 연소득
        output_df['existing_loan_percent'] = output_df['existing_loan_amt'] / output_df['yearly_income']

        return output_df
    
    
    # def __to_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
    #     # TODO
    #     # Birth 카테고리컬 화
    #     ## 이미 생년월일은 datetime type으로 변환된 상태임
    #     def ___birth_category(self, x) :
    #         # TODO
    #         ## 나이 계산한 결과 list를 따로 만들고 그걸 돌면서
    #         ## 연령대를 넣는 함수를 만들 예정
    #         return x



    #     output_df = df.copy()

        
        
    #     return output_df
    
    
    # def __sample_prep_3(self, df: pd.DataFrame) -> pd.DataFrame:
    #     # TODO
    #     print('loan prep3')
    #     return df
    
    
    def _preprocess(self) -> pd.DataFrame:
        prep_df = self.__to_datetime(self.raw_df)
        prep_df = self.__derived_variable_maker(prep_df)
        
        prep_df = self.__sample_prep_3(prep_df)
        # 여기서 index sort & reset 필수
        return prep_df