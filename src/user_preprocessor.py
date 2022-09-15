from .preprocessor import Preprocessor
import pandas as pd

## Emcoders
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder 

## datetime
from datetime import datetime

class User_Preprocessor(Preprocessor):
    
    def __init__(self, dataset: pd.DataFrame, prep_config: dict):
        super().__init__(dataset, prep_config)
        ## onehot encoder로 바뀔 cols
        self.onehot_cols = ['gender','income_type','employment_type','houseown_type',
                            'personal_rehabilitation_yn','personal_rehabilitation_complete_yn']

        ## ordinal encoder로 바뀔 cols
        self.ordinal_cols = ['birth_year','yearly_income']

        ## datetime type으로 바뀔 cols
        self.time_cols = ['insert_time', 'company_enter_month', 'birth_year']

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
            
            if i == 'birth_year' :

                output_df[i] = pd.to_datetime(output_df[i], format='%Y', errors='ignore')
                continue
            
            elif i == 'company_enter_month' :
                output_df[i] = pd.to_datetime(output_df[i], format='%Y%m%d', errors='ignore')
                continue

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
    
    
    def __to_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        print('카테고리화 시키는 중...')
        ## Birth 카테고리컬 화
        # 이미 생년월일은 datetime type으로 변환된 상태임
        def ___calculate_age(x) :
            this_yaer = datetime.now().year
            return this_yaer - x.year

        def ___birth_category(x) :
            if 0 <= x < 10 :
                return '0대'
            elif 10<= x <20 :
                return '10대'
            elif 20<= x <30 :
                return '20대'
            elif 30<= x <40 :
                return '30대'
            elif 40<= x <50 :
                return '40대' 
            elif 50<= x <60 :
                return '50대'
            elif 60<= x <70 :
                return '60대'
            else :
                return '70대 이상'

        output_df = df.copy()
        # 나이로 일단 변환
        output_df['age'] = output_df['birth_year'].apply(lambda x : ___calculate_age(x))

        # 연령대로 변환
        output_df['age_cat'] = output_df['age'].apply(lambda x : ___birth_category(x))

        ## yearly_income 카테고리 화 (일단 내부 quntile 기준으로 했습니다.)
        # HACK 그러나 EDA를 하면 알 수 있듯이, 연소득이라는 것이 격차가 매우 커서 이 기준에 대해 다시 생각해보는 게 좋을 것같습니다.
        q25 = output_df['yearly_income'].quantile(.25)
        q50 = output_df['yearly_income'].quantile(.50)
        q75 = output_df['yearly_income'].quantile(.75)       
        
        def ___income_category(x) :
            if x < q25 :
                return '1'
            elif q25 <= x <q50 :
                return '2'
            elif q50 <= x <q75 :
                return '3'
            else :
                return '4'

        output_df['yearly_income_cat'] = output_df['yearly_income'].apply(lambda x : ___income_category(x))

        return output_df

    def __to_one_hot(self, df) :
        print('원핫인코딩 중...')
        output_df = df.copy()
        for i in self.onehot_cols :
            ohe = OneHotEncoder(sparse=False)
            ohe_df = pd.DataFrame(ohe.fit_transform(output_df[[i]]),
                                columns = [f'{i}_'+ str(col) for col in ohe.categories_[0]])
            output_df = pd.concat([output_df.drop(columns=[i]),
                                ohe_df], axis=1)


        return output_df
    
    
    def _preprocess(self) -> pd.DataFrame:
        prep_df = self.__to_datetime(self.raw_df) # self.raw_df 는 preprocessor.py 에서 상속받음
        prep_df = self.__derived_variable_maker(prep_df)
        prep_df = self.__to_categorical(prep_df)
        prep_df = self.__to_one_hot(prep_df)
        # 여기서 index sort & reset 필수
        return prep_df