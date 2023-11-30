import pandas as pd
import numpy as np

class StarClassification:
    
    def __init__(self, db: pd.DataFrame, inputer, splitter, under_sampler, over_sampler, binary_encoder):
        self.db = db.copy()
        self.original_db = db.copy()
        
        self.under_db = None
        self.over_db = None
        
        self.inputer = inputer
        self.splitter = splitter
        self.under_sampler = under_sampler
        self.over_sampler = over_sampler
        self.binary_encoder = binary_encoder
        
    def determine_category(sp_type):
        roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
        sp_type = sp_type.upper()

        for numeral in roman_numerals:
            if numeral in sp_type:
                if numeral in ['I', 'II', 'III']:
                    return 'Giant'
                elif numeral in ['IV', 'V', 'VI', 'VII']:
                    return 'Dwarfs'

        return np.nan
    
    def outliers_remove(self, col: str):
        Q1 = self.db[col].quantile(0.25)
        Q3 = self.db[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outliers = self.db[(self.db[col] < lower_limit) | (self.db[col] > upper_limit)].copy()
        self.db = self.db[(self.db[col] >= lower_limit) & (self.db[col] <= upper_limit)]
        
        return outliers
    
    def input_values(self, col: str):
        self.db[col] = self.inputer.fit_transform(self.db[[col]])
        
    def encode_binary(self, col: str):
        self.db[col] = self.binary_encoder.fit_transform(self.db[col])
        
    def under(self, target: str):
        tmp = self.db.copy()
        tmp['ID'] = tmp.index

        balanced_data, balanced_target = self.under_sampler.fit_resample(tmp.drop(target, axis=1), tmp[target].copy())

        tmp2 = tmp.drop(balanced_data['ID'])
        balanced_data.drop('ID', axis=1, inplace=True)

        removed_registers_data = tmp2.drop([target, 'ID'], axis=1)
        removed_registers_target = tmp2[target].copy()

        self.under_db = pd.concat([balanced_data, balanced_target], axis=1)
        
        return removed_registers_data, removed_registers_target
    
    def over(self, target: [str]):
        train_set, test_set = self.split(target)
        
        balanced_data, balanced_target = self.over_sampler.fit_resample(train_set.drop(target, axis=1), train_set[target].copy())
        train_set = pd.concat([balanced_data, balanced_target], axis=1)
        
        return train_set, test_set
    
    def split(self, target: [str]):
        for i_train, i_test in self.splitter.split(self.db.copy(), self.db[target]):
            train_set = self.db.copy().iloc[i_train]
            test_set = self.db.copy().iloc[i_test]
            
        return train_set, test_set
    
    def split_under(self, target: [str]):
        for i_train, i_test in self.splitter.split(self.under_db.copy(), self.under_db[target]):
            train_set = self.under_db.copy().iloc[i_train]
            test_set = self.under_db.copy().iloc[i_test]
    
        return train_set, test_set