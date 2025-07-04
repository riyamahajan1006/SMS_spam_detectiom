import numpy as np
import pandas as pd
df = pd.read_csv('/content/spam.csv', encoding='ISO-8859-1')
df.info()
# drop extra columns
df.drop(columns=['Unnamed: 2',	'Unnamed: 3',	'Unnamed: 4'],inplace=True)
# rename columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
# encode target
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['target']=lb.fit_transform(df['target'])
#check for null values
df.isnull().sum()
# check for duplicates
df.duplicated().sum()
# if yes
df.drop_duplicates(keep='first',inplace=True)
df.duplicated().sum()