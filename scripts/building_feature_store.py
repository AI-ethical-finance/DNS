!pip install feast
!pip install scikit-learn
!pip install pandas
!pip install joblib
!pip install langchain

from sklearn import datasets
import pandas as pd

data_df = pd.read_csv('feature_data.csv', encoding ='utf-8')

data_df.to_parquet(path='data_df.parquet')

import pandas as pd
pd.read_parquet("/content/dns/feature_repo/data/data_df.parquet")

feast_repo_path = "/content/dns/feature_repo/"
store = FeatureStore(repo_path=feast_repo_path)

from langchain.prompts import PromptTemplate, StringPromptTemplate

template = """ '''{data_df}'''

 """


prompt = PromptTemplate.from_template(template)

response = get_completion(prompt)
print(response)

prompt_template = FeastPromptTemplate(input_variables=['Country'])

print(prompt_template.format(Country='Bolivia'))



