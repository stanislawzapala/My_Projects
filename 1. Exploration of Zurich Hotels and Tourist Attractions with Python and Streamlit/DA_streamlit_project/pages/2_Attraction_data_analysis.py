import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.metrics import RocCurveDisplay
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import regex as re

df_att = pd.read_csv('data\\data_attractions.csv')
df_att_fin = pd.read_csv('data\\data_attractions_v5.csv')
url = 'https://opendata.swiss/de/dataset/sehenswurdigkeiten-in-der-region-zurich-zurich-tourismus'

st.write('# Analysis of attractions data')
st.write('## Data obtension')
st.write('The database was obtained from the [opendata.swiss](%s) webpage'%url)
st.write('Obtained dataframe: ')
st.dataframe(df_att, use_container_width= True)

st.write('## Data preparation')
st.write('### Missing values')
st.write(df_att.isna().sum())

columns = df_att.keys()
na_count = df_att.isna().sum().tolist()
len_df = df_att.shape[0]
na_count = [val/len_df for val in na_count]
drop = ['@context']

for i, val in enumerate(na_count):
  if val > 0.3:
    drop.append(columns[i])

df_att.drop(columns= drop, inplace= True)

code1 = '''# Identification and dropping of columns that have a high proportion of missing values (more than 30% NA)
columns = df_att.keys()
na_count = df_att.isna().sum().tolist()
len_df = df_att.shape[0]
na_count = [val/len_df for val in na_count]
print('Columns to drop due to excess NA values: \n')
drop = ['@context']

for i, val in enumerate(na_count):
  if val > 0.3:
    drop.append(columns[i])
    print(columns[i])

df_att.drop(columns= drop, inplace= True)
df_att.head(5)'''

st.code(code1, line_numbers= True)
out1 = '''Columns to drop due to excess NA values: \n
@customType, 
tomasBookingId, 
zurichCardDescription, 
opens, 
openingHours, 
openingHoursSpecification.'''
st.write(out1)
st.dataframe(df_att.head(5), use_container_width= True)

st.write('### Handling language options of data')

def get_english_version(sub_df):
  result = re.search("'en': '(.*)', 'it'", sub_df)
  if result:
    return result.group(1)
  return None

language_options = ['copyrightHolder', 'name', 'disambiguatingDescription', 'description', 'titleTeaser', 'textTeaser', 'detailedInformation',
                     'price', 'specialOpeningHoursSpecification']
for col in language_options:
  print(col)
  df_att[col] = df_att[col].apply(get_english_version)
code2 = '''# Function to extract the English version from a column and apply it to some specific ones
# (most text columns have the names or texts in different languages)
def get_english_version(sub_df):
  result = re.search("'en': '(.*)', 'it'", sub_df)
  if result:
    return result.group(1)
  return None

language_options = ['copyrightHolder', 'name', 'disambiguatingDescription', 'description', 'titleTeaser', 'textTeaser', 'detailedInformation',
                     'price', 'specialOpeningHoursSpecification']
for col in language_options:
  print(col)
  df_att[col] = df_att[col].apply(get_english_version)
'''

st.code(code2, line_numbers= True)
st.dataframe(df_att.head(5), use_container_width= True)

st.write('The eval() function is applied to the address column to convert the json format into python dictionaries.')
code3 = '''# eval() applied
df_att['address'] = df_att['address'].apply(eval) 
# Convert nested JSON-like structure of the 'adress' column into a flat table
address_df = pd.json_normalize(df_att['address'])
# Combine address_df with the original dataframe
df_att = df_att.join(address_df)
df_att.head(5)
'''
st.code(code3, line_numbers= True)
st.dataframe(df_att_fin.head(5))