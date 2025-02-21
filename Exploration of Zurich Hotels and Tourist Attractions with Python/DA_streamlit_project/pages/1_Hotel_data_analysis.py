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

df_raw = pd.read_csv('data\\ZurichHotels.csv')
df = pd.read_csv('data\\zurich_hotels_coordinates.csv')
hotels_url = 'https://www.hometogo.ch/'
buf = io.StringIO()

st.write('# Analysis of scraped hotel data')
st.write('## Data scraping')
st.write('The hotels data was scraped from the [home to go](%s) main search page'%hotels_url)
st.write('Obtained dataframe: ')
st.dataframe(df_raw, use_container_width= True)

st.write('## Data preparation')
st.write('### Data types handling')

code1 = '''# Change 'price' to numeric
df['price'] = df['price'].str.replace('CHF', '')
df['price'] = pd.to_numeric(df['price'])

# Change 'no_reviews' to numeric
df['no_reviews'] = df['no_reviews'].str.replace(r'\(|\)|reviews|review', '', regex=True)
df['no_reviews'] = pd.to_numeric(df['no_reviews'])
'''
st.code(code1, language= 'python', line_numbers= True)

st.write('### Encoding of categorical variables')
st.write('categories found for "review_cat" column: \n["Excellent", "Good", "Average", "Acceptable", NaN]')

st.write('Encoding: ')

code2 = '''# Encoding of categoriacal variable
dicc_review = {'Average' : 1, 'Acceptable' : 2, 'Good' : 3, 'Excellent' : 4}
df['review_cat'] = df['review_cat'].map(dicc_review)
df.head(3)
'''
df_raw['review_cat'] = pd.Categorical(df_raw['review_cat'])

dicc_review = {'Average' : 1, 'Acceptable' : 2, 'Good' : 3, 'Excellent' : 4}
df_raw['review_cat'] = df_raw['review_cat'].map(dicc_review)
st.code(code2, language= 'python', line_numbers= True)
st.dataframe(df_raw.head(3), use_container_width= True)

st.write('### Missing values and duplicates')
st.write('The raw dataframe only had 2 missing values in the review_cat column. Thus the df.dropna() function was used.')

st.write('## Geocoding')
st.write('using the GEOPY API, the hotels were geocoded with latitude and longitude.')

code3 = '''# Function to get coordinates
geolocator = Nominatim(user_agent="exercise", timeout=10)

def get_coordinates(location):
    try:
        geo = geolocator.geocode(location)
        if geo:
            return geo.latitude, geo.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return None, None

# Apply the function to our data
df[['Latitude', 'Longitude']] = df['address'].apply(lambda loc: pd.Series(get_coordinates(loc)))
df.head()        
'''
st.code(code3, language= 'python', line_numbers= True)
st.write('after applying the function, no missing values were found.')
st.write('result of geocoding with layers based on review category: ')
code4 = '''df.mp_plot.folium_map(
    lat_column="Latitude", long_column="Longitude", layer_column = "review_cat", zoom_start=10)
'''
st.code(code4, language= 'python', line_numbers= True)
map_preparation = folium.Map(location= (47.35123276680492, 8.573852378200021), zoom_start= 13)
lats = df['Latitude'].tolist()
lons = df['Longitude'].tolist()
dicc_review = {'Average' : 1, 'Acceptable' : 2, 'Good' : 3, 'Excellent' : 4}
df['review_cat'] = df['review_cat'].map(dicc_review)
df.dropna(subset=['review_cat'], inplace= True)
rev = df['review_cat'].tolist()
rev = [int(i) for i in rev]
colors = ['purple', 'blue', 'lightblue', 'red', 'green']
for i in range(len(rev)):
  folium.Marker((lats[i], lons[i]), icon= folium.Icon(color= colors[rev[i]])).add_to(map_preparation)
st_folium(map_preparation, use_container_width= True) 

df['price'] = df['price'].str.replace('CHF', '')
df['price'] = pd.to_numeric(df['price'])
df['no_reviews'] = df['no_reviews'].str.replace(r'\(|\)|reviews|review', '', regex=True)
df['no_reviews'] = pd.to_numeric(df['no_reviews'])

st.write('## Exploratory data analysis')
st.write('### Non-graphical EDA')
df.info(buf= buf)
df_prep_info = buf.getvalue()
st.write('information table from dataframe')
st.code(df_prep_info)

st.write('summary statistics: ')
code5 = '''# Summary statistics of numerical variables
df[['price', 'no_reviews']].describe()'''
st.code(code5, language= 'python', line_numbers= True)
st.write(df[['price', 'no_reviews']].describe())

st.write('### graphical EDA')

fig1, ax1 = plt.subplots()
# Histogram of prices
sns.histplot(df['price'], bins=20, palette= 'Blues', kde=True, ax= ax1)
ax1.set_xlabel('Price')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Prices with Density Line')
fig2, ax2 = plt.subplots()
# Price boxplot (to identify outliers)
sns.boxplot(df['price'], palette='Blues', ax= ax2)
ax2.set_title('Outliers in prices')
# Review barplot
fig3, ax3 = plt.subplots()
sns.countplot(x='review_cat', data=df, palette='Blues', ax= ax3)
ax3.set_title('Review categories')
ax3.set_xlabel('Category')
ax3.set_ylabel('Frequency')
# Boxplot of prices by category of review
fig4, ax4 = plt.subplots()
sns.boxplot(x='review_cat', y='price', data=df, palette='Blues', ax = ax4)
ax4.set_title('Prices by category of review')
ax4.set_xlabel('Review category')
ax4.set_ylabel('Price')
# Distribution of number of reviews
fig5, ax5 = plt.subplots()
sns.histplot(df['no_reviews'], bins=30, kde=True, palette='Blues', ax= ax5)
ax5.set_title('Distribution of Number of Reviews', fontsize=16)
ax5.set_xlabel('Number of Reviews', fontsize=14)
ax5.set_ylabel('Frequency', fontsize=14)
# Correlation heatmap of numeric variables
fig6, ax6 = plt.subplots()
corr_matrix = df[['price', 'no_reviews', 'review_cat']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax= ax6, cmap= 'Blues')
ax6.set_title('Heatmap of numeric variables')

columns = st.columns(2)
figs = [fig1, fig2, fig3, fig4, fig5, fig6]
for i in range(6):
    col = columns[i%2]
    with col:
        st.pyplot(figs[i])

st.write('## Modeling')

st.write('### ANOVA')

code6 = '''# One-way Anova to determine if there are statistically significant differences in the average price across the four review categories.
# Create subsets (groups)
c1 = df_hotels.loc[df_hotels['review_cat'] == 1]
c2 = df_hotels.loc[df_hotels['review_cat'] == 2]
c3 = df_hotels.loc[df_hotels['review_cat'] == 3]
c4 = df_hotels.loc[df_hotels['review_cat'] == 4]
# Create ANOVA
fvalue, pvalue = stats.f_oneway(c1['price'],
                                c2['price'],
                                c3['price'],
                                c4['price'])
print('F-value:', fvalue.round(3), 'p-value', pvalue.round(4))'''

st.code(code6, line_numbers= True)
st.write('output: F-value: 20.223 p-value 0.0')

str1 = '''Hypothesis:

$H_0$: There is no significant difference in the average prices between the four review categories.

$H_1$: At least one review category has a significantly different average price compared to the others.

Since $p$-value < α (0.05), there is enough evidence to reject the null hypothesis, which suggests a difference in prices between the groups.
'''
st.markdown(str1)

st.write('### Random Forest Classifier')

X_train, X_test, y_train, y_test = train_test_split(df[['no_reviews', 'price']],
                                                    df['review_cat'],
                                                    test_size=0.20,
                                                    random_state=42)


code7 = '''# Random forest classifier the review category of hotels based on the two features (number of reviews and price)
# Create train and test samples
X_train, X_test, y_train, y_test = train_test_split(df_hotels[['no_reviews', 'price']],
                                                    df_hotels['review_cat'],
                                                    test_size=0.20,
                                                    random_state=42)
# Show X_train
print('X_train:')
print(X_train.head(), '\n')

# Show y_train
print('y_train:')
print(y_train.head())'''

st.code(code7, line_numbers= True)
columns2 = st.columns(2)
with columns2[0]:
    st.write('X_train:')
    st.write(X_train.head())

with columns2[1]:
    st.write('Y_train:')
    st.write(y_train.head())

clf = DecisionTreeClassifier(random_state=20,
                             max_depth=3)
# Train the classification tree model
clf = clf.fit(X_train, y_train)
# Plot the decision tree
fig = plt.figure(figsize=(12,5))
tree_plot = tree.plot_tree(clf,
                   feature_names=list(X_train.columns),
                   class_names=['1', '2', '3', '4'],
                   filled=True,
                   fontsize=10,
                   label='root')
code8 = '''# Initialize the classification tree model
clf = DecisionTreeClassifier(random_state=20,
                             max_depth=3)
# Train the classification tree model
clf = clf.fit(X_train, y_train)
# Plot the decision tree
fig = plt.figure(figsize=(12,5))
tree_plot = tree.plot_tree(clf,
                   feature_names=list(X_train.columns),
                   class_names=['1', '2', '3', '4'],
                   filled=True,
                   fontsize=10,
                   label='root')
'''

st.code(code8, line_numbers= True)
st.pyplot(fig)

y_pred = clf.predict(X_test)

st.write('Confusion Matrix:')
st.code(confusion_matrix(y_test, y_pred))

st.write('Classification report')
st.code(classification_report(y_test, y_pred))

st.write('### K-means clustering')
X = StandardScaler().fit_transform(df[['price', 'no_reviews', 'review_cat']])
# Plot the data
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], s=10, color='darkred')

code9 = '''X = StandardScaler().fit_transform(df[['price', 'no_reviews', 'review_cat']])
# Plot the data
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], s=10, color='darkred')'''

st.code(code9, line_numbers= True)
st.pyplot(plt.gcf())

# Sum of squared distances of samples to their closest cluster center
distortions = []

# Range of k's
K = range(1,16,1)

# Loop to find the optimal k
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

# Elbow plot
plt.figure(figsize=(5,3))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')

code10 = '''# Sum of squared distances of samples to their closest cluster center
distortions = []
# Range of k's
K = range(1,16,1)
# Loop to find the optimal k
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)
# Elbow plot
plt.figure(figsize=(5,3))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
'''

st.code(code10, line_numbers= True)
st.pyplot(plt.gcf())

# Number of clusters
k = 5

# k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)

# Predict the values
y2 = kmeans.predict(X)

# Plot the clusters
plt.figure(figsize=(6,4))
plt.scatter(X[:, 0], X[:, 1], c=y2, s=10)

code11 = '''# Number of clusters
k = 5
# k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
# Predict the values
y2 = kmeans.predict(X)
# Plot the clusters
plt.figure(figsize=(6,4))
plt.scatter(X[:, 0], X[:, 1], c=y2, s=10)
plt.show()'''

st.code(code11, line_numbers= True)
st.pyplot(plt.gcf())