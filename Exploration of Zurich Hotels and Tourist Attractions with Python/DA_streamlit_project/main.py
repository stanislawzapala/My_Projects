import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from misc import *

 # Setup streamlit header
st.set_page_config(page_title='Data Analytics Project')
st.write('# Data Analytics Project')
st.write('Authors:')
st.write('Adalía Fernanda Aneiros')
st.write('Diego Arturo González Juárez')
st.write('Stanislaw Zapala')

 # Read Data
hotels_df = pd.read_csv('data\\hotels_v1.csv')
attractions_df = pd.read_csv('data\\data_attractions_v5.csv')

hotels_df['centrality'] = hotels_df.apply(lambda row: '⭐'*row.distance_score, axis= 1)
hotels_df = hotels_df.sort_values(by= ['total_distance'], ascending=True, ignore_index= True)
display_hotels = hotels_df[['name', 'price', 'centrality']]

selection = st.dataframe(display_hotels, use_container_width= True, selection_mode='single-row', hide_index= True, on_select='rerun')

if 'index' not in st.session_state:
  st.session_state.index = 0

if not selection['selection']['rows']:
  selection_index = 0
else:
  st.session_state.index = selection['selection']['rows'][0]
  selection_index = st.session_state.index

selected_hotel = hotels_df.loc[selection_index]

name = selected_hotel['name']
address = selected_hotel['address']
description = selected_hotel['description']
review = selected_hotel['review_cat']
price = selected_hotel['price']
lat = selected_hotel['Latitude']
lon = selected_hotel['Longitude']
dist = selected_hotel['total_distance']
dist_score = selected_hotel['centrality']
n_revs = selected_hotel['no_reviews_num']

st.session_state.name = name
st.write(name)
st.write(price) #
st.write(dist_score)
st.write(description)
#hotels_df = hotels_df.sort_values(by= ['total_distance'], ascending=True, ignore_index= True)
selection_coords = np.array([lat, lon])

attractions_display = attractions_df[['name', 'address', 'telephone', 'email', 'url', 'betweenness_centrality', 'latitude', 'longitude']]
attractions_display['distance_to_selection'] = attractions_df.apply(lambda row: manhattan_distance(selection_coords, np.array([row.latitude, row.longitude])), axis= 1)
attractions_display = attractions_display.sort_values(by= ['distance_to_selection', 'betweenness_centrality'], ascending= [True, False], ignore_index= True)

st.write("## Recommended attractions nearby: ")
st.dataframe(attractions_display[['name', 'address', 'telephone', 'email', 'url']].head(20), use_container_width= True, hide_index= True)

lats = attractions_display.latitude.tolist()[:20]
lons = attractions_display.longitude.tolist()[:20]
midpoint = (sum(lats)/len(lats), sum(lons)/len(lons))

map = folium.Map(location= midpoint, zoom_start= 13)
#
folium.Marker(selection_coords, popup= f'Selected hotel: {name}', icon= folium.Icon(color= 'red')).add_to(map)
for i in range(20):
  coords = (attractions_display.loc[i, 'latitude'], attractions_display.loc[i, 'longitude'])
  name = attractions_display.loc[i, 'name']
  folium.Marker(coords, popup=name, icon= folium.Icon(color='blue')).add_to(map)

st_folium(map, use_container_width= True)

