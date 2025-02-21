import streamlit as st
from misc import *
import pandas as pd
from streamlit_folium import st_folium
import networkx as nx

df_att = pd.read_csv('data\\data_attractions_v5.csv')
df_h = pd.read_csv('data\\hotels_v1.csv')
h_df = pd.read_csv('data\\hotels_with_total_distances.csv')
selection_index = st.session_state.index
selected_hotel = df_h.loc[selection_index]
lat = selected_hotel['Latitude']
lon = selected_hotel['Longitude']

st.write('# Distance Metrics')

st.write('## Minimum spanning tree')
mst_desc = '''A **Minimum Spanning Tree (MST)** is a subset of the edges in a connected, weighted graph that connects all the vertices together without any cycles and with the minimum possible total distance. In simpler terms, it is the "shortest" way to connect all the nodes in a graph. MSTs are often used in network design, clustering, and optimization problems. Popular algorithms to find an MST include **Kruskal's algorithm**, **Prim's algorithm**, and **Borůvka's algorithm**.'''
st.write(mst_desc)
st.write("In this case, Kruskal's algorithm was used, with the selected hotel being the root node of the tree.")
st.write("The first step is implementing the Disjoint set union-find data structure.\nThen use Kruskal's algorithm to get the vertices and weighted edges of the MST")

code1 = '''# Minimum spanning tree to find the most efficient way to connect the hotels and attractions based on their geographical
# locations while minimizing the total travel distance.
# Prepare the coordinates for the algorithm
coords = df_att[['latitude', 'longitude']].to_numpy()
coords = [ np.array(val) for val in coords]
# Find the Minimum Spanning Tree using Kruskal's algorithm combined with the Disjoint Set Union Find data structure.
class DSUF:
    def __init__(self, coords):
      ids = list(range(len(coords)))
      self.Parent = dict(zip(ids, ids))
      self.Rank = dict(zip(ids, np.zeros(len(coords), dtype=int)))

    def distance(self, v1, v2):
      return sum(abs(v2 - v1))

    def edges(self, coords):
      N = len(coords)
      edges = []
      for i in range(N):
        for j in range(i + 1, N):
          dist = self.distance(coords[i], coords[j])
          edges.append((i, j, dist))
          edges.append((i, j, dist))
      edges = sorted(edges, key= lambda x: x[-1], reverse= False)
      print(f"edges: {edges}")
      return edges

    def find(self, x):
      if self.Parent[x] != x:
        return(self.find(self.Parent[x]))
      else:
        return x

    def union(self, x, y):
      if self.Rank[x] == self.Rank[y]:
        self.Parent[self.find(x)] = self.find(y)
        self.Rank[y] += 1
      elif self.Rank[x] > self.Rank[y]:
        self.Parent[self.find(y)] = self.find(x)
      else:
        self.Parent[self.find(x)] = self.find(y)

    def Kruskals(self, edges, n_vert):
      res = []
      cont = 0
      while (len(res) != (n_vert -1)):
        if len(edges) == 0:
          break
        edge = edges.pop(0)
        fr, to, wt = edge
        frp = self.find(fr)
        top = self.find(to)

        if frp != top:
          self.union(frp, top)
          res.append(edge)
        cont += 1
      return res

mst = DSUF(coords)
edges = mst.edges(coords)
n = len(coords)
resmst = mst.Kruskals(edges, n)
mst_coords = [(list(coords[i]), list(coords[j]), k) for i, j, k in resmst]'''

coords = df_att[['latitude', 'longitude']].to_numpy()
coords = [ np.array(val) for val in coords]
coords.append(np.array([lat, lon]))
mst = DSUF(coords)
edges = mst.edges(coords)
n = len(coords)
resmst = mst.Kruskals(edges, n)
mst_coords = [(list(coords[i]), list(coords[j]), k) for i, j, k in resmst]

st.code(code1, line_numbers= True)
st.write('Function to plot the MST on a map')

code2 = '''
# Function to plot MST on a map
def plot_mst_map(edges, midpoint):
    # Initialize a folium map centered around the midpoint of the first edge
    mst_map = folium.Map(location=midpoint, zoom_start=12)

    for edge in edges:
        point1, point2, dist = edge

        # Add the edge to the map
        folium.PolyLine(
            [point1, point2],
            color="blue",
            weight=2.5,
            tooltip=f"Distance: {dist} km",
        ).add_to(mst_map)

        # Add markers for the points
        folium.Marker(point1, popup=f"Point: {point1}").add_to(mst_map)
        folium.Marker(point2, popup=f"Point: {point2}").add_to(mst_map)

    return mst_map

# Find the midpoint of the coordintes
midpoint = (np.average(np.array(coords).T[0]), np.average(np.array(coords).T[1]))
mst_map = plot_mst_map(mst_coords, midpoint)'''

midpoint = (47.35123276680492, 8.573852378200021)
mst_map = plot_mst_map(mst_coords, midpoint)

st.code(code2, line_numbers= True)

st_folium(mst_map, use_container_width= True)

st.write('## Betweenness Centrality')
bc_desc = '''**Betweenness centrality** is a measure in network analysis that quantifies the importance of a node (or edge) based on its role as a bridge in the shortest paths between other nodes in the network. It calculates the fraction of all-pairs shortest paths that pass through a given node or edge. Nodes with high betweenness centrality are critical for communication and information flow in the network, as they often serve as intermediaries or bottlenecks. This metric is widely used in social networks, transportation systems, and biological networks to identify influential or critical points.'''
st.write(bc_desc)
st.write('First, the MST is converted to a networkx graph')

rows = np.zeros((len(coords), len(coords)))
for edge in resmst:
  x, y, dist = edge
  rows[x, y] = dist
  rows[y, x] = dist

# Calculate the adjacency matrix of the minimum spanning tree that gives the distance between the corresponding coordinates
df_adjacency_mat = pd.DataFrame(rows)
G_am = nx.from_pandas_adjacency(df_adjacency_mat)
nx.draw(G_am, with_labels= True)

code3 = '''# Find the largest distance between any two coordinates in the Minimum Spanning Tree
rows = np.zeros((len(coords), len(coords)))
for edge in resmst:
  x, y, dist = edge
  rows[x, y] = dist
  rows[y, x] = dist

# Calculate the adjacency matrix of the minimum spanning tree that gives the distance between the corresponding coordinates
df_adjacency_mat = pd.DataFrame(rows)
G_am = nx.from_pandas_adjacency(df_adjacency_mat)
nx.draw(G_am, with_labels= True)'''

st.code(code3, line_numbers= True)
st.pyplot(plt.gcf())

st.write('Finally, the betweenness centrality for each point in the graph is calculated and added to the final df')

code4 = '''# Calculate betweenness centrality for each node in the graph, which measures how often a node
# appears on the shortest paths between other nodes.
dist_resmst = {(i[0], i[1]): i[2] for i in resmst}
nx.set_edge_attributes(G_am, dist_resmst, 'distance')
betweenness_centr = nx.betweenness_centrality(G_am, weight= 'distance', normalized= False)
betweenness_centr
max_bc = max(betweenness_centr.values())
max_bc
betweenness_centr = list(betweenness_centr.values())
betweenness_centr = [i/max_bc for i in betweenness_centr]
# Add the betweenness centrality values to the dataframe
df_att['betweenness_centrality'] = betweenness_centr
df_att.head(5)
'''

st.code(code4, line_numbers= True)
st.dataframe(df_att.head(5), use_container_width= True)

st.write('## Distance Metrics Analysis')
st.write('For each hotel, a MST was calculated including all the atractions and said hotel. This value was stored in the total distance value of the hotels dataframe')

code8 = '''# Get the coordinates of the hotels in an array for suitable computations
hotel_coords = df_hotels[['Latitude', 'Longitude']].to_numpy()
hotel_coords = [ np.array(val) for val in hotel_coords]
# Obtain the total distance from each hotel to a set of key attractions
hotel_distances = []
for hotel in hotel_coords:
    total_distance = 0  # Sum of distances for this hotel
    # Loop over each attraction and calculate the distance to the hotel
    for attraction in mst_coords:
        attraction_coords = attraction[0]  # Attraction coordinates (first part of the tuple)
        hotel_coords = np.array(hotel)  # Hotel coordinates (from the hotel_data list)
        # Calculate the distance between the hotel and this attraction using the `distance` method
        total_distance += mst.distance(hotel_coords, attraction_coords)
    # Append the total distance for this hotel
    hotel_distances.append(total_distance)
# Create a new column in the hotels dataset with the previously calculated total distances
df_hotels['total_distance'] = hotel_distances
dt_hotels.head(5)
'''
st.code(code8, line_numbers= True)
st.dataframe(df_h.head(5), use_container_width= True)

code5 = '''# Calculate the minimum value from the total distance column
min_distance = min(h_df['total_distance'].to_list())
# Calculate the maximum value from the total distance column
max_distance = max(h_df['total_distance'].to_list())
# Boxplot of the total distances to identify outliers
plt.boxplot(h_df['total_distance'].tolist())
'''
plt.clf()
st.code(code5, line_numbers= True)
st.write('Min distance: 10.9356\nMax distance: 49.4136')
plt.boxplot(h_df['total_distance'].tolist())
st.pyplot(plt.gcf())

code6 = '''# Drop all the observations with a total distance greater than 15
h_df = h_df.drop(h_df[h_df.total_distance > 15].index)
# Boxplot of the total distances to check the last step
plt.boxplot(h_df['total_distance'].tolist())
'''

plt.clf()
plt.boxplot(df_h['total_distance'].tolist())

st.write('Remove outliers')
st.code(code6, line_numbers= True)
st.pyplot(plt.gcf())

st.write('calculate quartiles and recalculate edge values')
code7 = '''# Calculate the quantiles
h_df.total_distance.quantile([0.25, 0.5, 0.75])
# Get the maximum distance of all total distances (after dropping some observations)
max_distance = max(h_df['total_distance'].to_list())
# Funtion to compute a score based on the distance
def distance_score(dist): # works for distances between 11 -> 15
  return math.ceil(5 - dist%5)
h_df['distance_score'] = h_df.total_distance.apply(distance_score)
h_df.head()
'''

st.code(code7, line_numbers= True)
st.write('new max distance: 14.888')
# Calculate the quartiles
quartiles = h_df.total_distance.quantile([0.25, 0.5, 0.75])
columns = st.columns(2)
with columns[0]:
  st.write('Final hotels dataframe with total distance and distance score.')
  st.dataframe(df_h.head(5), use_container_width= True)
with columns[1]:
  st.write('Quartiles for total distance')
  st.write(quartiles)

st.write('Boxplot of betweenness_centrality')
plt.clf()
plt.boxplot(df_att['betweenness_centrality'].tolist())
st.pyplot(plt.gcf())

#st.write('Convert betweenness centrality to categorical variable for classification')
