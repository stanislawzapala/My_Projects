import numpy as np
import matplotlib.pyplot as plt
import folium

def manhattan_distance(v1: np.array, v2: np.array):
  return round(sum(abs(v2 - v1)), 2)

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
      print(f'edges: {edges}')
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

def plot_mst(edges):
    plt.figure(figsize=(8, 8))
    plt.title("Minimum Spanning Tree")

    for edge in edges:
        point1, point2, dist = edge
        x_coords = [point1[0], point2[0]]
        y_coords = [point1[1], point2[1]]

        # Plot the edge
        plt.plot(x_coords, y_coords, 'b-', label=f"{dist:.2f}" if dist < 10 else None)

        # Mark the points
        plt.scatter(point1[0], point1[1], color='red', zorder=5)
        plt.scatter(point2[0], point2[1], color='red', zorder=5)

        # Annotate with distances
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2
        plt.text(mid_x, mid_y, f"{dist:.2f}", color="green", fontsize=9)

    # Adjust plot
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

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
        folium.Marker(point1, popup=f"Point: {point1}", icon= folium.Icon()).add_to(mst_map)
        folium.Marker(point2, popup=f"Point: {point2}", icon= folium.Icon()).add_to(mst_map)

    return mst_map