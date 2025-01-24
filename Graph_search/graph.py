import csv
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import heapq
import numpy as np

# Load point data
data = []
with open("env_mx.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        polygon_id = int(row[0])
        x, y = float(row[1]), float(row[2])
        data.append((polygon_id, x, y))

# Load edges data
edges = []
with open("visibility_graph_env_mx.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        start_vertex = int(row[0])
        end_vertex = int(row[1])
        edges.append((start_vertex, end_vertex))

# Separate start, obstacles, and goal
start = data[0][1:]
goal = data[-1][1:]
obstacles = {}

for d in data[1:-1]:  # Exclude start and goal points
    poly_id, x, y = d
    if poly_id not in obstacles:
        obstacles[poly_id] = []
    obstacles[poly_id].append((x, y))

# Create obstacle polygons and plot them
obstacle_polygons = []
plt.figure(figsize=(8, 8))

# Start and Goal points
plt.scatter(*start, color="blue", label="Start", s=100)
plt.scatter(*goal, color="red", label="Goal", s=100)

# Draw each obstacle as a closed polygon
for points in obstacles.values():
    # Create and store each obstacle polygon
    polygon = Polygon(points)
    obstacle_polygons.append(polygon)
    
    # Plot the polygon by connecting its vertices
    x, y = zip(*points)
    plt.plot(x + (x[0],), y + (y[0],), color="black")  # Closing the polygon

# arrange all points
all_points = [start] + [pt for pts in obstacles.values() for pt in pts] + [goal]

# Plot visibility 
for edge in edges:
    start_vertex, end_vertex = edge
    p1, p2 = all_points[start_vertex], all_points[end_vertex]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', linewidth=1.0)

# Display the plot
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.title("Map with Start, Goal, Obstacles, and Visibility-based Connections")
plt.show()

# Define neighbors of each node
neighbors = {}
for v,n in edges:
    if v not in neighbors:
        neighbors[v] = []
    neighbors[v].append(n)

def euclidean_distance(q1,q2):
    distance = np.sqrt((q1[0]-q2[0])**2 + (q1[1]-q2[1])**2)
    return distance

def contains_cell(lst,neighbor):
    for _, cell_value in lst:
        if cell_value == neighbor:
            return True
    return False

def reconstruct_path(parent, current):
    path = [current]
    while current in parent.keys():
        current = parent[current]
        path = [current] + path
    return path[1:]

def a_star_graph(points,neighbors):

    open_set = []
    close_set = []

    vertecies = []
    for i in range(len(points)):
        vertecies.append(i)
    
    start = vertecies[0]
    goal = vertecies[-1]

    parent = {}
    parent[start] = None

    g_score = np.full(len(points),np.inf)
    g_score[start] = 0.0

    f_score = np.full(len(points),np.inf)
    f_score[start] = g_score[start] + euclidean_distance(points[0],points[-1])

    heapq.heappush(open_set,(f_score[start],start))

    while open_set:
        _ , current = heapq.heappop(open_set)
        close_set.append(current)

        if current == goal:
            return reconstruct_path(parent,current)
        
        for n in neighbors[current]:
            tentative_gscore = g_score[current] + euclidean_distance(points[n],points[current])
            if tentative_gscore < g_score[n]:
                parent[n] = current
                g_score[n] = tentative_gscore
                f_score[n] = tentative_gscore + euclidean_distance(points[n],points[goal])
                if not contains_cell(open_set,n):
                    heapq.heappush(open_set,(f_score[n],n))



path_vertecies = a_star_graph(all_points,neighbors)
print(path_vertecies)
path = []
for i in path_vertecies:
    path.append(all_points[i])
print(path)

plt.figure(figsize=(8, 8))
# Start and Goal points
plt.scatter(*start, color="blue", label="Start", s=100)
plt.scatter(*goal, color="red", label="Goal", s=100)
# Draw each obstacle as a closed polygon
for points in obstacles.values():
    # Create and store each obstacle polygon
    polygon = Polygon(points)
    obstacle_polygons.append(polygon)
    
    # Plot the polygon by connecting its vertices
    x, y = zip(*points)
    plt.plot(x + (x[0],), y + (y[0],), color="black")  # Closing the polygon
# Plot visibility 
for edge in edges:
    start_vertex, end_vertex = edge
    p1, p2 = all_points[start_vertex], all_points[end_vertex]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', linewidth=1.0)

for i in range(len(path)-1):
    start_vertex, end_vertex = path_vertecies[i] , path_vertecies[i+1]
    p1, p2 = all_points[start_vertex], all_points[end_vertex]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r', linewidth=1.0)

# Display the plot
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.title("Map with Start, Goal, Obstacles, and Visibility-based Connections")
plt.show()





    