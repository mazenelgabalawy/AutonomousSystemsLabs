import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import heapq #library used to create priority queue
import copy

# Load grid map
image = Image.open('map3.png').convert('L')
grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1])/255
# binarize the image
grid_map[grid_map > 0.5] = 1
grid_map[grid_map <= 0.5] = 0
# Invert colors to make 0 -> free and 1 -> occupied
grid_map = (grid_map * -1) + 1

# # Show grid map
# plt.matshow(grid_map)
# plt.colorbar()
# plt.show()

connectivity_4 = [(0,-1),(-1,0),(1,0),(0,1)]
connectivity_8 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

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

def a_star(grid_map,q_start,q_goal,connectivity):
    open_set = []
    close_set = []

    parent = {} # dictionary to contain parent of each explored cell
    parent[q_start] = None

    gScore = copy.deepcopy(grid_map).astype(float)
    gScore.fill(np.inf) #initialize g-cost of all cesll as infinity
    gScore[q_start] = 0.0 # set g-cost at start as zero

    fScore = copy.deepcopy(grid_map).astype(float)
    fScore.fill(np.inf)
    fScore[q_start] = gScore[q_start] + euclidean_distance(q_start,q_goal)

    heapq.heappush(open_set,(fScore[q_start],q_start))

    while open_set:
        # choose cell with lowest fScore
        _, current = heapq.heappop(open_set)
        # add current cell to closed list
        close_set.append(current)

        # if goal is reached, return path
        if current == q_goal:
            path = reconstruct_path(parent,current)
            cost = gScore[path[-1]]
            
            return path, cost
        
        # Check neighbours of current cell 
        for i,j in connectivity:
            nx,ny = current[0] + i , current[1] +j
            if nx in range(np.shape(grid_map)[0]) and ny in range(np.shape(grid_map)[1]) and grid_map[nx,ny] == 0:
                tentative_gScore = gScore[current] + euclidean_distance(current,(nx,ny))
                if tentative_gScore < gScore[(nx,ny)]:
                    parent[(nx,ny)] = current
                    gScore[(nx,ny)] = tentative_gScore
                    fScore[(nx,ny)] = gScore[(nx,ny)] + euclidean_distance(q_goal,(nx,ny))
                    if not contains_cell(open_set,(nx,ny)):
                        heapq.heappush(open_set,(fScore[(nx,ny)],(nx,ny)))

    pass


start = (50, 90) 
goal = (375, 375)
path4 , cost4 = a_star(grid_map,start,goal,connectivity_4)
print(cost4)
print(path4)
path8 , cost8 = a_star(grid_map,start,goal,connectivity_8)
y4,x4 = zip(*path4)
y8,x8 = zip(*path8)
# y_explore4, x_explore4 = zip(*explored4)
plt.matshow(grid_map)
# plt.scatter(x_explore4,y_explore4,color='cyan')
plt.scatter(start[1],start[0],color='green')
plt.scatter(goal[1],goal[0],color='red')
plt.plot(x4, y4, color='blue', marker='o', linestyle='-', linewidth=1, markersize=1)
plt.plot(x8, y8, color='green', marker='o', linestyle='-', linewidth=1, markersize=1)
plt.legend()
plt.show()

