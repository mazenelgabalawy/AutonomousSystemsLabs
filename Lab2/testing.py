import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import copy

# Load grid map
image = Image.open('map0.png').convert('L')
grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1])/255

# binarize the image
grid_map[grid_map > 0.5] = 1
grid_map[grid_map <= 0.5] = 0
# Invert colors to make 0 -> free and 1 -> occupied
grid_map = (grid_map * -1) + 1
# grid_map[0:3,:],grid_map[-1:-3,:] = 0,0
# grid_map[:,0:3],grid_map[:,-1:-3] = 0,0
# # Show grid map
# plt.matshow(grid_map)
# plt.colorbar()
# plt.show()

connectivity_4 = [(0,-1),(-1,0),(1,0),(0,1)]
connectivity_8 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

# def get_neighbours(point,no_neighbors=8):
    
#     neighbours = []

#     if no_neighbors == 4:
#         connectivity = [(0,-1),(-1,0),(1,0),(0,1)]
#     else:
#         connectivity = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

#     for i,j in connectivity:
#         neighbours.append((point[0] + i,point[1]+j))

#     return neighbours

def attraction_potential(grid_map,goal,zeta):
    attraction_map = np.zeros(np.shape(grid_map))

    rows = grid_map.shape[0]
    cols = grid_map.shape[1]
    for i in range(rows):
        for j in range(cols):
            attraction_map[i,j] = 0.5*zeta*((i-goal[0])**2 + (j-goal[1])**2)
    return attraction_map/(np.max(attraction_map))

def brushfire(grid_map,connectivity):

    distance_map = copy.deepcopy(grid_map)
    indices = np.where(grid_map==1)
    queue = list(zip(indices[0], indices[1]))

    while len(queue)!=0:
        current_x,current_y = queue[0]
        queue.pop(0)

        for i, j in connectivity:
            new_x, new_y = current_x + i , current_y + j
            if new_x in range(np.shape(grid_map)[0]) and new_y in range(np.shape(grid_map)[1]) and distance_map[new_x,new_y]== 0:
                distance_map[(new_x,new_y)] = distance_map[(current_x,current_y)] + 1
                queue.append((new_x,new_y))
    return distance_map

def repulsive_potential(grid_map,Q=10,connectivity=8):
    
    eta = 1
    repulsion_map = np.zeros(np.shape(grid_map))
    distance_map = brushfire(grid_map,connectivity)

    rows = grid_map.shape[0]
    cols = grid_map.shape[1]
    for i in range(rows):
        for j in range(cols):
            if distance_map[i,j] <= Q:
                repulsion_map[i,j] = 0.5*eta*(1/distance_map[i,j] - 1/Q)**2
    
    return repulsion_map/(np.max(repulsion_map))


def gradient_descent(potential_map, q_start,connectivity):
    path = [q_start]
    q = q_start
    temp_q = q
    
    max_iter = 500 #maximum number of iterations to ensure not getting stuck
    i = 0
    threshold = 10e-5
    gradient = float('inf')

    while abs(gradient)>threshold:
        gradient = float('inf')
        if i<=max_iter:
            for j,k in connectivity:
                new_x,new_y = q[0] + j , q[1] + k
                if new_x in range(np.shape(grid_map)[0]) and new_y in range(np.shape(grid_map)[1]):
                    new_gradient = potential_map[(new_x,new_y)] - potential_map[q]
                    if new_gradient<gradient and new_gradient<=0:
                        gradient = new_gradient
                        temp_q = (new_x,new_y)
            q = temp_q
            path.append(q)
            i +=1
        else:
            return path
    
    return path

def wave_front_planner(grid_map,q_goal,connectivity):

    wave_map = copy.deepcopy(grid_map)
    wave_map[q_goal] = 2
    queue = [q_goal]

    while len(queue)!=0:
        current_x , current_y = queue[0]
        queue.pop(0)

        for i, j in connectivity:
            new_x, new_y = current_x + i , current_y + j
            if new_x in range(np.shape(grid_map)[0]) and new_y in range(np.shape(grid_map)[1]):
                if wave_map[(new_x,new_y)] == 0 or wave_map[(new_x,new_y)]>wave_map[(current_x,current_y)]+1:
                    wave_map[(new_x,new_y)] = wave_map[(current_x,current_y)] + 1
                    queue.append((new_x,new_y))
                
    return wave_map

def find_path(wave_map,q_start,connectivity):
    path = [q_start] # initiazlie path from start point
    q = q_start
    goal_min = np.min(wave_map) + 1 #minimum of wave_map that is not an obstacle
    min_neighbour = float('inf')

    while min_neighbour!= goal_min:
        min_neighbour = float('inf') # set minimum neighbour value as infinite
        current_x , current_y = q # get current point
        for i,j in connectivity:
            new_x,new_y = current_x + i , current_y + j # get neighbour
            if new_x in range(np.shape(wave_map)[0]) and new_y in range(np.shape(wave_map)[1]): # check if neighbour is inise the bounds of the map
                if wave_map[(new_x,new_y)]<=min_neighbour and wave_map[(new_x,new_y)]!=1: # check if neighbour value is less than its surroundings and check if it it not an obstacle
                    min_neighbour = wave_map[(new_x,new_y)] # set value of new_neighbour
                    q = (new_x,new_y) # set q as neighbour with minimum_value
        path.append(q)

    return path

# grid_map = np.array([[0,0,0,0,0,0,0,0,0,0],
#                      [0,0,1,1,1,1,1,1,0,0],
#                      [0,0,1,1,1,1,1,1,1,1],
#                      [0,0,1,1,1,1,1,1,1,0],
#                      [0,0,1,1,1,1,1,1,1,0],
#                      [0,0,1,0,0,0,0,0,0,0],
#                      [0,0,1,0,0,0,0,0,0,0],
#                      [0,0,1,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0]
#                      ])



goal = (110,40)
start = (10,10)

# u_attraction = attraction_potential(grid_map,goal,zeta=0.0001)
# plt.matshow(u_attraction)
# plt.colorbar()
# plt.show()


# distance_map = brushfire(grid_map,connectivity_8)
# plt.matshow(distance_map)
# plt.colorbar()
# plt.show()

# u_repulsion = repulsive_potential(grid_map,Q=20,connectivity=connectivity_8)
# plt.matshow(u_repulsion)
# plt.colorbar()
# plt.show()

# total_u = u_attraction + u_repulsion
# plt.matshow(total_u)
# plt.colorbar()
# plt.show()

# path = gradient_descent(total_u,start,connectivity_8)
# y,x = zip(*path)
# plt.matshow(total_u)
# plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=1, markersize=1)
# plt.colorbar()
# plt.show()
wave_map = wave_front_planner(grid_map,goal,connectivity_4)
# plt.matshow(wave_map)
# plt.colorbar()
# plt.show()
path_wave = find_path(wave_map,start,connectivity_4)
y,x = zip(*path_wave)
plt.matshow(grid_map)
plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=1, markersize=1)
plt.colorbar()
plt.show()

# # def gradient_descent(potential_map, q_start,connectivity):
# #     path = [q_start]
# #     q = q_start
# #     temp_q = q
    
# #     max_iter = 500 #maximum number of iterations to ensure not getting into an infinite loop
# #     i = 0
# #     threshold = 10e-5 #Threshold to stop the algorithm
# #     gradient = float('inf') #initialize first gradient to infinity

# #     while abs(gradient)>threshold:
# #         gradient = float('inf') 
# #         if i<=max_iter:
# #             current_x , current_y = q
# #             for j,k in connectivity:
# #                 new_x,new_y = current_x + j , current_y + k #get neighbours according to connectivity
# #                 if new_x in range(np.shape(grid_map)[0]) and new_y in range(np.shape(grid_map)[1]): # check if neighbour is inside the gridmap
# #                     new_gradient = potential_map[(new_x,new_y)] - potential_map[q] # calculate gradient at neighbour point
# #                     if new_gradient<gradient and new_gradient<=0: # check if new_gradient is less than the current gradient
# #                         gradient = new_gradient
# #                         q = (new_x,new_y) # save the current neighbour
# #             path.append(q) # add current point to path
# #             i +=1
# #         else:
# #             return path
    
# #     return path

# def total_potential(U_att,U_rep):
#     return U_att + U_rep



# def test_maps(maps):
#     for key in maps:
#         # Define start and goal points
#         start = maps[key][0]
#         goal = maps[key][1]
        
#         # Load grid map
#         image = Image.open(key).convert('L')
#         grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1])/255
#         # binarize the image
#         grid_map[grid_map > 0.5] = 1
#         grid_map[grid_map <= 0.5] = 0
#         # Invert colors to make 0 -> free and 1 -> occupied
#         grid_map = (grid_map * -1) + 1
#         # Show grid map
#         plt.matshow(grid_map)
#         plt.title(key.replace(".png",""))
#         plt.colorbar()
#         plt.show()
#         #Attraction potential
#         u_attraction = attraction_potential(grid_map,goal,0.001)
#         plt.matshow(u_attraction)
#         plt.title("Attraction potential for " + key.replace(".png",""))
#         plt.colorbar()
#         plt.show()
#         #Repulsive potential
#         u_repulsion = repulsive_potential(grid_map,20,connectivity_8)
#         plt.matshow(u_repulsion)
#         plt.title("Repulsion potential for " + key.replace(".png",""))
#         plt.colorbar()
#         plt.show()
#         #Total Potential
#         u_total = total_potential(u_attraction,u_repulsion)
#         plt.matshow(u_total)
#         plt.title("Total potential for " + key.replace(".png",""))
#         plt.colorbar()
#         plt.show()

#         # Gradient Descent
#         path_4 = gradient_descent(u_total,start,connectivity_4)
#         y,x = zip(*path_4)
#         plt.matshow(u_total)
#         plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=1, markersize=1)
#         plt.title("Gradient Descent with connectivity 4")
#         plt.colorbar()
#         plt.show()

#         path_8 = gradient_descent(u_total,start,connectivity_8)
#         y,x = zip(*path_8)
#         plt.matshow(u_total)
#         plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=1, markersize=1)
#         plt.title("Gradient Descent with connectivity 8")
#         plt.colorbar()
#         plt.show()
#         # Wavefront Planner
#         # Connectivity 4
#         wave_map_4 = wave_front_planner(grid_map,goal,connectivity_4)
#         plt.matshow(wave_map_4)
#         plt.title("Wave Map using connectivity 4")
#         plt.colorbar()
#         plt.show()

#         path_wave_4 = find_path(wave_map_4,start,goal,connectivity_4)
#         y,x = zip(*path_wave_4)
#         plt.matshow(grid_map)
#         plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=1, markersize=1)
#         plt.title("Path using Wavefront using connectivity 4")
#         plt.colorbar()
#         plt.show()

#         #Connectivity 8
#         wave_map_8 = wave_front_planner(grid_map,goal,connectivity_8)
#         plt.matshow(wave_map_8)
#         plt.title("Wave Map using connectivity 8")
#         plt.colorbar()
#         plt.show()

#         path_wave_8 = find_path(wave_map_8,start,goal,connectivity_8)
#         y,x = zip(*path_wave_8)
#         plt.matshow(grid_map)
#         plt.plot(x, y, color='blue', marker='o', linestyle='-', linewidth=1, markersize=1)
#         plt.title("Path using Wavefront using connectivity 8")
#         plt.colorbar()
#         plt.show()


# maps = {'map0.png':[(10,10),(90,70)],
#         'map1.png':[(60,60),(90,60)],
#         'map2.png':[(8,31),(139,38)],
#         'map3.png':[(50,90),(375,375)]}
# test_maps(maps)