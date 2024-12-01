import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image
import random

from Point import Point

# Load grid map
image = Image.open("./data/map0.png").convert("L")
grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
# binarize the image
grid_map[grid_map > 0.5] = 1
grid_map[grid_map <= 0.5] = 0
# Invert colors to make 0 -> free and 1 -> occupied
grid_map = (grid_map * -1) + 1


def plot(grid_map, states, edges, path):
    plt.figure(figsize=(10, 10))
    plt.matshow(grid_map, fignum=0)
    # for i,v in enumerate(states):
    #     plt.plot(v.y, v.x, "+w")
    #     # plt.text(v.y, v.x, i, fontsize=14, color="w")

    for e in edges:
        plt.plot(
            [states[e[0]].y, states[e[1]].y],
            [states[e[0]].x, states[e[1]].x],
            "--g",
        )

    for i in range(1, len(path)):
        plt.plot(
            [states[path[i - 1]].y, states[path[i]].y],
            [states[path[i - 1]].x, states[path[i]].x],
            "r",
        )
    # Start
    plt.plot(states[0].y, states[0].x, "r*")
    # Goal
    plt.plot(states[-1].y, states[-1].x, "g*")


class RRT(Point):
    
    def __init__(self,gridmap,start,goal,sample_goal_probability=0.2,max_iter=10000,dq=10,edge_divisions=10,min_dist_to_goal=10,search_radius=20):

        self.gridmap = gridmap
        self.start = Point(start[0],start[1])
        self.goal = Point(goal[0],goal[1])
        self.sample_goal_probability = sample_goal_probability
        self.max_iter = max_iter
        self.dq = dq
        self.edge_divisions = edge_divisions
        self.min_dist_to_goal = min_dist_to_goal
        self.search_radius = search_radius

    def random_sample(self,gridmap,p):
        """Sample a random point in the gridmap.

        p: probability that we sample the goal point.

        """
        x = random.randint(0,gridmap.shape[0]-1)
        y = random.randint(0,gridmap.shape[1]-1)
    
        if random.uniform(0,1) < p:
            point = self.goal
        else:
            point = Point(x,y)
        return point

    def nearest_vertex(self,qrand,configs):
        min_distance = np.inf
        qnearest = Point(0,0)
        for i in range(len(configs)):
            if qrand.dist(configs[i]) < min_distance:
                min_distance = qrand.dist(configs[i])
                qnearest = configs[i]

        return qnearest
    
    def new_config(self,qrand,qnear,dq):

        direction = qnear.vector(qrand)
        distance = direction.norm()
        unit_vector = direction.unit()

        if distance == 0:
            return qnear

        step = unit_vector*min(dq,distance)

        return qnear.__add__(step)
        
    def is_segment_free(self,q1,q2,divisions):
            p1 = q1.numpy()
            p2 = q2.numpy()

            x = np.int_(np.linspace(p1,p2,divisions))
            x = [tuple(i) for i in x]

            for i in x:
                if grid_map[i] == 1:
                    return False

            return True

    def rrt(self):

        configs = []
        edges = []
        path = []

        configs.append(self.start)

        for i in range(self.max_iter):
            qrand = self.random_sample(self.gridmap,self.sample_goal_probability)
            qnear = self.nearest_vertex(qrand,configs)
            qnew = self.new_config(qrand,qnear,self.dq)

            if self.is_segment_free(qnear,qnew,self.edge_divisions):
                configs.append(qnew)
                
                idx_near = configs.index(qnear)
                idx_new = configs.index(qnew)

                edges.append((idx_near,idx_new))

                if qnew.dist(self.goal) < self.min_dist_to_goal and self.is_segment_free(self.goal,qnew,self.edge_divisions):
                    configs.append(self.goal)
                    edges.append((idx_near,configs.index(self.goal)))

                    edges.reverse()
                    path = [edges[0][1]]
                    next_v = edges[0][0]
                    i = 1
                    while next_v != 0:
                        while edges[i][1] != next_v:
                            i += 1
                        path.append(edges[i][1])
                        next_v = edges[i][0]
                    path.append(0)
                    edges.reverse()
                    path.reverse()

                    print("Number of iterations: ", i)
                    return configs, edges, path
        return None
    
    def smooth(self,configs,edges,path):
        j = -1
        i = 0
        new_path = [path[-1]]

        while new_path[0]!=0:
            if self.is_segment_free(configs[path[i]],configs[j],self.edge_divisions):
                new_path = [path[i]] + new_path
                j = path[i]
                i = 0
            else:
                i += 1

        return configs,edges,new_path

    def rrt_star(self):
        configs = []
        edges = []
        path = []

        configs.append(self.start)

        # Cost of reaching the start node (this is typically 0)
        costs = {}
        costs[self.start] = 0

        for i in range(self.max_iter):
            qrand = self.random_sample(self.gridmap,self.sample_goal_probability)
            qnear = self.nearest_vertex(qrand,configs)
            qnew = self.new_config(qrand,qnear,self.dq)
            
            if self.is_segment_free(qnear,qnew,self.edge_divisions):
                configs.append(qnew)
                costs[qnew] = costs[qnear] + qnew.dist(qnear)
                
                idx_near = configs.index(qnear)
                idx_new = configs.index(qnew)

                edges.append((idx_near,idx_new))

                qnew_neighbors = []
                for j in configs:
                    if qnew.dist(j)<=self.search_radius and self.is_segment_free(qnew,j,self.edge_divisions):
                        qnew_neighbors.append((j,configs.index(j)))
                        new_cost = costs[j] + qnew.dist(j)
                        if new_cost < costs[qnew]:
                            edges.remove((idx_near,idx_new))
                            
                            qmin = j
                            idx_min = configs.index(qmin)
                            edges.append((idx_min,idx_new))
                            costs[qnew] = new_cost
                            idx_near = idx_min
                
                for neighbor, idx_neighbor in qnew_neighbors:
                    new_neighbor_cost = costs[qnew] + qnew.dist(neighbor)
                    if new_neighbor_cost < costs[neighbor]:
                        edges = [edge for edge in edges if edge[1] != idx_neighbor] # remove previous edge
                        edges.append((idx_new,idx_neighbor))
                        costs[neighbor] = new_neighbor_cost

        return configs, edges, path

    
if __name__ == "__main__":

    graph = RRT(gridmap=grid_map,start=(10, 10) ,goal=(90, 70),
                sample_goal_probability=0.2,max_iter=1000,dq=10,edge_divisions=100,min_dist_to_goal=5)
    try:
        # configs, edges, path= graph.rrt()
        # total_distance = 0
        # plot(grid_map, configs, edges, path)
        # for i,j in zip(path,path[1:]):
        #     total_distance += configs[i].dist(configs[j])
        # print(total_distance)
        # print(len(path))
        # plt.show()

        # configs, edges, path = graph.smooth(configs,edges,path)

        # total_distance = 0
        # plot(grid_map, configs, edges, path)
        # for i,j in zip(path,path[1:]):
        #     total_distance += configs[i].dist(configs[j])
        # print(total_distance)
        # print(len(path))
        # plt.show()

        configs, edges, path = graph.rrt_star()

        # total_distance = 0
        plot(grid_map, configs, edges, path)
        # for i,j in zip(path,path[1:]):
        #     total_distance += configs[i].dist(configs[j])
        # print(total_distance)
        # print(len(path))
        plt.show()

    except TypeError:
        print("No path found")