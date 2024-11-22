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
    for i,v in enumerate(states):
        plt.plot(v.y, v.x, "+w")
        plt.text(v.y, v.x, i, fontsize=14, color="w")

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

def fill_path(vertices, edges):
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
    return vertices, edges, path


class RRT(Point):
    
    def __init__(self,gridmap,start,goal,sample_goal_probability,max_iter,dq,edge_divisions,min_dist_to_goal):

        self.gridmap = gridmap
        self.start = Point(start[0],start[1])
        self.goal = Point(goal[0],goal[1])
        self.sample_goal_probability = sample_goal_probability
        self.max_iter = max_iter
        self.dq = dq
        self.edge_divisions = edge_divisions
        self.min_dist_to_goal = min_dist_to_goal
        self.configs = []
        self.edges = []

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
        
    def is_segment_free(self,qnear,qnew,divisions):
            p1 = qnear.numpy()
            p2 = qnew.numpy()

            free = True

            x = np.int_(np.linspace(p1,p2,divisions))
            x = [tuple(i) for i in x]

            for i in x:
                if grid_map[i] == 1:
                    free = False

            return free

    def rrt(self):
        self.configs.append(self.start)

        for i in range(self.max_iter):
            qrand = self.random_sample(self.gridmap,self.sample_goal_probability)
            qnear = self.nearest_vertex(qrand,self.configs)
            qnew = self.new_config(qrand,qnear,self.dq)

            if self.is_segment_free(qnear,qnew,self.edge_divisions):
                self.configs.append(qnew)
                
                idx_near = self.configs.index(qnear)
                idx_new = self.configs.index(qnew)

                self.edges.append((idx_near,idx_new))

                if qnew.dist(self.goal) < self.min_dist_to_goal:
                    self.configs.append(self.goal)
                    self.edges.append((idx_near,self.configs.index(self.goal)))
                    print(i)
                    return self.configs,self.edges
        return None
    
    def smooth():
        pass

    def rrt_star():
        pass
    

try:
    graph = RRT(gridmap=grid_map,start=(10, 10) ,goal=(90, 70),
                sample_goal_probability=0.2,max_iter=1000,dq=10,edge_divisions=15,min_dist_to_goal=5)
    configs, edges= graph.rrt()

    states, edges, path = fill_path(configs, edges)
    plot(grid_map, states, edges, path)
    plt.show()
except TypeError:
    print("No path found")