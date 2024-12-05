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


def plot(grid_map,graph,path):
    plt.figure(figsize=(10, 10))
    plt.matshow(grid_map, fignum=0)

    # Plot edges
    for parent in graph.keys():
        for child in graph[parent]:
            plt.plot([parent.y,child.y],
                     [parent.x,child.x],
                     "--g")
            
    # Plot configurations
    # for i,p in enumerate(graph.keys()):
    #     plt.plot(p.y,p.x,"+w")
    #     plt.text(p.y,p.x,i,fontsize=10,color="w")

    # Plot path
    for i in range(1,len(path)):
        plt.plot([path[i-1].y,path[i].y],
                 [path[i-1].x,path[i].x],
                 "r"
                 )
    
    ## Start
    start = list(graph)[0]
    plt.plot(start.y,start.x,"r*",markersize=10)
    ## Goal
    goal = list(graph)[-1]
    plt.plot(goal.y,goal.x,"b*",markersize=10)
            


def reconstruct_path(graph,current,goal,path):

    """
    Implement Depth-First-Search to tranverse the tree and find path from start to goal
    """
    path.append(current)

    if current == goal:
        return path
    
    for child in graph.get(current,[]):
        result = reconstruct_path(graph,child,goal,path)
        if result:
            return path
        
    path.pop() # remove current node if no path is found
    return None

class RRT:
    
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
    
    def nearest_vertex(self,sample,graph):
        min_distance = np.inf
        for key in graph.keys():
            if sample.dist(key) < min_distance:
                min_distance = sample.dist(key)
                qnear = key

        return qnear
    
    def new_config(self,p1,p2,dq):
        direction = p2.vector(p1)
        distance = direction.norm()
        unit_vector = direction.unit()

        if distance == 0:
            return p2

        step = unit_vector*min(dq,distance)
        return p2.__add__(step)
    
    def is_segment_free(self,p1,p2,divisions):
        p1 = p1.numpy()
        p2 = p2.numpy()

        ps = np.int_(np.linspace(p1,p2,divisions))
        for x, y in ps:
            if grid_map[x, y] == 1:
                return False

        return True
    
    def get_neighbors(self,graph,q,search_radius):
        neighbors = []
        for p in graph.keys():
            if q.dist(p) <= search_radius:
                neighbors.append(q)

        return neighbors

    def rrt(self):
        graph = {}
        graph[self.start] = []

        for i in range(self.max_iter):
            qrand = self.random_sample(self.gridmap,self.sample_goal_probability)
            qnear = self.nearest_vertex(qrand,graph)
            qnew = self.new_config(qrand,qnear,self.dq)

            if self.is_segment_free(qnear,qnew,self.edge_divisions):
                graph[qnear].append(qnew)
                graph[qnew] = []
                if qnew.dist(self.goal) <= self.min_dist_to_goal and self.is_segment_free(self.goal,qnew,self.edge_divisions):
                    graph[qnew].append(self.goal)
                    graph[self.goal] = []
                    print("Number of iterations: ", i)
                    return graph,self.start,self.goal
        return None
    
    def smooth(self,path):
        next_node = path[-1] #set as goal
        i = 0
        smooth_path = [path[-1]] # add goal to 

        while smooth_path[-1]!=path[0]: # check if start is reached
            if self.is_segment_free(path[i],next_node,self.edge_divisions):
                smooth_path.append(path[i])
                next_node = path[i]
                i = 0
            else:
                i+=1
        smooth_path.reverse()
        return smooth_path
    
    def rrt_star(self):
        graph = {}
        graph[self.start] = []
        costs = {}
        costs[self.start] = 0

        for i in range(self.max_iter):
            qrand = self.random_sample(self.gridmap,self.sample_goal_probability)
            qnear = self.nearest_vertex(qrand,graph)
            qnew = self.new_config(qrand,qnear,self.dq)

            if self.is_segment_free(qnear,qnew,self.edge_divisions):    
                costs[qnew] = costs[qnear] + qnew.dist(qnear)
                # Cost Optimization
                qmin = qnear
                neighbors = self.get_neighbors(graph,qnew,self.search_radius) # list to contain neighbors of qnew
                for neighbor in neighbors:
                    new_cost = costs[neighbor] + neighbor.dist(qnew)
                    if self.is_segment_free(qnew,neighbor,self.edge_divisions) and new_cost < costs[qnew]:
                        qmin = neighbor
                        costs[qnew] = new_cost
                
                if qnew not in graph:
                    graph[qnew] = []
                    graph[qmin].append(qnew)

                # if qnew == self.goal:
                #     print("Goal reached")
                #     return graph
                
                # # Rewiring
                # for neighbor in neighbors:
                #     cost_from_qnew = costs[qnew] + neighbor.dist(qnew)
                #     if cost_from_qnew < costs[neighbor]:
                #         old_parent = [key for key,values in graph.items() if neighbor in values][0]
                #         graph[old_parent].remove(neighbor)
                #         graph[qnew].append(neighbor)
                #         costs[neighbor] = cost_from_qnew

        # closest = self.start
        # min_distance = np.inf
        # for q in graph.keys():
        #     new_distance = self.goal.dist(q)
        #     if new_distance < min_distance and q!=self.goal:
        #         min_distance = new_distance
        #         closest = q
        # graph[closest].append(self.goal)
        # graph[self.goal] = []

        return graph,self.start,self.goal


if __name__ == "__main__":
    test1 = RRT(grid_map,(10,10),(90,70),sample_goal_probability=0.2,max_iter=500,dq = 5,edge_divisions=100,min_dist_to_goal=0)
    # try:
    #     # graph1,start1,goal1 = test1.rrt()
    #     # path = []
    #     # path = reconstruct_path(graph1,start1,goal1,path)
    #     # plot(grid_map,graph1,path)
    #     # plt.show()

    #     # smooth_path = test1.smooth(path)
    #     # plot(grid_map,graph1,smooth_path)
    #     # plt.show()
    # except:
    #     print("No path found")

    # graph1 = test1.rrt_star()

    # for i in range(10):
    #     graph1,_,_ = test1.rrt_star()
    #     for key,value in graph1.items():
    #         if key in value:
    #             print("We have loops")
    #             break
    #     print("No loops")
    #     plot(grid_map,graph1,[])
    #     plt.show()

    # graph1 = test1.rrt_star()
    # key_idx = list(graph1)
    # self_referncing = [key for key,values in graph1.items() if key in values]
    # self_referncing_idx = [key_idx.index(self_referncing[i]) for i in range(len(self_referncing))]
    # print(self_referncing_idx)
    # print([(self_referncing[i].y,self_referncing[i].x) for i in range(len(self_referncing))])
    # print(self_referncing)
    # print([graph1[i] for i in self_referncing])

    graph1,start1,goal1 = test1.rrt_star()
    # path = []
    # path = reconstruct_path(graph1,start1,goal1,path)
    plot(grid_map,graph1,[])
    plt.show()


