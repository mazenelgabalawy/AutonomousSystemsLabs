import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Point import Point




# Load grid map
image = Image.open("./data/map2.png").convert("L")
grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
# binarize the image
grid_map[grid_map > 0.5] = 1
grid_map[grid_map <= 0.5] = 0
# Invert colors to make 0 -> free and 1 -> occupied
grid_map = (grid_map * -1) + 1

def plot(grid_map,configs,parents,path):
    
    edges = []
    for parent,child in parents.items():
        if child != None:
            edges.append((configs.index(parent),configs.index(child)))

    plt.figure(figsize=(10, 10))
    plt.matshow(grid_map, fignum=0)
    # for i,v in enumerate(configs):
    #     plt.plot(v.y, v.x, "+w")
        # plt.text(v.y, v.x, i, fontsize=14, color="w")

    for e in edges:
        plt.plot(
            [configs[e[0]].y, configs[e[1]].y],
            [configs[e[0]].x, configs[e[1]].x],
            "--g",
        )
    
    for i in range(1, len(path)):
        plt.plot(
            [configs[path[i - 1]].y, configs[path[i]].y],
            [configs[path[i - 1]].x, configs[path[i]].x],
            "r",
        )
    # Start
    plt.plot(configs[0].y, configs[0].x, "r*")
    # Goal
    plt.plot(configs[-1].y, configs[-1].x, "g*")

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
        """Sample a random point in the gridmap in valid space.

        p: probability that we sample the goal point.

        """

        valid_rows,valid_cols = np.where(gridmap == 0) # get indicies for valid configs
        rand_idx = np.random.choice(len(valid_cols)) # select random index
    
        x = valid_rows[rand_idx]
        y = valid_cols[rand_idx]

        if np.random.uniform(0,1) < p:
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

    def reconstruct_path(self,configs,parents):
        current = configs[-1]
        path_objects = [current]

        while current in parents.keys():
            current = parents[current]
            path_objects = [current] + path_objects

        path_objects = path_objects[1:]
        path = [configs.index(p) for p in path_objects]

        return path
        
    def rrt(self):
        configs = []
        configs.append(self.start)
        parents = {}
        parents[self.start] = None

        for i in range(self.max_iter):
            qrand = self.random_sample(self.gridmap,self.sample_goal_probability)
            qnear = self.nearest_vertex(qrand,configs)
            qnew = self.new_config(qrand,qnear,self.dq)

            if self.is_segment_free(qnear,qnew,self.edge_divisions):
                configs.append(qnew)
                parents[qnew] = qnear

                if qnew.dist(self.goal) < self.min_dist_to_goal and self.is_segment_free(self.goal,qnew,self.edge_divisions):
                    configs.append(self.goal)
                    parents[self.goal] = qnew
                    print("Number of iterations: ", i)
                    return configs, parents
        return None
    
    def smooth(self,configs,parents,path):
        j = -1
        i = 0
        path_to_goal = [path[-1]]

        while path_to_goal[0]!=0:
            if self.is_segment_free(configs[path[i]],configs[j],self.edge_divisions):
                path_to_goal = [path[i]] + path_to_goal
                j = path[i]
                i = 0
            else:
                i += 1

        return configs,parents,path_to_goal

    def rrt_star(self):
        configs = []
        configs.append(self.start)
        parents = {}
        parents[self.start] = None

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
                qnew_neighbors = []
                qmin = qnear
                #cost Optimization
                for j in configs:
                    if qnew.dist(j) < self.search_radius and self.is_segment_free(qnew,j,self.edge_divisions):
                        qnew_neighbors.append(j)
                        new_cost = costs[j] + qnew.dist(j)
                        if new_cost < costs[qnew]:
                            qmin = j
                            costs[qnew] = new_cost

                if qnew not in parents:
                    parents[qnew] = qmin
                # Rewiring
                for neighbor in qnew_neighbors:
                    if neighbor != qmin:
                        new_neighbor_cost = costs[qnew] + qnew.dist(neighbor)
                        if new_neighbor_cost < costs[neighbor]:
                            parents[neighbor] = qnew
                            costs[neighbor] = new_neighbor_cost

        configs.append(self.goal)
        closest = self.start
        min_distance = float('inf')
        for i in configs:
            new_distance = self.goal.dist(i)
            if new_distance < min_distance and i!=self.goal:
                min_distance = new_distance
                closest = i
        parents[self.goal] = closest

        return configs, parents

    
if __name__ == "__main__":

    graph = RRT(gridmap=grid_map,start=(8, 31) ,goal=(139, 38),sample_goal_probability=0.2,
                max_iter=20000,dq=10,edge_divisions=20,min_dist_to_goal=0,search_radius=10)
    try:
        configs, parents= graph.rrt()
        path = graph.reconstruct_path(configs,parents)
        total_distance = 0
        for i,j in zip(path,path[1:]):
            total_distance += configs[i].dist(configs[j])
        print(total_distance)
        plot(grid_map,configs,parents,[])
        plt.show()
        #####
        configs, parents, smooth_path = graph.smooth(configs,parents,path)
        total_distance = 0
        for i,j in zip(smooth_path,smooth_path[1:]):
            total_distance += configs[i].dist(configs[j])
        print(total_distance)
        plot(grid_map, configs,parents,smooth_path)
        plt.show()

        # configs, parents = graph.rrt_star()
        # # for key,value in parents.items():
        # #     if key==value:
        # #         print("We have loops")
        # #         break
        # # print("No loops")
        # # path = graph.reconstruct_path(configs,parents)
        # # print(path)
        # plot(grid_map, configs,parents,[])
        # plt.show()

    #     # total_distance = 0
    #     # plot(grid_map, configs, edges, path)
    #     # for i,j in zip(path,path[1:]):
    #     #     total_distance += configs[i].dist(configs[j])
    #     # print(total_distance)
    #     # print(len(path))
    #     # plt.show()

    except TypeError:
        print("No path found")

    # for i in range(10):
    #     configs, parents = graph.rrt_star()
    #     for key,value in parents.items():
    #         if key==value:
    #             print("We have loops")
    #             break
    #     print("No loops")
        # plot(grid_map,graph1,[])
        # plt.show()