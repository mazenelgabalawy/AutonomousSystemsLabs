import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Point import Point
import sys


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

    def __init__(self,gridmap,max_iter,dq,p,start_x,start_y,goal_x,goal_y):
        self.gridmap = gridmap
        self.max_iter = max_iter
        self.dq = dq
        self.p = p
        self.start = Point(start_x,start_y)
        self.goal = Point(goal_x,goal_y)

    def sample_random_point(self,gridmap,p):

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

    def get_qnew(self,qrand,qnear,dq):
        direction = qnear.vector(qrand)
        distance = direction.norm()
        unit_vector = direction.unit()

        if distance == 0:
            return qnear

        step = unit_vector*min(dq,distance)
        return qnear.__add__(step)
    
    def is_segment_free(self,p1,p2):
        p1 = p1.numpy()
        p2 = p2.numpy()

        ps = np.int_(np.linspace(p1,p2,20))
        for x, y in ps:
            if self.gridmap[x, y] == 1:
                return False

        return True
    
    def reconstruct_path(self,configs,parents):
        path = [(self.goal.x,self.goal.y)]

        for parent, child in parents.items():
            if child != None:
                path.append(parent.to_str())

        return path

    def run(self):
        configs = []
        configs.append(self.start)
        parents = {}
        parents[self.start] = None

        for i in range(self.max_iter):
            qrand = self.sample_random_point(self.gridmap,self.p)
            qnear = self.nearest_vertex(qrand,configs)
            qnew = self.get_qnew(qrand,qnear,self.dq)
            if self.is_segment_free(qnear,qnew):
                configs.append(qnew)
                parents[qnear] = qnew

                if qnew.dist(self.goal)==0:
                    configs.append(self.goal)
                    path = self.reconstruct_path(configs,parents)
                    print("Path found in " + str(i) + " iterations")
                    return configs,parents, path
        
        return None


if __name__ == "__main__":
    gridmap = sys.argv[1]
    max_iter = int(sys.argv[2])
    dq = float(sys.argv[3])
    p = float(sys.argv[4])
    start_x = float(sys.argv[5])
    start_y = float(sys.argv[6])
    goal_x = float(sys.argv[7])
    goal_y = float(sys.argv[8])

    image = Image.open(gridmap).convert("L")
    gridmap = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    gridmap[gridmap > 0.5] = 1
    gridmap[gridmap <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    gridmap = (gridmap * -1) + 1

    rrt = RRT(gridmap,max_iter,dq,p,start_x,start_y,goal_x,goal_y)

    configs,parents,path = rrt.run()
    print(path)