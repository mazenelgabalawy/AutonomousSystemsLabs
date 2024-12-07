import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Point import Point
import sys

class RRT:
    def __init__(self,gridmap,dq,p,start,goal):
        self.gridmap = gridmap
        self.dq = dq
        self.p = p
        self.start = Point(start[0],start[1])
        self.goal = Point(goal[0],goal[1])

        # Find Valid Configurations
        self.valid_rows,self.valid_cols = np.where(gridmap == 0)

        self.configs = [self.start]
        self.parents = {}
        self.path = []

    def sample_random_point(self,valid_rows,valid_cols,probablity):
        
        rand_idx = np.random.choice(len(valid_cols)) # Select random index

        # Select random valid point
        x = valid_rows[rand_idx] 
        y = valid_cols[rand_idx]

        # Select goal with probablity p
        if np.random.uniform(0,1) < p:
            return self.goal
        else:
            return Point(x,y)

    def nearest_vertex(self):
        pass

    def get_qnew(self):
        
        pass

    def is_segment_free(self):
        pass

    def reconstruct_path(self):
        pass

    def run(self,n):
        for i in range(n):
            qrand = self.sample_random_point(self.valid_rows,self.valid_cols,self.p)
            self.get_qnew(self.configs,qrand)
            if self.configs[-1].dist(self.goa)==0:
                print("Path Found in {i} iterations")
                return self.configs, self.parents

        print("No Path Found")
        return [],[],np.inf

def plot():
    pass

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Not Enough arguments. Please use the following interface to run the program:\n")
        print("\tpython3 rrt.py path_to_grid_map_image K Δq p qstart_x qstart_y qgoal_x qgoal_y")
        sys.exit(0)
    
    # Get command line arguments
    map = sys.argv[1]
    max_iter = int(sys.argv[2])
    dq = float(sys.argv[3])
    p = float(sys.argv[4])
    start_x = float(sys.argv[5])
    start_y = float(sys.argv[6])
    goal_x = float(sys.argv[7])
    goal_y = float(sys.argv[8])

    start = (start_x,start_y)
    goal = (goal_x,goal_y)

    image = Image.open(map).convert("L")
    gridmap = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    gridmap[gridmap > 0.5] = 1
    gridmap[gridmap <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    gridmap = (gridmap * -1) + 1
    # Print Gridmap without any nodes
    plot(gridmap,start,goal,{},[])
    plt.title(f'{map}')
    plt.show()


    rrt = RRT(gridmap,dq,p,start,goal)
    nodes, edges, path = rrt.run(max_iter)