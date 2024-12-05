import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Point import Point
import sys

def plot(gridmap,start,goal,tree,path,show_vertcies=False):

    """
    Visualizes the grid map, the path taken from start to goal, and optionally displays the vertices and their connections.

    Parameters:
    - gridmap (2D array): The grid map representing the environment where the pathfinding is occurring.
    - start (tuple): The (row, col) coordinates of the start position.
    - goal (tuple): The (row, col) coordinates of the goal position.
    - tree (dict): A dictionary mapping each node (vertex) to its parent node, used for visualizing the path taken.
    - path (list): A list of vertices (nodes) representing the path from start to goal.
    - show_vertcies (bool): If True, vertices will be shown on the plot with their indices. Default is False. Recommended to set to True only for small number of iternations

    Returns:
    - None: This function only generates a plot.
    
    Notes:
    - The grid map is visualized using a matrix with `matshow`.
    - The path is shown as a series of red lines connecting consecutive nodes in the path.
    - Start and goal positions are marked with distinct symbols (`r*` for start and `g*` for goal).
    - If `show_vertcies` is True, each vertex will be marked with a green plus sign, and its index will be displayed next to it.
    - The edges between connected vertices are drawn in white.
    """

    plt.figure(figsize=(10, 10))
    plt.matshow(gridmap, fignum=0)

    # Plot vertcies
    if show_vertcies:
        for i,v in enumerate(tree.keys()):
            plt.plot(v.y, v.x, "+g")
            plt.text(v.y, v.x, i, fontsize=14, color="w")

    # Plot Edges
    for parent,child in tree.items():
        if child != None:
            plt.plot(
                [parent.y,child.y],
                [parent.x,child.x],
                '-w'
            )
    # Plot given Path
    for i in range(1, len(path)):
        plt.plot(
            [path[i - 1].y, path[i].y],
            [path[i - 1].x, path[i].x],
            "r",
        )
    # Start
    plt.plot(start[1], start[0], "r*",markersize=10,label='Start')
    # Goal
    plt.plot(goal[1] ,goal[0], "g*",markersize=10,label='Goal')
    plt.legend()

def plot2(gridmap,start,goal,tree,original_path,smooth_path,show_vertcies=False):
    plt.figure(figsize=(10, 10))
    plt.matshow(gridmap, fignum=0)

    # Plot Vertcies
    if show_vertcies:
        for i,v in enumerate(tree.keys()):
            plt.plot(v.y, v.x, "+g")
            plt.text(v.y, v.x, i, fontsize=14, color="w")

    # Plot Edges
    for parent,child in tree.items():
        if child != None:
            plt.plot(
                [parent.y,child.y],
                [parent.x,child.x],
                '-w'
            )
    # Plot Original Path
    for i in range(1, len(original_path)):
        plt.plot(
            [original_path[i - 1].y, original_path[i].y],
            [original_path[i - 1].x, original_path[i].x],
            "r", label = "Original-Path" if i==1 else ""
        )
    # Plot Smooth Path
    for i in range(1, len(smooth_path)):
        plt.plot(
            [smooth_path[i - 1].y, smooth_path[i].y],
            [smooth_path[i - 1].x, smooth_path[i].x],
            "y", label = "Smooth-Path" if i==1 else ""
        )

    # Start
    plt.plot(start[1], start[0], "r*",markersize=10,label='Start')
    # Goal
    plt.plot(goal[1] ,goal[0], "g*",markersize=10,label='Goal')
    plt.legend()

class RRT:
    def __init__(self,gridmap,max_iter,dq,p,start,goal):
        self.gridmap = gridmap
        self.max_iter = max_iter
        self.dq = dq
        self.p = p
        self.start = Point(start[0],start[1])
        self.goal = Point(goal[0],goal[1])

    def sample_random_point(self,gridmap,p):

        valid_rows,valid_cols = np.where(gridmap == 0) # Get indcies for valid configurations
        rand_idx = np.random.choice(len(valid_cols)) # Select random index

        # Select random valid point
        x = valid_rows[rand_idx] 
        y = valid_cols[rand_idx]

        # Select goal with probablity p
        if np.random.uniform(0,1) < p:
            return self.goal
        else:
            return Point(x,y)
    
    def nearest_vertex(self,qrand,tree):
        min_distance = np.inf
        # Loop over all existing Points and find nearest to qrand
        for point in tree.keys():
            if qrand.dist(point) < min_distance:
                min_distance = qrand.dist(point)
                qnearest = point

        return qnearest

    def get_qnew(self,qrand,qnear,dq):
        direction = qnear.vector(qrand) # Steer in direction of qrand from qnear
        distance = direction.norm() # Calculate distance between two points
        unit_vector = direction.unit()
        
        # Check if two points are the same to avoid dividing by zero later
        if distance == 0:
            return qnear

        step = unit_vector*min(dq,distance) # Move in direction of rand with smaller distance between dq and distance
        return qnear.__add__(step)
    
    def is_segment_free(self,p1,p2):
        p1 = p1.numpy()
        p2 = p2.numpy()

        ps = np.int_(np.linspace(p1,p2,20)) # Divide the line into 20 points
        # Check all points on the line if they are invalid
        for x, y in ps:
            if self.gridmap[x, y] == 1:
                return False

        return True
    
    def reconstruct_path(self,tree,q):
        current = q # Set current node as end Goal
        path = [current] # Start path from goal
        path_cost = 0  # Initialize path cost to zero

        # Traverse the tree from the Goal node to the Start
        while current in tree.keys():
            current = tree[current]
            path.append(current)
        path.reverse() # Reverse path

        # Add cost of each edge to the path
        for i in range(2,len(path[1:])):
            path_cost += path[i-1].dist(path[i])

        return path[1:],path_cost

    def run(self):
        # Initialize the tree with the starting node that has no parent
        tree = {}
        tree[self.start] = None

        for i in range(self.max_iter):
            qrand = self.sample_random_point(self.gridmap,self.p) # Sample a point from the grid
            qnear = self.nearest_vertex(qrand,tree) # Find the nearest node to qrand
            qnew = self.get_qnew(qrand,qnear,self.dq) # Create a new node in the direction of qrand
            if self.is_segment_free(qnear,qnew):# Check if line between qnear and qnew doesn't pass through and obstacle
                tree[qnew] = qnear # Add qnew to the tree
                if qnew.dist(self.goal)==0: # If qnew is the Goal, return the path
                    tree[self.goal] = qnew
                    path,path_cost = self.reconstruct_path(tree,self.goal)
                    print("Path found in " + str(i) + " iterations")
                    return tree, path,path_cost
        print("No Path Found")
        return tree,[],np.inf
    
    def smooth(self,path):
        
        if path == []: # return empty path is no path was found
            return [],np.inf
        
        next_node = path[-1] #Set as goal
        i = 0
        smooth_path = [path[-1]] # Add goal to smooth-path
        smooth_path_cost = 0

        while smooth_path[-1]!=path[0]: # Check if start is reached
            if self.is_segment_free(path[i],next_node): # Check if a direct path is free from node i to next_node
                smooth_path.append(path[i]) 
                smooth_path_cost += path[i].dist(next_node)
                next_node = path[i] # Set next_node as node i and repeat
                i = 0
            else:
                i+=1
        smooth_path.reverse() # Reverse path 
        return smooth_path,smooth_path_cost

if __name__ == "__main__":

    # Get command line arguments
    gridmap = sys.argv[1]
    max_iter = int(sys.argv[2])
    dq = float(sys.argv[3])
    p = float(sys.argv[4])
    start_x = float(sys.argv[5])
    start_y = float(sys.argv[6])
    goal_x = float(sys.argv[7])
    goal_y = float(sys.argv[8])

    # Set start and goal points
    start = (start_x,start_y)
    goal = (goal_x,goal_y)

    image = Image.open(gridmap).convert("L")
    gridmap = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    gridmap[gridmap > 0.5] = 1
    gridmap[gridmap <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    gridmap = (gridmap * -1) + 1

    rrt = RRT(gridmap,max_iter,dq,p,start,goal)

    tree,path,path_cost = rrt.run()
    smooth_path,smooth_path_cost = rrt.smooth(path)


    # Original Path
    print("Total Path Cost: ", path_cost)
    print("Path length: ", len(path))
    print("Path to follow: ")
    print(*path,sep='\n')
    print('\n')
    plot(gridmap,start,goal,tree,path)
    plt.title("Original Path",fontsize = 18)
    plt.show()

    # Smooth Path
    print("Smooth Path Cost: ", smooth_path_cost)
    print("Smooth Path length: ", len(path))
    print("Smooth Path: ")
    print(*smooth_path,sep='\n')
    plot(gridmap,start,goal,tree,smooth_path)
    plt.title("Smooth Path",fontsize = 18)
    plt.show()

    # Overlay both paths
    plot2(gridmap,start,goal,tree,path,smooth_path)
    plt.title("Original-Path vs Smooth-Path", fontsize = 18)
    plt.show()

    