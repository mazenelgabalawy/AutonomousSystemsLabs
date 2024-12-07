import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Point import Point
import sys
import copy

def plot(gridmap,start,goal,tree,path,show_vertcies=False):
    """
    Visualizes the grid map, tree structure, and the path from start to goal, with optional vertex labeling.

    Parameters:
    - gridmap (2D array): The grid map representing the environment, where `0` indicates free spaces and non-zero values represent obstacles.
    - start (tuple): The (row, col) coordinates of the start position.
    - goal (tuple): The (row, col) coordinates of the goal position.
    - tree (dict): A dictionary where keys are parent vertices and values are child vertices, representing the tree structure.
    - path (list): A list of vertices (nodes) representing the path from start to goal.
    - show_vertcies (bool): If True, displays vertices with their indices. Default is False.

    Returns:
    - None: This function generates a visual plot but does not return any value.

    Notes:
    - The grid map is visualized using `matshow`, providing a background of the environment.
    - Tree edges are drawn as white lines between parent and child nodes.
    - The given path is shown as a series of red lines connecting the nodes in the path.
    - Start and goal positions are marked with distinct symbols (`r*` for start and `g*` for goal), and labeled in the legend.
    - If `show_vertcies` is True, each vertex in the tree is marked with a green plus sign (`+`), and its index is displayed next to it.
    - A legend is included to distinguish between the start and goal positions.
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


class RRTStar:
    def __init__(self,gridmap,max_iter,dq,p,max_search_distance,start,goal):
        self.gridmap = gridmap
        self.max_iter = max_iter
        self.dq = dq
        self.p = p
        self.max_search_distance = max_search_distance
        self.start = Point(start[0],start[1])
        self.goal = Point(goal[0],goal[1])
        self.valid_rows,self.valid_cols = np.where(self.gridmap == 0) # Get indcies for valid configurations


    def sample_random_point(self,valid_rows,valid_cols,p):
            
        """
        Samples a random point from the set of valid grid locations, with a probability of returning the goal point.

        Parameters:
        - valid_rows (array-like): A list or array of valid row indices for free spaces in the grid.
        - valid_cols (array-like): A list or array of valid column indices for free spaces in the grid.
        - p (float): The probability of returning the goal point instead of a randomly chosen point. Must be between 0 and 1.

        Returns:
        - Point: A randomly sampled point from the set of valid grid locations. The point will either be the goal (with probability `p`) or a random valid point (with probability `1 - p`).

        Notes:
        - The function selects a random index from the `valid_rows` and `valid_cols` arrays to determine a valid free point in the grid.
        - With probability `p`, the function returns the goal point instead of a random valid point.
        - This method assumes that `valid_rows` and `valid_cols` represent the valid free spaces in the grid.
        """

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
        """
        Finds the nearest vertex in the tree to a randomly sampled point.

        Parameters:
        - qrand (Point): The randomly sampled point for which the nearest vertex in the tree is to be found.
        - tree (dict): A dictionary representing the RRT tree, where keys are vertices (Points) and values are their parent vertices.

        Returns:
        - Point: The nearest vertex in the tree to the given random point `qrand`.

        Notes:
        - The function calculates the Euclidean distance between the random point and all vertices in the tree, selecting the vertex with the minimum distance.
        """

        min_distance = np.inf
        # Loop over all existing Points and find nearest to qrand
        for point in tree.keys():
            if qrand.dist(point) < min_distance and qrand.dist(point)!=0:
                min_distance = qrand.dist(point)
                qnearest = point

        return qnearest
    
    def get_qnew(self,qrand,qnear,dq):

        """
        Steers from a nearest vertex toward a random point, generating a new vertex within a step size.

        Parameters:
        - qrand (Point): The randomly sampled point toward which the new point is steered.
        - qnear (Point): The nearest vertex in the tree to `qrand`.
        - dq (float): The maximum allowable step size for the new vertex.

        Returns:
        - Point: A new vertex (`qnew`) generated by moving from `qnear` in the direction of `qrand` by a distance that is the smaller of `dq` or the actual distance to `qrand`.

        Notes:
        - If `qrand` and `qnear` are the same, `qnear` is returned to avoid division by zero.
        - The function calculates a unit vector in the direction of `qrand` from `qnear` and uses it to determine the new vertex by scaling the step size.
        """

        direction = qnear.vector(qrand) # Steer in direction of qrand from qnear
        distance = direction.norm() # Calculate distance between two points
        unit_vector = direction.unit()
        
        # Check if two points are the same to avoid dividing by zero later
        if distance == 0:
            return qnear

        step = unit_vector*min(dq,distance) # Move in direction of rand with smaller distance between dq and distance
        return qnear.__add__(step)
    
    def is_segment_free(self,p1,p2):

        """
        Checks whether the straight line segment between two points is free of obstacles.

        Parameters:
        - p1 (Point): The starting point of the line segment.
        - p2 (Point): The ending point of the line segment.

        Returns:
        - bool: `True` if the segment is free of obstacles, `False` otherwise.

        Notes:
        - The function divides the line segment into 20 equally spaced points.
        - Each point on the segment is checked against the grid map to ensure it lies in a free space (`gridmap[x, y] == 0`).
        - If any point on the segment lies in an obstacle (`gridmap[x, y] == 1`), the function returns `False`.
        - The coordinates of the points are converted to integers to match the grid map indices.
        """

        p1 = p1.numpy()
        p2 = p2.numpy()

        ps = np.int_(np.linspace(p1,p2,50)) # Divide the line into 20 points
        # Check all points on the line if they are invalid
        for x, y in ps:
            if self.gridmap[x, y] == 1:
                return False

        return True
    
    def reconstruct_path(self,tree,q):

        """
        Reconstructs the path from the goal to the start node using the RRT tree and calculates the total path cost.

        Parameters:
        - tree (dict): A dictionary representing the RRT tree, where keys are child vertices (Points) and values are their parent vertices.
        - q (Point): The goal node from which the path reconstruction begins.

        Returns:
        - tuple: A tuple containing:
            - path (list): A list of vertices (Points) representing the path from the start to the goal.
            - path_cost (float): The total cost of the reconstructed path, calculated as the sum of distances between consecutive points.

        Notes:
        - The function traces the tree from the goal node back to the start by following parent nodes.
        - The resulting path is reversed to order it from start to goal.
        - Path cost is calculated as the sum of Euclidean distances between consecutive vertices in the path.
        """

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

        return path[1:-1],path_cost

    
    def run(self):
        configs = []
        configs1 = configs
        configs.append(self.start)
        parents = {}
        parents1 = parents
        parents[self.start] = None

        # Cost of reaching the start node (this is typically 0)
        costs = {}
        costs[self.start] = 0

        reached_goal = False

        for i in range(self.max_iter):
            qrand = self.sample_random_point(self.valid_rows,self.valid_cols,self.p)
            qnear = self.nearest_vertex(qrand,parents)
            qnew = self.get_qnew(qrand,qnear,self.dq)
            
            if self.is_segment_free(qnear,qnew):
                configs.append(qnew)
                costs[qnew] = costs[qnear] + qnew.dist(qnear)
                qnew_neighbors = []
                qmin = qnear
                #cost Optimization
                for j in configs:
                    if qnew.dist(j) < self.max_search_distance and self.is_segment_free(qnew,j):
                        qnew_neighbors.append(j)
                        new_cost = costs[j] + qnew.dist(j)
                        if new_cost < costs[qnew]:
                            qmin = j
                            costs[qnew] = new_cost
                            
                if qnew not in parents:
                    parents[qnew] = qmin

                if qnew.dist(self.goal)==0 and reached_goal == False:
                    reached_goal = True
                    configs1 = copy.deepcopy(configs)
                    parents1 = copy.deepcopy(parents)
                    path1,pah1_cost = self.reconstruct_path(parents,qnew)
                    print(f"First path found after {i} iterations")
                    
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

        path,path_cost = self.reconstruct_path(parents,self.goal)
        return configs1, parents1, path1, configs, parents,path

if __name__ == "__main__":
    if len(sys.argv)!=10:
        print("Not Enough arguments. Please use the following interface to run the program:\n")
        print("\tpython3 rrt.py path_to_grid_map_image K Δq p max_search_radius qstart_x qstart_y qgoal_x qgoal_y")
        sys.exit(0)

    # Get command line arguments
    map = sys.argv[1]
    max_iter = int(sys.argv[2])
    dq = float(sys.argv[3])
    p = float(sys.argv[4])
    max_search_distance = float(sys.argv[5])
    start_x = float(sys.argv[6])
    start_y = float(sys.argv[7])
    goal_x = float(sys.argv[8])
    goal_y = float(sys.argv[9])

    # Set start and goal points
    start = (start_x,start_y)
    goal = (goal_x,goal_y)

    image = Image.open(map).convert("L")
    gridmap = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    # binarize the image
    gridmap[gridmap > 0.5] = 1
    gridmap[gridmap <= 0.5] = 0
    # Invert colors to make 0 -> free and 1 -> occupied
    gridmap = (gridmap * -1) + 1
    # # Print Gridmap without any nodes
    # plot(gridmap,)
    # plt.title(f'{map}')
    # plt.show()

    rrt_star = RRTStar(gridmap,max_iter,dq,p,max_search_distance,start,goal)
    configs1,parents1,path1,configs,parents ,path= rrt_star.run()
    plot(gridmap,start,goal,parents1,path1)
    plt.show()
    plot(gridmap,start,goal,parents,path)
    plt.show()
