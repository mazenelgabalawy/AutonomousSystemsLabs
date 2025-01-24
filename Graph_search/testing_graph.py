import csv
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_polygons_with_start_goal(csv_file):
    polygons = defaultdict(list)

    # Read and parse the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            belongs_to_polygon, vertex_x, vertex_y = row[0], float(row[1].strip()), float(row[2].strip())
            polygons[belongs_to_polygon].append((vertex_x, vertex_y))

    # Add start and goal to each polygon
    result = {}
    for polygon_id, vertices in polygons.items():
        result[polygon_id] = vertices

    return result

# Usage example
csv_file = 'env_0.csv'
polygons_with_start_goal = extract_polygons_with_start_goal(csv_file)

# Print the results
# for polygon_id, vertices in polygons_with_start_goal.items():
#     print(f"Polygon {polygon_id}: {vertices}")
#     pass

# for key in polygons_with_start_goal.keys():
#     for key , vertex in polygons_with_start_goal.items():
#         fig = plt.figure()
#         plt.scatter(vertex[0][0],vertex[0][1])
#         plt.show()
# for key in polygons_with_start_goal.keys():
#     plt.scatter([vertex[int(key)][0] for key , vertex in polygons_with_start_goal.items()],[vertex[int(key)][1] for key , vertex in polygons_with_start_goal.items()])
#     plt.show()

# x = []
# y = []
# for key in polygons_with_start_goal.keys():
#     for elements in polygons_with_start_goal[key]:
#         x.append(elements[0])
#         y.append(elements[1])

# plt.scatter(x,y)
# plt.show()