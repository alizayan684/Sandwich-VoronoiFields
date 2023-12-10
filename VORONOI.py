import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

# Constants for the cost function
alpha = 0.5
d_max_O = 5.0
bounds = [(10, 10)]

# Function to calculate the cost function for the Voronoi Field potential
def calculate_voronoi_field_cost(x, y, obstacles):

    min_dist = min(distance.euclidean((x, y), obs) for obs in obstacles)
    dv = min(distance.euclidean((x, y), bound) for bound in bounds)

    # Handling potential division by zero errors
    epsilon = 1e-6
    cost = (alpha / (alpha + min_dist + epsilon)) * (dv / (dv + min_dist + epsilon)) * (pow((min_dist - d_max_O), 2) / 4)
    return cost

# A* Algorithm Implementation
def astar(start, goal, obstacles):
    open_set = [start]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: distance.euclidean(start, goal)+ calculate_voronoi_field_cost(start[0], start[1], obstacles)}

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]
            return path


        open_set.remove(current)
        valid_neighbors = get_neighbors(current, obstacles)
        for neighbor in valid_neighbors:
            tentative_g_score = g_score.get(current, float('inf')) + distance.euclidean(current, neighbor)
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + distance.euclidean(neighbor, goal) + calculate_voronoi_field_cost(neighbor[0], neighbor[1], obstacles)
                if neighbor not in open_set:
                    open_set.append(neighbor)

    return None

def get_neighbors(point, obstacles):
    neighbors = [(point[0] + dx, point[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx != 0 or dy != 0)]
    valid_neighbors = []
    for neighbor in neighbors:
        if neighbor not in obstacles and 0 <= neighbor[0] <= 10 and 0 <= neighbor[1] <= 10:  # Check if the neighbor is within bounds
            valid_neighbors.append(neighbor)
    return valid_neighbors

# Your existing code for grid creation
grid_size = 10
num_points = 100
x = np.linspace(0, grid_size, num_points)
y = np.linspace(0, grid_size, num_points)
points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

# Calculate obstacles (modified coordinates)


#First Case: Alot of obstacles
# obstacles = [
#     (1, 5), (2, 6), (2, 8),
#     (3, 5), (3, 7), (3, 9),
#     (5, 5), (6, 6), (6, 8),
#     (7, 3), (9, 1), (4, 4),
#     (5, 7), (8, 5), (5, 8),            # Additional obstacles
#     (2, 2), (7, 7), (1, 8),
#     (6,4 ), (8, 7), (4, 5),
#     (5,1 ), (5, 3),
# ]   #A lot of obstacles
# start_point = (0, 0)  # Define your start point
# end_point = (7, 9)    # Define your end point


# #Second Case: No Path
# obstacles=[(0,2),(1,2), (2,2) , (3,2),(4,2), (5,2), (6,2), (7,2), (8,2), (9,2) , (10,2)]   #No Path (Horizontal Line)
# start_point = (1, 1)  # Define your start point
# end_point = (8, 7)    # Define your end point


# Thrid Case : Empty Grid
# obstacles=[(3,9)]   #No Path (Horizontal Line)
# start_point = (1, 1)  # Define your start point
# end_point = (8, 7)    # Define your end point#


# # Fourth Case: Around the start point
obstacles=[(0,5) , (1,2) , (2,2) , (3,2) , (4,2) , (5,2), (6,2), (7,2), (8,2), (9,3) , (8,4)]
start_point = (1, 1)  # Define your start point
end_point = (8, 7)    # Define your end point

# # #Fifth Case: Around the end points
# obstacles=[(10,10) , (10,0) , (0,10) , (5,5)]
# start_point = (0, 0)  # Define your start point
# end_point = (7, 9)    # Define your end point




# A* pathfinding
start_time = time.time()
path = astar(start_point, end_point, obstacles)
end_time = time.time()

# Calculate Voronoi Field potential costs for each point
voronoi_costs = [calculate_voronoi_field_cost(point[0], point[1], obstacles) for point in points]

# Print distance value of the path
if path:
    total_distance = sum(distance.euclidean(path[i], path[i+1]) for i in range(len(path)-1))
    print("Distance of the path:", total_distance)
else:
    print("No valid path found!")

# Print time to produce the path
print("Time to produce the path:", end_time - start_time, "seconds")

# Plotting the Voronoi Field potential and A* path avoiding obstacles
plt.scatter(points[:, 0], points[:, 1], c=voronoi_costs, cmap='viridis')
plt.colorbar(label='Cost')
plt.scatter(*zip(*obstacles), color='red', marker='s', label='Obstacles', s=100)  # Set size for obstacles

if path:
    plt.plot(*zip(*path), marker='o', color='green', label='A Path')
    plt.scatter(*start_point, color='blue', label='Start', marker='^', s=100)  # Set size for start point
    plt.scatter(*end_point, color='purple', label='End', marker='v', s=100)    # Set size for end point
else:
    print("No valid path found!")

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Voronoi Field Potential Costs with A* Path Avoiding Obstacles')
plt.legend()
plt.grid(True)
plt.show()