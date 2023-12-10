import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from queue import PriorityQueue
from matplotlib.path import Path
from shapely.geometry import Point, LineString, Polygon


# Define the motion space P as a rectangular region with dimensions [-Pi, 2 Pi] x [-Pi, Pi]
P = np.array([[-np.pi, -np.pi], [2*np.pi, np.pi]])

# Define smaller obstacles R
obstacle_size = 0.2
R = [
    np.array([[-np.pi + obstacle_size, -np.pi/2 + obstacle_size],
              [-np.pi + obstacle_size, 0 - obstacle_size],
              [-np.pi/2 - obstacle_size, 0 - obstacle_size],
              [-np.pi/2 - obstacle_size, -np.pi/2 + obstacle_size]]),
    np.array([[-np.pi/2 + obstacle_size, -np.pi/2 + obstacle_size],
              [-np.pi/2 + obstacle_size, 0 - obstacle_size],
              [0 - obstacle_size, 0 - obstacle_size],
              [0 - obstacle_size, -np.pi/2 + obstacle_size]]),
    np.array([[0 + obstacle_size, 0 + obstacle_size],
              [0 + obstacle_size, np.pi/2 - obstacle_size],
              [np.pi/2 - obstacle_size, np.pi/2 - obstacle_size],
              [np.pi/2 - obstacle_size, 0 + obstacle_size]]),
    np.array([[np.pi/2 + obstacle_size, np.pi/2 + obstacle_size],
              [np.pi/2 + obstacle_size, np.pi - obstacle_size],
              [np.pi - obstacle_size, np.pi - obstacle_size],
              [np.pi - obstacle_size, np.pi/2 + obstacle_size]]),
    np.array([[np.pi + obstacle_size, np.pi/2 + obstacle_size],
              [np.pi + obstacle_size, 0 - obstacle_size],
              [3*np.pi/2 - obstacle_size, 0 - obstacle_size],
              [3*np.pi/2 - obstacle_size, np.pi/2 + obstacle_size]]),
    np.array([[3*np.pi/2 + obstacle_size, 0 + obstacle_size],
              [3*np.pi/2 + obstacle_size, -np.pi/2 - obstacle_size],
              [2*np.pi - obstacle_size, -np.pi/2 - obstacle_size],
              [2*np.pi - obstacle_size, 0 + obstacle_size]])
]

# Define the navigable space S as the region difference between P and R
S = np.array([[-np.pi, -np.pi, 2*np.pi, 2*np.pi, -np.pi],
              [-np.pi, np.pi, np.pi, -np.pi, -np.pi]])

# Define the planning space C as a rectangular region with dimensions [0, 2 Pi] x [-2, 3]
C = np.array([[0, -2], [2*np.pi, 3]])

# Corrected potential function phi
def phi(node, goal):
    return np.linalg.norm(node - goal)

# Corrected stream function psi
def psi(point):
    x, y = point
    if y < -np.pi/2:
        return 0
    elif -np.pi/2 <= y < 0:
        return 1
    elif 0 <= y < np.pi/2:
        return 2
    elif np.pi/2 <= y < np.pi:
        return 3

# Define the mapping between S and C as a function of phi and psi
# Adjusted mapping function f
def f(point):
    return phi(point), psi(point)


# Define the system of ODEs for x and y
def ode_system(t, z):
    return [np.cos(z[0] - z[1]), 1]

def is_segment_valid(point1, point2, obstacle):
    # Check if the line segment between point1 and point2 intersects with the obstacle
    x1, y1 = point1
    x2, y2 = point2

    for i in range(len(obstacle)):
        x3, y3 = obstacle[i]
        x4, y4 = obstacle[(i + 1) % len(obstacle)]

        # Check for intersection
        if (
            min(x1, x2) <= max(x3, x4) and
            min(x3, x4) <= max(x1, x2) and
            min(y1, y2) <= max(y3, y4) and
            min(y3, y4) <= max(y1, y2) and
            ((x1 - x3) * (y4 - y3) - (y1 - y3) * (x4 - x3)) * ((x2 - x3) * (y4 - y3) - (y2 - y3) * (x4 - x3)) <= 0
        ):
            return False  # Intersection detected

    return True  # No intersection


# Define start and goal points
start_point = np.array([-np.pi/2, 1/2])
goal_point = np.array([3*np.pi/2, 1])

# Solve the ODEs with start point as initial conditions
solution = solve_ivp(ode_system, [0, 2*np.pi], start_point, method='RK45', dense_output=True)

# Plot the streamlines in both spaces using StreamPlot
x_range = np.linspace(-np.pi, 2*np.pi, 100)
y_range = np.linspace(-2, 3, 100)  # Adjusted y-axis limits for obstacles and planning space
X, Y = np.meshgrid(x_range, y_range)

plt.figure(figsize=(15, 6))

# Plot streamlines in the motion space
plt.subplot(1, 3, 1)
plt.streamplot(X, Y, np.cos(X - Y), np.ones_like(X), color='gray', density=1, linewidth=0.5)
plt.fill_between([-np.pi, 2*np.pi], -np.pi, np.pi, color='lightblue')
for obstacle in R:
    plt.fill(obstacle[:, 0], obstacle[:, 1], color='lightblue', edgecolor='black')
plt.plot(start_point[0], start_point[1], color='green', marker='o', label='Start Point')
plt.plot(goal_point[0], goal_point[1], color='red', marker='x', label='Goal Point')
plt.title('Motion Space')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(-2, 3)  # Set adjusted y-axis limits
plt.legend()
plt.grid(True)

# Plot obstacles in the motion space
plt.subplot(1, 3, 2)
plt.title('Obstacles in Motion Space')
plt.xlabel('X')
plt.ylabel('Y')
for obstacle in R:
    plt.fill(obstacle[:, 0], obstacle[:, 1], color='lightblue', edgecolor='black')
plt.ylim(-2, 3)  # Set adjusted y-axis limits
plt.grid(True)

# Plot streamlines in the planning space
plt.subplot(1, 3, 3)

# Evaluate the solution on a meshgrid
phi_range, psi_range = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(-2, 3, 100))
solution_points = solution.sol(phi_range.flatten()).T

plt.streamplot(phi_range, psi_range, solution_points[:, 0].reshape(100, 100), np.ones_like(phi_range),
               color='gray', density=1, linewidth=0.5)
plt.fill_between([0, 2*np.pi], -2, 3, color='lightgreen')
plt.plot([0, 2*np.pi], [1, 1], color='red', linewidth=2)
plt.plot([0, 2*np.pi], [2, 2], color='red', linewidth=2)
plt.title('Planning Space')
plt.xlabel('Phi')
plt.ylabel('Psi')
plt.grid(True)
plt.tight_layout()
plt.show()

# A* algorithm implementation
def astar(start, goal, obstacles):
    def heuristic(node):
        return np.linalg.norm(node - goal_indices)

    def is_valid(node, obstacles):
        for obstacle in obstacles:
            for poly in obstacle:
                if len(poly) < 3:
                    continue  # Skip invalid polygons

                obstacle_path = Path(poly)
                if obstacle_path.contains_point(node):
                    return False  # Node is inside an obstacle

        return True  # Node is not inside any obstacle

    def is_line_valid(start, end, obstacles):
        line = LineString([start, end])

        for obstacle in obstacles:
            obstacle_poly = Polygon(obstacle)
            if line.intersects(obstacle_poly):
                return False  # Intersection detected

        return True  # No intersection

    open_set = PriorityQueue()
    start_indices = np.round((start - C[:, 0]) / (C[:, 1] - C[:, 0]) * 99).astype(int)
    goal_indices = np.round((goal - C[:, 0]) / (C[:, 1] - C[:, 0]) * 99).astype(int)
    open_set.put((heuristic(start_indices), tuple(start_indices)))  # Use the heuristic for the initial priority
    came_from = {tuple(start_indices): None}
    g_score = {tuple(start_indices): 0}

    closed_set = set()  # Use a set for faster lookup

    while not open_set.empty():
        current_indices = open_set.get()[1]

        if current_indices in closed_set:
            continue  # Skip already evaluated nodes

        if np.linalg.norm(np.array(current_indices) - goal_indices) < 1:
            path_indices = [current_indices]
            while current_indices in came_from:
                current_indices = came_from[current_indices]
                if current_indices is not None:
                    path_indices.append(current_indices)
            path_indices.reverse()

            # Convert indices back to coordinates
            path = np.array(path_indices) / 99 * (C[:, 1] - C[:, 0]) + C[:, 0]
            return path

        closed_set.add(current_indices)

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue  # Skip the current node
                neighbor_indices = tuple(np.array(current_indices) + np.array([i, j]))

                neighbor = np.array(neighbor_indices) / 99 * (C[:, 1] - C[:, 0]) + C[:, 0]

                if (
                        neighbor_indices not in closed_set and
                        is_valid(neighbor, obstacles) and
                        is_line_valid(np.array(current_indices) / 99 * (C[:, 1] - C[:, 0]) + C[:, 0], neighbor,
                                      obstacles)
                ):
                    tentative_g_score = g_score[current_indices] + np.linalg.norm(
                        np.array(neighbor_indices) - np.array(current_indices))

                    if neighbor_indices not in g_score or tentative_g_score < g_score[neighbor_indices]:
                        g_score[neighbor_indices] = tentative_g_score
                        f_score = tentative_g_score + heuristic(np.array(neighbor_indices))
                        open_set.put((f_score, neighbor_indices))
                        came_from[neighbor_indices] = current_indices
    return np.array([])  # Return an empty array if no path is found


# Before A* Algorithm
print("Start Point:", start_point)
print("Goal Point:", goal_point)

# Find the A* path
astar_path = astar(start_point, goal_point, R)
print("A* Path:", astar_path)

# After A* Algorithm
if astar_path.size > 0:
    print("A* Path Found")
else:
    print("No Path Found")

# Plot the streamlines in both spaces using StreamPlot
plt.figure(figsize=(15, 6))

# Plot streamlines in the motion space
plt.subplot(1, 3, 1)
plt.streamplot(X, Y, np.cos(X - Y), np.ones_like(X), color='gray', density=1, linewidth=0.5)
plt.fill_between([-np.pi, 2*np.pi], -np.pi, np.pi, color='lightblue')
for obstacle in R:
    plt.fill(obstacle[:, 0], obstacle[:, 1], color='lightblue', edgecolor='black')
if astar_path.size > 0:
    plt.plot(astar_path[:, 0], astar_path[:, 1], color='orange', linewidth=2, label='A* Path')
plt.scatter(start_point[0], start_point[1], color='green', marker='o', label='Start Point')
plt.scatter(goal_point[0], goal_point[1], color='red', marker='x', label='Goal Point')
plt.title('Motion Space')
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(-2, 3)  # Set adjusted y-axis limits
plt.legend()
plt.grid(True)

# Plot obstacles in the motion space
plt.subplot(1, 3, 2)
plt.title('Obstacles in Motion Space')
plt.xlabel('X')
plt.ylabel('Y')
for obstacle in R:
    plt.fill(obstacle[:, 0], obstacle[:, 1], color='lightblue', edgecolor='black')
plt.ylim(-2, 3)  # Set adjusted y-axis limits
plt.grid(True)

# Plot streamlines in the planning space
plt.subplot(1, 3, 3)

# Evaluate the solution on a meshgrid
phi_range, psi_range = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(-2, 3, 100))
solution_points = solution.sol(phi_range.flatten()).T

plt.streamplot(phi_range, psi_range, solution_points[:, 0].reshape(100, 100), np.ones_like(phi_range),
               color='gray', density=1, linewidth=0.5)
plt.fill_between([0, 2*np.pi], -2, 3, color='lightgreen')
plt.plot([0, 2*np.pi], [1, 1], color='red', linewidth=2)
plt.plot([0, 2*np.pi], [2, 2], color='red', linewidth=2)
if astar_path.size > 0:
    plt.plot(astar_path[:, 0], astar_path[:, 1], color='orange', linewidth=2, label='A* Path')
plt.scatter(start_point[0], start_point[1], color='green', marker='o', label='Start Point')
plt.scatter(goal_point[0], goal_point[1], color='red', marker='x', label='Goal Point')
plt.title('Planning Space')
plt.xlabel('Phi')
plt.ylabel('Psi')
plt.grid(True)

plt.tight_layout()
plt.show()
