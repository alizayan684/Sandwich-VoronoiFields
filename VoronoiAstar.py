# A Star Algorithm Visualization implementation in Python

# Importing the required libraries
import math
from queue import PriorityQueue
import time

import numpy as np
import pygame
from scipy.spatial import Voronoi

# Importing all the colors from colors module
from colors import *

# Global Variables

# Width of the pygame window
WIDTH = 800

# Setting the Width and Height of the pygame window
WIN = pygame.display.set_mode((WIDTH, WIDTH))

# Setting the title the pygame window
pygame.display.set_caption("A Star Path Finding Algorithm")

# Constants for the cost function
alpha = 100
d_max_O = 9#9


# Function to calculate the distance to the nearest obstacle
def dO(x, y, obstacles):
    return np.min(
        np.array(
            [
                np.sqrt((obstacles[i].getPosition()[0] - x) ** 2 + (obstacles[i].getPosition()[1] - y) ** 2) 
                for i in range(len(obstacles))
            ]
        )
    )


# Function to calculate the distance to the nearest edge of the Generalized Voronoi Diagram (GVD)


def dV(x, y, gvd_edges):
    return np.min(
        np.array(
            [
                np.sqrt((gvd_edges[i][j][0] - x) ** 2 + (gvd_edges[i][j][1] - y) ** 2)
                for i in range(len(gvd_edges))
                for j in range(len(gvd_edges[i]))
            ]
        )
    )


# Voronoi Field potential function
def voronoi_field_potential(x, y, obstacles, gvd_edges):
    distance_to_obstacle = dO(x, y, obstacles)
    distance_to_gvd_edge = dV(x, y, gvd_edges)

    if distance_to_obstacle >= d_max_O:
        return 0
    else:
        return (
            (alpha / (alpha + distance_to_obstacle))
            * (distance_to_gvd_edge / (distance_to_obstacle + distance_to_gvd_edge))
            * ((distance_to_obstacle - d_max_O) ** 2 / (d_max_O**2))
        )


def calc_v_edges(voronoi):
    voronoi_edges = []
    for v_pair in voronoi.ridge_vertices:
        if v_pair[0] >= 0 and v_pair[1] >= 0:
            v0 = voronoi.vertices[v_pair[0]]
            v1 = voronoi.vertices[v_pair[1]]
            voronoi_edges.append([v0, v1])
    return voronoi_edges


# Class Node
class Node:
    def __init__(self, row, col, width, totalRows):
        self.row = row
        self.col = col
        self.width = width
        self.totalRows = totalRows
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []

    # Method which return the position of the node within the pygame window
    def getPosition(self):
        return self.row, self.col

    # Method which checks if the node is open
    def isOpen(self):
        return self.color == GREEN

    # Method which checks if the node is closed
    def isClosed(self):
        return self.color == RED

    # Method which checks if the node is a barrier
    def isBarrier(self):
        return self.color == BLACK

    # Method which checks if the node is the starting node
    def isStart(self):
        return self.color == ORANGE

    # Method which checks if the node is the ending node
    def isEnd(self):
        return self.color == TURQUOISE

    # Method which resets the node
    def reset(self):
        self.color = WHITE

    # Method which makes the node open
    def makeOpen(self):
        self.color = GREEN

    # Method which makes the node closed
    def makeClosed(self):
        self.color = RED

    # Method which makes the node a barrier
    def makeBarrier(self):
        self.color = BLACK

    # Method which makes the node the starting node
    def makeStart(self):
        self.color = ORANGE

    # Method which makes the node the ending node
    def makeEnd(self):
        self.color = TURQUOISE

    # Method which makes the node a part of the path from source to destination node
    def makePath(self):
        self.color = PURPLE

    # Method which draws the node to the pygame window
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    # Method which updates the neighbors of the node
    def updateNeighbors(self, grid):
        self.neighbors = []

        
        # North Neighbor
        if self.row > 0 and not grid[self.row - 1][self.col].isBarrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # South Neighbor
        if self.row < self.totalRows - 1 and not grid[self.row + 1][self.col].isBarrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # East Neighbor
        if self.col < self.totalRows - 1 and not grid[self.row][self.col + 1].isBarrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        # West Neighbor
        if self.col > 0 and not grid[self.row][self.col - 1].isBarrier():
            self.neighbors.append(grid[self.row][self.col - 1])
        
        # North east Neighbor
        if self.row > 0 and self.col < self.totalRows - 1 and not grid[self.row - 1][self.col + 1].isBarrier():
            self.neighbors.append(grid[self.row - 1][self.col + 1])
        
        # North west Neighbor
        if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].isBarrier():
            self.neighbors.append(grid[self.row - 1][self.col - 1])
        
        # South east Neighbor
        if self.row < self.totalRows - 1 and self.col < self.totalRows - 1 and not grid[self.row + 1][self.col + 1].isBarrier():
            self.neighbors.append(grid[self.row + 1][self.col + 1])
        
        # South west Neighbor
        if self.row < self.totalRows - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].isBarrier():
            self.neighbors.append(grid[self.row + 1][self.col - 1])


# Function which draws the path from the source to the destination node
def drawPath(predecessor, currentNode, draw):
    total = 0
    while currentNode in predecessor:
        temp = currentNode
        currentNode = predecessor[currentNode]
        total += heuristicFunction(temp.getPosition(), currentNode.getPosition())
        currentNode.makePath()
        draw()
    return total


# A Star Algorithm Function
def AStarAlgorithm(draw, grid, startNode, endNode, obstacles, voronoi):
    count = 0

    # OpenSet is a priority queue
    openSet = PriorityQueue()
    openSet.put((0, count, startNode))

    predecessor = {}

    # Global Score (gScore)
    gScore = {node: float("inf") for row in grid for node in row}
    gScore[startNode] = 0

    # Heuristic Score (fScore)
    fScore = {node: float("inf") for row in grid for node in row}
    fScore[startNode] = heuristicFunction(
        startNode.getPosition(), endNode.getPosition()
    ) + voronoi_field_potential(startNode.getPosition()[0],startNode.getPosition()[1] , obstacles= obstacles, gvd_edges=calc_v_edges(voronoi))

    openSetHash = {startNode}

    while not openSet.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = openSet.get()[2]
        openSetHash.remove(current)

        if current == endNode:
            total = drawPath(predecessor, endNode, draw)
            endNode.makeEnd()
            print(f"The End Node is {total} units away from the Source Node")
            return True

        for neighbor in current.neighbors:
            temp_gScore = gScore[current] + 1

            if temp_gScore < gScore[neighbor]:
                predecessor[neighbor] = current
                gScore[neighbor] = temp_gScore
                fScore[neighbor] = (
                    temp_gScore
                    + heuristicFunction(neighbor.getPosition(), endNode.getPosition())
                    + voronoi_field_potential(neighbor.getPosition()[0],neighbor.getPosition()[1], obstacles, gvd_edges= calc_v_edges(voronoi))*10
                )
                if neighbor not in openSetHash:
                    count += 1
                    openSet.put((fScore[neighbor], count, neighbor))
                    openSetHash.add(neighbor)
                    neighbor.makeOpen()

        draw()

        if current != startNode:
            current.makeClosed()

    return False


# Function for calculating the heuristic value
def heuristicFunction(d1, d2):
    x1, y1 = d1
    x2, y2 = d2

    # Euclidean Distance
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Diagonal Distance
    # return max(abs(x1 - x2), abs(y1 - y2))

    # Manhattan Distance
    # return abs(x1 - x2) + abs(y1 - y2)


# Function which makes a grid of nodes
def makeGrid(rows, width):
    grid = []
    gap = width // rows

    for i in range(rows):
        grid.append([])

        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid


# Function for drawing the border of the nodes in the grid
def drawGrid(win, rows, width):
    gap = width // rows

    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))

        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


# Function which draws the nodes from the grid on the pygame window
def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(win)

    drawGrid(win, rows, width)
    pygame.display.update()


# Main Function
def main(win, width):
    # Number of rows
    ROWS = 80
    grid = makeGrid(ROWS, width)
    barriers = []

    # Initializing the starting and the ending node
    row, col = (9, 3)
    node = grid[row][col]
    startNode = node
    startNode.makeStart()

    row, col = (50, 50)
    node2 = grid[row][col]
    endNode = node2
    endNode.makeEnd()

    # Define the maze as a two-dimensional list
    maze = [
        ['x',  0,  0,  0,  0,  0,  0,  0, 0,  0 , 0,  0,  0,  0,  0,  0,  0,  0,  0, 0],
        [0, 'x','x', 0, 'x', 0,  0,  0, 0,  0, 'x','x', 0, 'x', 0,  0,  0,  0,  0, 0],
        [0,  0, 'x','x','x','x','x','x',0,  0,  0, 'x','x','x','x','x','x' ,0,  0, 0],
        [0,  0, 'x','x', 0,  0, 'x', 0, 0,  0,  0,  0, 'x','x', 0,  0, 'x', 0,  0, 0],
        [0,  0, 'x', 0, 'x', 0, 'x', 0, 0,  0,  0,  0, 'x', 0, 'x', 0, 'x', 0,  0, 0],
        [0,  0, 'x', 0, 'x', 0,  0,  0, 0,  0,  0,  0, 'x', 0, 'x', 0,  0,  0,  0, 0],
        [0, 'x','x', 0, 'x', 0, 'x','x','x',0,  0, 'x','x', 0, 'x', 0, 'x','x','x',0],
        [0,  0,  0,  0, 'x', 0,  0,  0,  0, 0,  0,  0,  0,  0, 'x', 0,  0,  0,  0, 0],
        [0,  0, 'x','x','x','x','x', 0,  0, 0,  0,  0, 'x','x','x','x','x', 0,  0, 0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,'x', 0,  0,  0,  0,  0,  0,  0,  0,  0,'x'],
        [0,  0,  0,  0,  0,  0,  0,  0,  0, 0  ,0, 'x','x', 0, 'x', 0,  0,  0,  0, 0],
        [0, 'x','x', 0, 'x', 0,  0,  0, 0,  0,  0, 'x','x', 0, 'x', 0,  0,  0,  0, 0],
        [0,  0, 'x','x','x','x','x','x',0,  0,  0,  0, 'x','x','x','x','x','x', 0, 0],
        [0,  0, 'x','x', 0,  0, 'x', 0, 0,  0,  0,  0, 'x','x', 0,  0, 'x', 0, 0,  0],
        [0,  0, 'x', 0, 'x', 0, 'x', 0, 0,  0,  0,  0, 'x', 0, 'x', 0, 'x', 0, 0,  0],
        [0,  0, 'x', 0, 'x', 0,  0,  0, 0,  0,  0,  0, 'x', 0, 'x', 0,  0,  0, 0,  0],
        [0, 'x','x', 0, 'x', 0, 'x','x','x',0,  0, 'x','x', 0, 'x', 0, 'x','x','x',0],
        [0,  0,  0,  0, 'x', 0,  0,  0,  0, 0,  0,  0,  0,  0, 'x', 0,  0,  0,  0, 0],
        [0,  0, 'x','x','x','x','x', 0,  0, 0,  0,  0, 'x','x','x','x','x', 0,  0, 0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,'x', 0,  0,  0,  0,  0,  0,  0,  0,  0,'x'],
    ]
    # Loop over the maze list and find the walls
    obstacles = [(j, i) for i in range(20) for j in range(20) if maze[i][j] == 'x']
    for obs in obstacles:
        row = obs[0]*3
        col = obs[1]*3
        for i in range(-1,2):
            for j in range(-1,2):
                node = grid[row+i][col+j]
                node.makeBarrier()
                barriers.append(node)

    # random.seed(12)

    # # Generate random positions for each object
    # for _ in range(num_objects):
    #     row = random.randint(0, 80 - 1)
    #     col = random.randint(0, 80 - 1)
    #     node = grid[row][col]
    #     node.makeBarrier()
    #     barriers.append(node)

    vor = Voronoi([barriers[i].getPosition() for i in range(80)])
    run = True
    while run:
        # Drawing the grid onto the pygame window
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and startNode and endNode:
                    for row in grid:
                        for node in row:
                            node.updateNeighbors(grid)
                    first = time.time()
                    AStarAlgorithm(
                        lambda: draw(win, grid, ROWS, width),
                        grid,
                        startNode,
                        endNode,
                        barriers,
                        vor)
                    second = time.time()
                    print("computational time",second-first)

    pygame.quit()


# Main Loop
if __name__ == "__main__":
    main(WIN, WIDTH)