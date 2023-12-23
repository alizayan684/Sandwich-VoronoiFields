import heapq
# Various colors used in the visualization
RED = (255, 255, 255) # For the start and end markers
GREEN = (255, 255, 255) # For the start and end markers
BLUE = (255, 255, 255) # For the start and end markers
YELLOW = (128, 0, 200) # For the start and end markers
WHITE = (255, 255, 255) # For the grid lines
BLACK = (0, 0, 0) # For the background
PURPLE = (128, 0, 128) # For the path
ORANGE = (128, 128, 128) # For the visited cells
GREY = (128, 128, 128) # For the grid lines
TURQUOISE = (64, 224, 208) # For the start and end markers

heap = []
heapq.heapify(heap)
heapq.heappush(heap, 5)
heapq.heappush(heap, 6)

h = heapq.heappop(heap)
print(h)
dic = {"d": 5}
dic.values()