import heapq

a = []
heapq.heappush(a,(10,(3,2)))
# print(a)
heapq.heappush(a,(3,(10,10)))
# print(a)
heapq.heappush(a,(2,(2,10)))
# print(a)
heapq.heappush(a,(5,(3,10)))
# print(a)
heapq.heappop(a)
print(a)

def contains_cell(lst,neighbor):
    
    for _, cell_value in lst:
        if cell_value == neighbor:
            return (_,cell_value), True
    return None, False

_, in_open_set = contains_cell(a,(3,10))
print(_)
heapq.heapreplace(a,)