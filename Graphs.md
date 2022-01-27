# Graph Problems using stack,queues and recursion (Python)
### Graph Traversal using DFS & BFS


```python
graph={
    'a':['b','c'],
    'b':['d'],
    'c':['e'],
    'd':['f'],
    'e':[],
    'f':[]
}
```

- DFS Traversal with Using stack


```python
def graph_traversal(graph):
    stack=[]
    result=[]
    ## append to stack
    stack.append('a')
    while len(stack):
        curr_node=stack.pop()
        result.append(curr_node)
        for node in graph[curr_node]:
            stack.append(node)
    return result
```


```python
graph_traversal(graph)
```




    ['a', 'c', 'e', 'b', 'd', 'f']



- BFS Traversal using queue 

```python
from queue import deque
def bfs_traversal(graph):
    track_que=deque()
    track_que.append('a')
    result=[]
    while track_que:
        curr_node=track_que.popleft()
        result.append(curr_node)
        for node in graph[curr_node]:
            track_que.append(node)
    return result
```


```python
bfs_traversal(graph)
```




    ['a', 'b', 'c', 'd', 'e', 'f']



- Working with cyclic graph


```python
def is_cyclic_graph(graph):
    stack=[]
    result={}
    stack.append('a')
    while stack:
        curr_node=stack.pop()
        if curr_node in result:
            return True
        result[curr_node]=True
        for node in graph[curr_node]:
            stack.append(node)
    return False
```


```python
graph1={
    'a':['b','c'],
    'b':['d'],
    'c':['e','b'],
    'd':['f'],
    'e':[],
    'f':[]
}
```


```python
is_cyclic_graph(graph1)
```




    True



```python
graph={
    'a':['b','c'],
    'b':['d'],
    'c':['e'],
    'd':['f'],
    'e':[],
    'f':[]
}
```

```python
is_cyclic_graph(graph)
```




    False


- traversal cyclic graph

```python
def traversal_cyclic_graph(graph):
    stack,result=[],{}
    stack.append('a')
    while stack:
        curr_node=stack.pop()
        if curr_node in result:
            continue
        result[curr_node]=True
        for node in graph[curr_node]:
            stack.append(node)
    return list(result.keys())
```


```python
traversal_cyclic_graph(graph1)
```




    ['a', 'c', 'b', 'd', 'f', 'e']



- HashPath from one node to another node
    - using stacks


```python
graph = {
  'f': ['g', 'i'],
  'g': ['h'],
  'h': [],
  'i': ['g', 'k'],
  'j': ['i'],
  'k': []
}

```


```python
def is_travel_possible(graph,source,destination):
    stack=[]
    stack.append(source)
    while stack:
        curr_node=stack.pop()
        if curr_node==destination:
            return True
        for node in graph[curr_node]:
            stack.append(node)
    return False
```


```python
is_travel_possible(graph,'f','k')
```




    True




```python
is_travel_possible(graph,'i','f')
```




    False


- HashPath from one node to another node
    - Using recursion   


```python
def is_travel_possible(graph,source,dest):
    if source==dest:
        return True
    for node in graph[dest]:
        return is_travel_possible(graph,source,node)
    return False
```


```python
graph = {
  'f': ['g', 'i'],
  'g': ['h'],
  'h': [],
  'i': ['g', 'k'],
  'j': ['i'],
  'k': []
}

is_travel_possible(graph,'i','f')
```




    False




```python
is_travel_possible(graph,'f','k')
```




    False


- build adjacent list when given edges as input for graph problem 

```python
edges = [
  ['i', 'j'],
  ['k', 'i'],
  ['m', 'k'],
  ['k', 'l'],
  ['o', 'n']
];

```


```python
def buildAdjcentList(edges):
    graph={}
    for edge in edges:
        node1,node2=edge
        #print(node1,node2)
        if node1 not in graph:
            graph[node1]=[]
        if node2 not in graph:
            graph[node2]=[]
            
        graph[node1].append(node2)
        graph[node2].append(node1)
    return graph

```


```python
buildAdjcentList(edges)
```




    {'i': ['j', 'k'],
     'j': ['i'],
     'k': ['i', 'm', 'l'],
     'm': ['k'],
     'l': ['k'],
     'o': ['n'],
     'n': ['o']}


- undirected graph traversal using edges of pairs

```python
def undir_graph_traversal(graph,source,target):
    stack=[]
    stack.append(source)
    visted_nodes={}
    while stack:
        curr_node=stack.pop()
        if curr_node==target:
            return True
        if curr_node in visted_nodes:
            continue
        visted_nodes[curr_node]=True
        for node in graph[curr_node]:
            stack.append(node)
    return False
    
```


```python
adj_graph=buildAdjcentList(edges)
undir_graph_traversal(adj_graph,'l', 'j')
```




    True




```python
undir_graph_traversal(adj_graph,'k', 'o')
```




    False



#### connected components count
    Write a function, connectedComponentsCount, that takes in the adjacency list of an undirected graph. The function should return the number of connected components within the graph.
    
         connectedComponentsCount({
          0: [8, 1, 5],
          1: [0],
          5: [0, 8],
          8: [0, 5],
          2: [3, 4],
          3: [2, 4],
          4: [3, 2]
        }); // -> 2

- using stack
```python
def connected_components(graph):
    def graph_traversal(node):
        stack=[]
        stack.append(node)
        while stack:
            curr_node=stack.pop()
            if curr_node in visted_nodes:
                continue
            visted_nodes[curr_node]=True
            for node in graph[curr_node]:
                stack.append(node)
    count=0
    visted_nodes={}
    for node in graph.keys():
        if node not in visted_nodes:
            count+=1
            graph_traversal(node)
    return count
```


```python
graph={
      0: [8, 1, 5],
      1: [0],
      5: [0, 8],
      8: [0, 5],
      2: [3, 4],
      3: [2, 4],
      4: [3, 2]
    }
connected_components(graph)
```




    2




```python
graph={
  3: [],
  4: [6],
  6: [4, 5, 7, 8],
  8: [6],
  7: [6],
  5: [6],
  1: [2],
  2: [1]
}
connected_components(graph)
```




    3



- using recurssion 
```python
def connected_components(graph):
    def graph_traversal(node):
        if node in visted_nodes:
            return 
        visted_nodes[node]=True
        for node in graph[node]:
            graph_traversal(node)
    count=0
    visted_nodes={}
    for node in graph.keys():
        if node not in visted_nodes:
            count+=1
            graph_traversal(node)
    return count
```


```python
graph={
  3: [],
  4: [6],
  6: [4, 5, 7, 8],
  8: [6],
  7: [6],
  5: [6],
  1: [2],
  2: [1]
}
connected_components(graph)
```




    3



#### find largest connected components count
- using stack
```python
def largest_component1(graph):
    def graph_traversal(node):
        if node in visted_nodes:
            return 0
        visted_nodes[node]=True
        counter=1
        for edge_node in graph[node]:
            counter+=graph_traversal(edge_node)
        return counter
    visted_nodes={}
    large_component=float('-inf')
    for node in graph.keys():
        if node not in visted_nodes:
            component_len=graph_traversal(node)
            print(node,component_len)
            large_component=max(large_component,component_len)
    return large_component     
```


```python
graph={
  3: [],
  4: [6],
  6: [4, 5, 7, 8],
  8: [6],
  7: [6],
  5: [6],
  1: [2],
  2: [1]
}
largest_component1(graph)
```

    3 1
    4 5
    1 2





    5


- using recursion 

```python
def largest_component(graph):
    def graph_traversal(node):
        stack=[]
        stack.append(node)
        visting_len=0
        while stack:
            curr_node=stack.pop()
            if curr_node in visted_nodes:
                continue
            visting_len+=1
            visted_nodes[curr_node]=True
            for edge_node in graph[curr_node]:
                stack.append(edge_node)
        return visting_len
    
    visted_nodes={}
    large_component=float('-inf')
    for node in graph.keys():
        if node not in visted_nodes:
            prev_len=len(visted_nodes)
            len1=graph_traversal(node)
            component_len=len(visted_nodes)-prev_len
            print(component_len,node,component_len)
            large_component=max(large_component,component_len)
    return large_component   
```


```python
graph={
  3: [],
  4: [6],
  6: [4, 5, 7, 8],
  8: [6],
  7: [6],
  5: [6],
  1: [2],
  2: [1]
}
largest_component(graph)
```

    1 3 1
    5 4 5
    2 1 2





    5



### shortest path
    Write a function, shortest_path, that takes in a list of edges for an undirected graph and two nodes (node_A, node_B). The function should return the length of the shortest path between A and B. Consider the length as the number of edges in the path, not the number of nodes. If there is no path between A and B, then return -1.
    
        edges = [
      ['w', 'x'],
      ['x', 'y'],
      ['z', 'y'],
      ['z', 'v'],
      ['w', 'v']
    ]

    shortest_path(edges, 'w', 'z') # -> 2

- using queue
- 
```python
from queue import deque
def buil_adjcent_list(edges):
    adj_list={}
    for node1,node2 in edges:
        if node1 not in adj_list:
            adj_list[node1]=[]
        if node2 not in adj_list:
            adj_list[node2]=[]
        adj_list[node2].append(node1)
        adj_list[node1].append(node2)
    return adj_list


def shortest_path(graph,source,target):
    que=deque()
    que.append((source,0))
    visted_nodes={}
    short_path=float('inf')
    while que:
        curr_node,no_of_edges=que.popleft()
        if curr_node in visted_nodes:
            continue
        if curr_node==target:
            short_path=min(short_path,no_of_edges)
        visted_nodes[curr_node]=True
        for node in graph[curr_node]:
            que.append((node,no_of_edges+1))
    return -1 if short_path==float('inf') else short_path

```


```python
    edges = [
  ['w', 'x'],
  ['x', 'y'],
  ['z', 'y'],
  ['z', 'v'],
  ['w', 'v']
]
adj_list=buil_adjcent_list(edges)
graph_traversal(adj_list,'w','z')
```




    2




```python
adj_list
```




    {'w': ['x', 'v'],
     'x': ['w', 'y'],
     'y': ['x', 'z'],
     'z': ['y', 'v'],
     'v': ['z', 'w']}




```python
edges = [
  ['a', 'c'],
  ['a', 'b'],
  ['c', 'b'],
  ['c', 'd'],
  ['b', 'd'],
  ['e', 'd'],
  ['g', 'f']
]

adj_list=buil_adjcent_list(edges)
#graph_traversal(adj_list,'w','z')
shortest_path(adj_list, 'a', 'e') # -> 3
```




    3



### island count
    Write a function, island_count, that takes in a grid containing Ws and Ls. W represents water and L represents land. The function should return the number of islands on the grid. An island is a vertically or horizontally connected region of land.


```python
grid = [
  ['W', 'L', 'W', 'W', 'W'],
  ['W', 'L', 'W', 'W', 'W'],
  ['W', 'W', 'W', 'L', 'W'],
  ['W', 'W', 'L', 'L', 'W'],
  ['L', 'W', 'W', 'L', 'L'],
  ['L', 'L', 'W', 'W', 'W'],
]
```

- using recursion 

```python
def find_number_of_islands(grid):
    def island_traversal(row,col):
        if row<0 or row>=rows or col<0 or col>=columns or (row,col) in visted_nodes or grid[row][col]=='W':
            return 
        visted_nodes[(row,col)]=True
        island_traversal(row+1,col)
        island_traversal(row-1,col)
        island_traversal(row,col-1)
        island_traversal(row,col+1)
   
    rows,columns=len(grid),len(grid[0])
    visted_nodes={}
    no_of_islands=0
    for row in range(rows):
        for col in range(columns):
            if grid[row][col]=='W' or (row,col) in visted_nodes:
                continue
            island_traversal(row,col)
            no_of_islands+=1
    return no_of_islands
```


```python
find_number_of_islands(grid)
```




    3



- using stack


```python
def find_number_of_islands(grid):
    def island_traversal(row,col):
        stack=[]
        stack.append((row,col))
        while stack:
            curr_row,curr_col=stack.pop()
            row_out_bounds=(curr_row<0 or curr_row>=rows)
            col_out_bounds=(curr_col<0 or curr_col>=columns)
            is_visted=(curr_row,curr_col) in visted_nodes
            if row_out_bounds or col_out_bounds or is_visted or grid[curr_row][curr_col]=='W':
                continue
            visted_nodes[(curr_row,curr_col)]=True
            stack.append((curr_row+1,curr_col))
            stack.append((curr_row-1,curr_col))
            stack.append((curr_row,curr_col+1))
            stack.append((curr_row,curr_col-1))
    
    rows,columns=len(grid),len(grid[0])
    visted_nodes={}
    no_of_islands=0
    for row in range(rows):
        for col in range(columns):
            if grid[row][col]=='W' or (row,col) in visted_nodes:
                continue
            island_traversal(row,col)
            no_of_islands+=1
    return no_of_islands
            
```


```python
find_number_of_islands(grid)
```




    3



### max island len 
    Write a function, island_count, that takes in a grid containing Ws and Ls. W represents water and L represents land. The function should return the number of islands on the grid. An island is a vertically or horizontally connected region of land.


```python
def max_of_island(grid):
    def island_traversal(row,col):
        if row<0 or row>=rows or col<0 or col>=columns or (row,col) in visted_nodes or grid[row][col]=='W':
            return 0
        visted_nodes[(row,col)]=True
        return (1+island_traversal(row+1,col)+
        island_traversal(row-1,col)+
        island_traversal(row,col-1)+
        island_traversal(row,col+1))
   
    rows,columns=len(grid),len(grid[0])
    visted_nodes={}
    no_of_islands=0
    max_of_island=float('-inf')
    for row in range(rows):
        for col in range(columns):
            if grid[row][col]=='W' or (row,col) in visted_nodes:
                continue
            len_of_island=island_traversal(row,col)
            max_of_island=max(max_of_island,len_of_island)
            no_of_islands+=1
    return max_of_island
```


```python
grid = [
  ['W', 'L', 'W', 'W', 'W'],
  ['W', 'L', 'W', 'W', 'W'],
  ['W', 'W', 'W', 'L', 'W'],
  ['W', 'W', 'L', 'L', 'W'],
  ['L', 'W', 'W', 'L', 'L'],
  ['L', 'L', 'W', 'W', 'W'],
]
max_of_island(grid)
```




    5




### minimum island
    Write a function, minimum_island, that takes in a grid containing Ws and Ls. W represents water and L represents land. The function should return the size of the smallest island. An island is a vertically or horizontally connected region of land.

    You may assume that the grid contains at least one island.


```python
def minimum_islands(grid):
    def island_traversal(row,col):
        row_out_bounds=(row<0 or row>=rows)
        col_out_bounds=(col<0 or col>=columns)
        if row_out_bounds or col_out_bounds or (row,col) in visted_nodes or grid[row][col]=='W':
            return 0
        visted_nodes[(row,col)]=True
        return (1+
                island_traversal(row-1,col)+
                island_traversal(row+1,col)+
                island_traversal(row,col-1)+
                island_traversal(row,col+1)
               )
    rows,columns=len(grid),len(grid[0])
    visted_nodes={}
    min_island_len=float('inf')
    for row in range(rows):
        for col in range(columns):
            if grid[row][col]=='W' or (row,col) in visted_nodes:
                continue
            island_len=island_traversal(row,col)
            min_island_len=min(min_island_len,island_len)
    return min_island_len
```


```python
grid = [
  ['W', 'L', 'W', 'W', 'W'],
  ['W', 'W', 'W', 'W', 'W'],
  ['W', 'W', 'W', 'L', 'W'],
  ['W', 'W', 'L', 'L', 'W'],
  ['L', 'W', 'W', 'L', 'L'],
  ['L', 'L', 'W', 'W', 'W'],
]
minimum_islands(grid)
```




    1



### 463. Island Perimeter

    Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
    Output: 16
    Explanation: The perimeter is the 16 yellow stripes in the image above.


```python
def island_perimeter(grid):
    rows,columns=len(grid),len(grid[0])
    perimeter=0
    for row in range(rows):
        for col in range(columns):
            if grid[row][col]==0:
                continue
            per_val=sum([grid[row+1][col] if (row+1)<rows else 0,
                        grid[row-1][col] if (row-1)>=0 else 0 ,
                        grid[row][col-1] if (col-1)>=0 else 0 ,
                        grid[row][col+1] if (col+1)<columns else 0 ])
            perimeter+=(4-per_val)
            print(4-per_val,per_val,(row,col))
    return perimeter
            
```


```python
island_perimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]])
```

    16




```python

```

#### 694. Number of Distinct Islands

    You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
    An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other.
    Return the number of distinct islands.
 ```python
 def find_unique_islands(grid):
    def island_traversal(row,col):
        is_row_outbound=(row<0 or row>=rows)
        is_col_outbound=(col<0 or col>=columns)
        is_visted=(row,col) in visted_nodes
        if is_row_outbound or is_col_outbound or is_visted or grid[row][col]==0:
            return
        visted_nodes[(row,col)]=True
        local_islands.append((row-row_origon,col-col_origon))
        island_traversal(row-1,col)
        island_traversal(row+1,col)
        island_traversal(row,col-1)
        island_traversal(row,col+1)
    
    rows,columns=len(grid),len(grid[0])
    visted_nodes={}
    hash_ilands={}
    for row_origon in range(rows):
        for col_origon in range(columns):
            if (row_origon,col_origon) in visted_nodes or grid[row_origon][col_origon]==0:
                continue
            local_islands=[]
            island_traversal(row_origon,col_origon)
            hash_ilands[hash(str(local_islands))]=True
    return len(hash_ilands)
    
```

```python
    grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
    find_unique_islands(grid)
```    
    
    1
    
    
