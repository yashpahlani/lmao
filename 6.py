import matplotlib.pyplot as plt

import networkx as nx
import pandas as pd
import random

# Generate a random undirected graph with 50 nodes
graph = nx.Graph()
num_nodes = 50
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if random.random() < 0.2:  # Adjust the probability to control sparsity
            weight = random.randint(1, 10)  # Random weight between 1 and 10
            graph.add_edge(i, j, weight=weight)

# Convert the graph to a DataFrame
data = []
for edge in graph.edges(data=True):
    data.append((edge[0], edge[1], edge[2]['weight']))
df = pd.DataFrame(data, columns=['From', 'To', 'Weight'])

# Write DataFrame to CSV
df.to_csv('undirected_graph.csv', index=False)
# load Edgelist
graph = nx.from_pandas_edgelist(df, source='From', target='To')
graph

len(graph.nodes())

len(graph.edges)

plt.figure(figsize=(5, 4))

node_size = 100
label_font_size = 5

# Draw the graph with customized node and label sizes
nx.draw(graph, with_labels=True, node_size=node_size, font_size=label_font_size)

# Show the plot
plt.show()

### Degree Centrality

nx.degree_centrality(graph)

sorted(nx.degree_centrality(graph).values())
m_influencial = nx.degree_centrality(graph)
for w in sorted(m_influencial,key= m_influencial.get,reverse = True):
    print(w,m_influencial[w])

### Betweeness_Centrality

pos = nx.spring_layout(graph)
betCent = nx.betweenness_centrality(graph,normalized=True,endpoints=True)
node_color = [200000.0*graph.degree(v) for v in graph]
node_size = [v*10000 for v in betCent.values()]
# plt.figure(figsize= (20,20))
plt.figure(figsize = (12,12))
nx.draw_networkx(graph,pos=pos,with_labels= False, node_color = node_color, node_size= node_size)
sorted(betCent,key= betCent.get, reverse= True)[:5]

### Closeness Centrality

closeness_centrality = nx.centrality.closeness_centrality(graph)
sorted(closeness_centrality.items(),key = lambda item: item[1],reverse = True)[:8]

node_size = [v*500 for v in closeness_centrality.values()]
plt.figure(figsize= (15,8))
nx.draw_networkx(graph, pos = pos, node_size =node_size, with_labels = False,width= 0.15)
plt.axis('off')

nx.average_clustering(graph)

triangle_per_node = list(nx.triangles(graph).values())
sum(triangle_per_node)/3

nx.has_bridges(graph)
