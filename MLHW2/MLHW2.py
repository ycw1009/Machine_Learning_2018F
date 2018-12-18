# # MLHW2
# ## 1. K-Means



get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd




from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import itertools




class Kmeans(object):
    def __init__(self,cluster_num,data):
        self.centers = np.zeros((cluster_num,data.shape[1]), dtype=float)
        self.data = data
        self.cluster_num = cluster_num
        self.data_num = data.shape[0]
        self.feature_num = data.shape[1]
        self.clusters = np.zeros(data.shape[0],dtype=int)
        self.distances = np.zeros((data.shape[0],cluster_num), dtype=float)
        self.is_complete = False
        
    # Initial center by randomly selecting data    
    def initialize_centers(self):
        self.centers = np.array([self.data[i] 
                                 for i in random.sample(range(0, self.data_num), 
                                                        self.cluster_num)])
    
    # Assignment data to the nearest center
    def assignment_step(self): 
        # Calcaulate distance between two data
        def dis(a,b): 
            return np.sqrt(np.sum((a-b)**2))
        
        for index_datum, datum in enumerate(self.data):# For each datum
            for index_center, center in enumerate(self.centers): # calculate distance to every center
                self.distances[index_datum][index_center] = dis(datum,center)
            self.clusters[index_datum] = np.argmin(self.distances[index_datum])# choose the nearest one
            
            # check the clustering situation
            numbers = np.bincount(self.clusters) # bincount the clustering situation
            # Bad center choose, there is at least one cluster has one member
            if np.count_nonzero(numbers) < self.cluster_num : 
                self.initialize_centers()

    # Update new center
    def update_step(self):
        centers_old = deepcopy(self.centers)
        self.centers = np.zeros((self.cluster_num,self.feature_num), 
                                dtype=float)
        for index_datum, datum in enumerate(self.data):
            self.centers[self.clusters[index_datum]]+= datum
        numbers = np.bincount(self.clusters)
        numbers.resize(self.cluster_num,1)
        self.centers = self.centers / numbers
        # if this time the center doesn't change, end the clustering.
        error = np.linalg.norm(self.centers - centers_old)
        if error == 0:
            self.is_complete = True

    # Calculate the whole clustering distance
    def distance_to_centroid(self):
        sum = 0
        for index_datum, datum in enumerate(self.data):
            sum += self.distances[index_datum][self.clusters[index_datum]]
        return sum/(self.cluster_num*self.data_num)
    
    # Do clustering
    def clustering(self):
        while self.is_complete is False:
            self.assignment_step()
            self.update_step()




def decide_clustering_num(data):
    threshold_slope = 0.5
    cluster_num = None
    distance = np.zeros(9,dtype=float)
    x = [i for i in range(1,10)]
    #print(x)
    
    for i in range(1,10):
        print(f"Calculating {i} clustering distance...")
        kmeans = Kmeans(i,data)
        kmeans.clustering()
        distance[i-1] = kmeans.distance_to_centroid()
    for i in range(1,9):
        print(abs(distance[i]-distance[i-1]),end=' ')
        if cluster_num is None and abs(distance[i]-distance[i-1]) < abs(threshold_slope):
            cluster_num = i
    
    plt.plot(x,distance,'-o')
    plt.ylabel('distance')
    plt.xlabel('k')
    plt.show()
    print(f"""We choose {cluster_num} as our clustering number,
because its absolute slope is smaller than {threshold_slope}
""")
    return cluster_num





def calculate_accuracy(model,truth):
    truth = [''.join(item) for item in truth] # np.array to list
    permutations = list(itertools.permutations([str(i) for i in range(kmeans.cluster_num)]))
    kmeans_result = np.array([str(i) for i in model])
    result_label = []
    accuracy = 0
    for i in range(len(permutations)):# count all possible permetations
        sum = 0
        # change the content of the string to be coherent
        pitch_type = [item.replace("FF",permutations[i][0]) for item in truth]
        pitch_type = [item.replace("CH",permutations[i][1]) for item in pitch_type]
        pitch_type = [item.replace("CU",permutations[i][2]) for item in pitch_type]
        # compare the list
        for i, j in zip(pitch_type,kmeans_result):
            if i == j:
                sum += 1
        #the best is the result we want
        if sum /len(kmeans_result) > accuracy:
            accuracy = sum /len(kmeans_result)
    print("%.2f%%" % (accuracy*100))




df = pd.read_csv("./datasets/data_noah.csv")
data = df[['x','y']].values
pitch_type = df[['pitch_type']].values.tolist()
# Number of clusters

cluster_num = decide_clustering_num(data)

kmeans = Kmeans(cluster_num,data)
kmeans.clustering()

print("K-Means clustering Accuracy: ",end = " ")
labels = calculate_accuracy(kmeans.clusters,pitch_type)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(data[:, 0], data[:, 1], c= kmeans.clusters,marker='.') #C是第三維度 已顏色做維度
for center in kmeans.centers:
    plt.plot([center[0]],[center[1]],marker='*',color="red")
plt.show()




# Add two attribute 'vx0','speed' to do clustering
print("Add two attribute 'vx0','speed' to do clustering, Accuracy =", end= " ")
data_2 = df[['x','y','vx0','speed']].values

kmeans_2 = Kmeans(cluster_num,data_2)
kmeans_2.clustering()
calculate_accuracy(kmeans_2.clusters,pitch_type)


# ## 2. KD-Tree



import pprint


def build_kd_tree(points, depth = 0 ):
    n = len(points)
    if n <= 0 :
        return None
    axis = (depth + SPILTTING_PLANE) % 2
    sorted_points = sorted(points, key=lambda point:point[axis]) 
    return {
        'left': build_kd_tree(sorted_points[:n//2],depth+1),
        'root': sorted_points[n//2],
        'right': build_kd_tree(sorted_points[n//2+1:],depth+1),
        'axis': axis
    }

def draw_kd_tree( node, x_min, x_max, y_min, y_max):
    if node is not None:
        plt.plot(node['root'][0],node['root'][1],'k.')
        if node['axis']: # X axis
            plt.plot([x_min,x_max], [node['root'][1],node['root'][1]],color='r',linewidth=0.5)
            draw_kd_tree(node['left'], x_min, x_max,y_min, node['root'][1])
            draw_kd_tree(node['right'], x_min, x_max, node['root'][1], y_max)
        else: # Y axis
            plt.plot([node['root'][0],node['root'][0]],[y_min,y_max],color='b',linewidth=0.5)
            draw_kd_tree(node['left'], x_min,node['root'][0],y_min, y_max)
            draw_kd_tree(node['right'], node['root'][0], x_max, y_min, y_max)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("KD-Tree")
   




data = np.empty(shape=[0, 2])

with open("./datasets/points.txt") as file:
    for line in file:
        data = np.append(data, [[int(num) for num in line.split(" ")]],axis=0)

coordinate = ['X','Y']
print(f"std of {coordinate[0]}: {np.std(data[:,0])}, std of {coordinate[1]}: {np.std(data[:,1])}")

# Choose a axis with bigger std
SPILTTING_PLANE = 0 if np.std(data[:,0]) > np.std(data[:,1]) else 1

print(f"Choose {coordinate[SPILTTING_PLANE]} as axis-aligned splitting plane")
# Set range of coordinate
x_min = min(data[:,0])-1
y_min = min(data[:,1])-1
x_max = max(data[:,0])+1
y_max = max(data[:,1])+1




kdtree = build_kd_tree(data)
print("Show KD-Tree Structure\n")
pprint.pprint(kdtree)
print("\nShow KD-Tree Picture\n")
draw_kd_tree(kdtree, x_min, x_max, y_min, y_max)

