import numpy as np
import statistics as ss
import pandas as ps
import csv
import matplotlib.pyplot as plt
import random as rdm
import copy
import string
from scipy.spatial import distance as dst

class trajectory:
    
    
    def __init__(self):
        self.xs = np.array([], (float))
        self.ys = np.array([], (float))
        self.timestamp = np.array([], (float))
        
    def add_point(self,time,x,y):
        self.xs = np.append(self.xs,x)
        self.ys = np.append(self.ys,y)
        self.timestamp = np.append(self.timestamp,time)
        
    def get_trajectory():     
        print("not implemented")
        
class trajectories:
     
    def __init__(self):
        self.pathdict = {}
        
    def add_trajectory(self,id, trajectory):
        self.pathdict[id] = trajectory
        
    def kmeansclustering(self, k):
        #Generate k centroids
        rdmkeys = []
        uniquekey = False
        for i in range(0, k):
            while not uniquekey:
                newkey = rdm.choice(list(self.pathdict.keys()))
                uniquekey = True
                for key in rdmkeys:
                    if newkey is key:
                        uniquekey = False
            if uniquekey:
                rdmkeys.append(newkey)
                
        centroids = {}
        for key in rdmkeys:
            centroids[''.join(rdm.choices(string.ascii_uppercase + string.digits, k=5))] = copy.deepcopy(self.pathdict[key])
        #self.plotwx(centroids)
        
        clusters = {}
        for key in centroids:
            clusters[key] = []
            
        for pathkey in self.pathdict:
            mindistance = self.calc_distance(centroids[next(iter(centroids))], self.pathdict[pathkey])
            minkey = next(iter(centroids.items()))[0]
            for key in centroids:
                newdistance = self.calc_distance(centroids[key], self.pathdict[pathkey])
                if  newdistance < mindistance:
                    minkey = key
                    mindistance = newdistance
            clusters[minkey].append(pathkey)
        
        numofpoints = len(self.pathdict[next(iter(self.pathdict))].xs)
        for key in centroids:
            xsum = 0
            ysum = 0
            for clusterkey in clusters:
                self.pathdict[clusters[clusterkey]].xs
         #centroids[key]
            
            
        
        
    def calc_distance(self,traj1 , traj2):
        sum = 0.0
        numofpoints = len(traj1.xs)
        for i in range(0, numofpoints): #assumes traj1 and traj2 has equal data points
            a = np.matrix((traj1.xs[i], traj1.ys[i]))
            b = np.matrix((traj2.xs[i], traj2.ys[i]))
            sum += np.linalg.norm(a-b)
        return sum    
        
            

    def plotwx(self, x):
        
        for id in x:
            plt.plot(x[id].xs, x[id].ys, "*")
        
        self.plot()
        
    def plot(self):
        
        plt.axis([-50000,50000.0,-50000.0,50000.0]) # xmin, xmax, ymin, ymax
        #plt.gca().set_autoscale_on(False)
        
        for id in self.pathdict:
            plt.plot(self.pathdict[id].xs, self.pathdict[id].ys)
            
        plt.show();

trajs = trajectories()

def check_if_valid_trajectory(traj, minimumtraveldistance = 1):
    sum = 0
    length = len(traj.xs)
    
    for i in range(length):
        for j in range(i+1, length):
            sum += abs(traj.xs[j]) - abs(traj.xs[i])
            sum += abs(traj.ys[j]) - abs(traj.ys[i])
    if(sum < minimumtraveldistance):
        return False;
    return True;
    
        
def readcsvfile(numoftrajstoread=0):
    global trajs;
    
    with open('testfile.csv') as data:
        data = csv.reader(data, delimiter=',')
        trajnr = 0
        
        isnewtrajectory = True
        id = 0
        for row in data:
            if(row[0] == '###'):
                if check_if_valid_trajectory(newtrajectory, 1000) and len(newtrajectory.timestamp) == 33:
                    trajs.add_trajectory(id,newtrajectory)
                    trajnr = trajnr + 1;
                if numoftrajstoread is not 0 and trajnr >= numoftrajstoread:
                    break;
                newtrajectory = trajectory()
                isnewtrajectory = True
                
            else:
                if not isnewtrajectory:
                    newtrajectory.add_point(row[0], int(row[2]),int(row[3])) 
                    
                else:
                    id = row[1]
                    newtrajectory = trajectory()
                    isnewtrajectory = False                    
                    
                
            
                

#readcsvfile(50)

readcsvfile(5)

#keys = []
#for key in trajs.pathdict:
#    keys.append(key)    
#trajs.calc_distance(trajs.pathdict[keys[0]], trajs.pathdict[keys[1]])

#trajs.plot()
trajs.kmeansclustering(3)