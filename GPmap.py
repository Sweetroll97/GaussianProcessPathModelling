import numpy as np
import statistics as ss
import pandas as ps
import csv
import matplotlib.pyplot as plt

class trajectory:
    xs = np.array([], (float))
    ys = np.array([], (float))
    timestamp = np.array([], (float))
    
    #def __init__(self, id):
    #    self.id = id;
        
    def add_point(self,time,x,y):
        self.xs.append(x)
        self.ys.append(y)
        self.timestamp.append(time)
        
    def get_trajectory():     
        print("not implemented")
        
class trajectories:
     
    def __init__(self):
        self.pathdict = {}
        
    def add_trajectory(self,id, trajectory):
        self.pathdict[id] = trajectory
    def plot():
        for id in self.pathdict:
            plt.scatter(self.pathdict[id].x, self.pathdict[id].y)
        plt.plot();
        print("hej");

trajs = trajectories()
 
        
def readcsvfile():
    global trajs;
    
    with open('testfile.csv') as data:
        data = csv.reader(data, delimiter=',')
        linenr = 0
        
        isnewtrajectory = true
        id
        #newtrajectory = trajectory();
        for row in data:
            if(row[0] is "###"):
                trajs.add_trajectory(id,newtrajectory)
                newtrajectory = trajectory()
                isnewtrajectory = true
                
            else:
                if not isnewtrajectory:
                    newtrajectory.add_point(row[0], row[2],row[3]) 
                    
                else:
                    id = row[1]
                    newtrajectory = trajectory()
                    isnewtrajectory = false                    
                    
                
            
            
                
        #trajectories = {id, trajectory}

readcsvfile()
trajs.plot()