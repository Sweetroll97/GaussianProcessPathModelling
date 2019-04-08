import numpy as np
import statistics as ss
import pandas as ps
import csv
import matplotlib.pyplot as plt
 

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
    def plot(self):

        for id in self.pathdict:
            plt.scatter(self.pathdict[id].xs, self.pathdict[id].ys)
            
        plt.show();

trajs = trajectories()
 
        
def readcsvfile():
    global trajs;
    
    with open('testfile.csv') as data:
        data = csv.reader(data, delimiter=',')
        linenr = 0
        
        isnewtrajectory = True
        id = 0
        #newtrajectory = trajectory();
        for row in data:
            if(row[0] == '###'):
                trajs.add_trajectory(id,newtrajectory)
                newtrajectory = trajectory()
                isnewtrajectory = True
                
            else:
                if not isnewtrajectory:
                    newtrajectory.add_point(row[0], row[2],row[3]) 
                    
                else:
                    id = row[1]
                    newtrajectory = trajectory()
                    isnewtrajectory = False                    
                    
                
            linenr = linenr + 1
            if linenr >= 10:
                break;
            
                
        #trajectories = {id, trajectory}

readcsvfile()
#print(len(trajs.pathdict))
trajs.plot()