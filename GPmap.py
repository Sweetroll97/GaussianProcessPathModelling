import numpy as np;
import statistics as ss;
import pandas as ps;
import csv;

class trajectory:
    xs = np.array([], (float));
    ys = np.array([], (float));
    timestamp = np.array([], (float));
    
    #def __init__(self, id):
    #    self.id = id;
        
    def add_point(self,x,y, time):
        self.xs.append(x);
        self.ys.append(y);
        self.timestamp.append(time);
        
    def get_trajectory():     
        print("not implemented");
        
class trajectories:
     
    def __init__(self):
        self.pathdict = {};
        
    def add_trajectory(self,id, trajectory):
        self.pathdict[id] = trajectory;
        
def readcsvfile():
    with open('testfile.csv') as data:
        data = csv.reader(data, delimiter=',')
        linenr = 0;
        for row in data:
            newtrajectory = trajectory();
            newtrajectory.add_point(row[0]
            
        #trajectories = {id, trajectory}

#readcsvfile();