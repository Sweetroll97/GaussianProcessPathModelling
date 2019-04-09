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
        
    def add_interpol_point(self,time,x,y):
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
        
    def interpol_points(self):
        max_length = 10000
        for id in self.pathdict:
            if len(self.pathdict[id].xs) > max_length:
                max_length = len(self.pathdict[id].xs)
        for id in self.pathdict:
            if len(self.pathdict[id].xs) > 0:
                #Set first values
                curr_value = self.pathdict[id].xs[0]
                y_value = self.pathdict[id].ys[0]
                
                #Get max and min values for x
                local_max = max(self.pathdict[id].xs)
                local_min = min(self.pathdict[id].xs)
                
                #Calculate movement
                movement = (float)(local_max-local_min)/(max_length-1)
                
                if(movement == 0):
                    continue
                interp_xs = np.array([], (float))
                interp_ys = np.array([], (float))   
                #print("Max: "+str(local_max)+ "Curr_value:" + str(curr_value))
                while curr_value <= local_max and curr_value >= local_min:
                    interp_ys=np.append(interp_ys,y_value)
                    interp_xs=np.append(interp_xs,curr_value)
                    #Update next step
                    curr_value = curr_value + movement
                    y_value = np.interp(curr_value,self.pathdict[id].xs,self.pathdict[id].ys)
                #Set new interpolated trajectory 
                self.pathdict[id].xs = interp_xs
                self.pathdict[id].ys = interp_ys
                
        
    def plot(self):
        
        plt.axis([-50000,50000.0,-50000.0,50000.0]) # xmin, xmax, ymin, ymax
        plt.gca().set_autoscale_on(False)
        
        for id in self.pathdict:
            plt.plot(self.pathdict[id].xs, self.pathdict[id].ys)
            
        plt.show();

trajs = trajectories()
 
        
def readcsvfile(numoftrajstoread=0):
    global trajs;
    
    with open('testfile.csv') as data:
        data = csv.reader(data, delimiter=',')
        trajnr = 0
        
        isnewtrajectory = True
        id = 0
        #newtrajectory = trajectory();
        for row in data:
            if(row[0] == '###'):
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
                    
                
            #if linenr >= 10:
             #   break;
            
                
        #trajectories = {id, trajectory}

readcsvfile(500)
trajs.interpol_points()
#print(len(trajs.pathdict))
trajs.plot()