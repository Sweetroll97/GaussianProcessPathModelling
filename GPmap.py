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
        
    def get_traveled_dist(self,x1,y1,x2,y2):
        return ((x1-x2)**2+(y1-y2)**2)**.5
        
    def get_function_length(self,array_x1,array_y1):
        total_length = 0.0
        for i in range(len(array_x1)-2):
            total_length+= self.get_traveled_dist(array_x1[i],array_y1[i],array_x1[i+1],array_y1[i])
        return total_length
        
    
    def get_next_point(self, last_point, curr_point, next_point, movement):
        return curr_point+((next_point-last_point)/np.linalg.norm((next_point-last_point)))*movement
        
        
    def interpol_points(self):
        number_of_observations = 307-1
        #for id in self.pathdict:
         #   if len(self.pathdict[id].xs) > max_length:
          #      max_length = len(self.pathdict[id].xs)
        for id in self.pathdict:
            #Set initual values
            curr_interpol_point = np.matrix(( self.pathdict[id].xs[0],self.pathdict[id].ys[0]))
            
            #Calculate total length of teh function
            funcion_length = self.get_function_length(self.pathdict[id].xs, self.pathdict[id].ys)
            
            #Calculate how much to move at each time step
            Wi= funcion_length/number_of_observations                                    
            if(Wi == 0):
                continue
            #Creation of temporary lists
            interp_xs = np.array([], (float))
            interp_ys = np.array([], (float))
            
            curr_size = len(self.pathdict[id].xs)            
            i = 0
            while i*Wi <= funcion_length:
                #LOCAL FUNCTION          
                last_point = next_point = np.matrix(( [self.pathdict[id].xs[i],self.pathdict[id].ys[i]]))
                if i != curr_size-1:
                    next_point = np.matrix(([self.pathdict[id].xs[i+1],self.pathdict[id].ys[i+1]]))
                    point_distance = np.linalg.norm((next_point-last_point))
                    curr_distance = 0.0
                 
                    while curr_distance <= point_distance:
                        
                        #Calculate new point and direction
                        interp_xs = np.append(interp_xs,curr_interpol_point.item(0))
                        interp_ys = np.append(interp_ys,curr_interpol_point.item(1))
                        temp_point = curr_interpol_point
                        curr_interpol_point = self.get_next_point(temp_point, curr_interpol_point, next_point, Wi)  
                        curr_distance += np.linalg.norm((curr_interpol_point-temp_point))
                        print(curr_interpol_point)
                        
                i+=1
                print(last_point)
                
                
        
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

readcsvfile(5)
print(str(trajs.get_traveled_dist(3,4,0,0)))
trajs.interpol_points()
#print(len(trajs.pathdict))
trajs.plot()