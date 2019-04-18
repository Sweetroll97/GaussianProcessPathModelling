import numpy as np
import statistics as ss
import pandas as ps
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

class trajectory:
    def __init__(self):
        self.xs = np.array([], (float))
        self.ys = np.array([], (float))
        self.points = np.empty([0,2])#np.array([], (float), ndmin=2)
        
        self.timestamp = np.array([], (float))
        
    def add_point(self,time,x,y):
        self.xs = np.append(self.xs,x)
        self.ys = np.append(self.ys,y)
        self.points = np.append(self.points, [[x,y]], axis=0)
        self.timestamp = np.append(self.timestamp,time)
        
        
    def get_length(self):
        #return np.linalg.norm(np.matrix(self.xs)-np.matrix(self.ys))
        return sum([np.linalg.norm(p1-p2) for p1,p2 in zip(self.points, self.points[1:])])
        
        #for valx, valy in xs,ys:
        #    total_length += 
        
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
        for i in range(len(array_x1)-1):
            total_length+= self.get_traveled_dist(array_x1[i],array_y1[i],array_x1[i+1],array_y1[i+1])
        return total_length
        
    def get_next_point(self,curr_index, matrix, distance,):
        
        for i in range(len(array_x1)-1):
            total_length+= self.get_traveled_dist(array[i],matrix.item(0),array[i],matrix.item)
        return total_length    
    
    def get_next_point(self, last_point, curr_point, next_point, movement):
        return curr_point+((next_point-last_point)/np.linalg.norm((next_point-last_point)))*movement
    
    def get_next_point(self, last_point, next_point, movement): 
        v = next_point-last_point
        return last_point+v/np.linalg.norm(v)*movement
    
    def interpol_test(self, N):
        for id in tqdm(self.pathdict):
            WL = self.pathdict[id].get_length()/(N-1)
            L = [np.linalg.norm(p1-p2) for p1,p2 in zip(self.pathdict[id].points, self.pathdict[id].points[1:])]
          
            newtraj = trajectory();
            newtraj.add_point(self.pathdict[id].timestamp, self.pathdict[id].points[0][0], self.pathdict[id].points[0][1])
            pointofinterest=0
            walkdist = 0
            for i in range(1, N-1):            
                if i*WL <= L[0]:
                    walkdist += WL
                    
                while(i*WL > L[0]):
                    temp = L.pop(0)
                    L[0] += temp
                    pointofinterest+=1
                    walkdist = L[0]-i*WL
                
                newpoint = self.get_next_point(self.pathdict[id].points[pointofinterest], self.pathdict[id].points[pointofinterest+1], walkdist)
                newtraj.add_point(self.pathdict[id].timestamp[pointofinterest], newpoint[0], newpoint[1]) 
            newtraj.add_point(self.pathdict[id].timestamp, self.pathdict[id].points[-1][0], self.pathdict[id].points[-1][1])
            self.pathdict[id] = newtraj       
    def interpol_points(self):
        number_of_observations = 30
        #for id in self.pathdict:
         #   if len(self.pathdict[id].xs) > number_of_observations:
          #      number_of_observations = len(self.pathdict[id].xs)-1
                
        for id in tqdm(self.pathdict):
            
            if(len(self.pathdict[id].xs) == 0):
                continue            
            #Calculate total length of the function
            funcion_length = self.get_function_length(self.pathdict[id].xs, self.pathdict[id].ys)
            
            #Calculate how much to move at each time step
            num_of_segments = funcion_length/number_of_observations 
            
            #Creation of temporary lists
            interp_xs = np.array([], (float))
            interp_ys = np.array([], (float))    
            
            #Get 
            num_of_points = len(self.pathdict[id].xs)-1            
            i = 0
        
            curr_interpol_point = np.matrix(( self.pathdict[id].xs[i],self.pathdict[id].ys[i])) 
            global_distance = 0.0
            
            while global_distance <= funcion_length:
                #If we stil not on the last point
                if i <= num_of_points-1:
                    last_point =  np.matrix(( [self.pathdict[id].xs[i],self.pathdict[id].ys[i]]))
                    next_point = np.matrix(([self.pathdict[id].xs[i+1],self.pathdict[id].ys[i+1]]))
                    point_distance = np.linalg.norm((next_point-last_point))
                    local_distance = 0.0
                 
                    while local_distance < point_distance:
                        
                        #Add the x and y values to the lists
                        interp_xs = np.append(interp_xs,curr_interpol_point.item(0))
                        interp_ys = np.append(interp_ys,curr_interpol_point.item(1))
                        
                        #interpolate a new point                        
                        curr_interpol_point = self.get_next_point(last_point, curr_interpol_point, next_point, num_of_segments) 
                        
                        #Add the walked distance to current distance
                        local_distance += np.linalg.norm((curr_interpol_point-last_point))
                        #if curr_distance < point_distance:
                        last_point = curr_interpol_point
                elif i == num_of_points:
                    interp_xs = np.append(interp_xs,self.pathdict[id].xs[num_of_points])
                    interp_ys = np.append(interp_ys,self.pathdict[id].ys[num_of_points])  
                    
                #else:
                    #print(i)
                    
                global_distance += local_distance
                i+=1
            plt.scatter(self.pathdict[id].xs,self.pathdict[id].ys)    
            plt.scatter(interp_xs,interp_ys)
            
        plt.show()

        
    def plot(self):
        
        plt.axis([-50000,50000.0,-50000.0,50000.0]) # xmin, xmax, ymin, ymax
        plt.gca().set_autoscale_on(False)
        
        for id in self.pathdict:
            plt.scatter(self.pathdict[id].xs, self.pathdict[id].ys)
            
        plt.show();

trajs = trajectories()
 
        
def readcsvfile(numoftrajstoread=0):
    global trajs;
    
    with open('testfile1.csv') as data:
        data = csv.reader(data, delimiter=',')
        trajnr = 0
        
        isnewtrajectory = True
        id = 0
        for row in data:
            if(row[0] == '###'):
                #if(newtrajectory.get_length() > 1000):
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
                    newtrajectory.add_point(row[0], int(row[2]),int(row[3])) 
                    isnewtrajectory = False                    

readcsvfile(1)
trajs.interpol_test(10)

#trajs.interpol_points()
#print(len(trajs.pathdict))
trajs.plot()


