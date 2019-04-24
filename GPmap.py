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
        self.points = np.empty([0,2])
        self.timestamp = np.array([], (float))
        
    def add_point(self,time,x,y):
        self.xs = np.append(self.xs,x)
        self.ys = np.append(self.ys,y)
        self.points = np.append(self.points, [[x,y]], axis=0)
        self.timestamp = np.append(self.timestamp,time)
        
    def get_length(self):
        return sum([np.linalg.norm(p1-p2) for p1,p2 in zip(self.points, self.points[1:])])
        
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
        for i in range(len(array_x1)-1):
            total_length+= self.get_traveled_dist(array_x1[i],array_y1[i],array_x1[i+1],array_y1[i+1])
        return total_length
        
    def get_right_point(self,curr_index, matrix, distance,):
        
        for i in range(len(array_x1)-1):
            total_length+= self.get_traveled_dist(array[i],matrix.item(0),array[i],matrix.item)
        return total_length    
    
    def get_next_point(self, last_point, curr_point, next_point, movement):
        direct = ((next_point-last_point)/np.linalg.norm((next_point-last_point)))
        return curr_point+direct*movement

    def get_length_to_each_point(self,id):
        last_object = [self.pathdict[id].xs[0],self.pathdict[id].ys[0]]
        dist = 0.0
        li = [0.0]
        i = 0
        for point in self.pathdict[id].points:
            if i != 0:
                dist += self.get_traveled_dist(last_object[0],last_object[1],point[0],point[1])
                li.append(dist)
            last_object = point                
            i+=1
        print(li)
        return li
        
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
            segment_length = funcion_length/number_of_observations 

            #Creation of temporary lists
            interp_xs = np.array([], (float))
            interp_ys = np.array([], (float))    

            #Get num of points 
            number_of_points = len(self.pathdict[id].xs)-1
            
            #Set init values
            i = 0
            curr_interpol_point = last_point = np.matrix(( self.pathdict[id].xs[i],self.pathdict[id].ys[i]))            
            global_distance = 0.0
            point_distance = 0.0
            rest_segment = 0.0
            temp_traj = trajectory()
            temp_traj.add_point(self.pathdict[id].timestamp,self.pathdict[id].xs[0],self.pathdict[id].ys[0])
            while global_distance < funcion_length:

                #Set the last point visited from
                if i <= number_of_points-1:
                    last_point = np.matrix(([self.pathdict[id].xs[i],self.pathdict[id].ys[i]]))
                    next_point = np.matrix(([self.pathdict[id].xs[i+1],self.pathdict[id].ys[i+1]]))
                    point_distance = np.linalg.norm((next_point-last_point))                    
                    local_distance = 0.0                    
                    while local_distance < point_distance and i < number_of_points:
                        if rest_segment > 0.00000001:
                            if rest_segment > point_distance:    #if rest is bigger then point distance
                                if abs(rest_segment-point_distance) < 0.1:
                                    curr_interpol_point = self.get_next_point(last_point, last_point, next_point, rest_segment)     
                                    temp_traj.add_point(self.pathdict[id].timestamp,curr_interpol_point.item(0),curr_interpol_point.item(1))
                                    
                                rest_segment = rest_segment-point_distance
                                local_distance = point_distance
                                i+=1
                            else:      #If rest smaller then point distance
                                curr_interpol_point = self.get_next_point(last_point, last_point, next_point, rest_segment)  
                                temp_traj.add_point(self.pathdict[id].timestamp,curr_interpol_point.item(0),curr_interpol_point.item(1))
                                local_distance+= rest_segment
                                rest_segment = 0.0
                             #If next step not the first and behind next point   
                        elif (local_distance+segment_length)> point_distance:
                            if abs(local_distance+segment_length-point_distance) < 0.1:
                                curr_interpol_point = self.get_next_point(last_point, curr_interpol_point, next_point, segment_length)      
                                temp_traj.add_point(self.pathdict[id].timestamp,curr_interpol_point.item(0),curr_interpol_point.item(1))
                            rest_segment = (local_distance+segment_length)-point_distance
                            local_distance = point_distance
                            i+=1
                            # If next step before next point
                        else:
                            curr_interpol_point = self.get_next_point(last_point, curr_interpol_point, next_point,segment_length)    
                            temp_traj.add_point(self.pathdict[id].timestamp,curr_interpol_point.item(0),curr_interpol_point.item(1))
                            local_distance+= segment_length

                global_distance += local_distance
                if id != '9190802':
                    plt.plot(self.pathdict[id].xs,self.pathdict[id].ys)
                    plt.plot(temp_traj.points[:,0],temp_traj.points[:,1])
                    plt.scatter(temp_traj.points[:,0],temp_traj.points[:,1],color='orange')
                    plt.show()                
                    

            print(".............................")
            
            print("Goal:")
            print("x: ",self.pathdict[id].xs[-1])
            print("x: ",temp_traj.xs[-1])
            print("y: ",self.pathdict[id].ys[-1])
            print("y: ",temp_traj.ys[-1])            
            print(".............................")
            plt.plot(self.pathdict[id].xs,self.pathdict[id].ys)
            plt.plot(temp_traj.points[:,0],temp_traj.points[:,1])
            plt.scatter(temp_traj.points[:,0],temp_traj.points[:,1],color='orange')
            plt.show()      
            
    def interpol_points2(self):
        number_of_observations = 10
        
        #for id in self.pathdict:
         #   if len(self.pathdict[id].xs) > number_of_observations:
          #      number_of_observations = len(self.pathdict[id].xs)-1

        for id in tqdm(self.pathdict):

            if(len(self.pathdict[id].xs) == 0):
                continue            
            #Calculate total length of the function
            funcion_length = self.get_function_length(self.pathdict[id].xs, self.pathdict[id].ys)

            #Calculate how much to move at each time step
            segment_length = funcion_length/number_of_observations 
            li = self.get_length_to_each_point(id)
            temp_traj = trajectory()            
            total_length = 0.0
            j = 1
            for i in range(0,number_of_observations+1):
                while i*segment_length > li[j]+0.0002:
                    if  j < len(li)-1:
                        j+=1
                last_point = self.pathdict[id].points[j-1]
                next_point = self.pathdict[id].points[j]
                rest_segment = i*segment_length-li[j-1]
                curr_interpol_point = self.get_next_point(last_point, last_point, next_point,rest_segment) 
                temp_traj.add_point(self.pathdict[id].timestamp,curr_interpol_point.item(0),curr_interpol_point.item(1))
            print(".............................")
            
            print("Goal:")
            print("x: ",self.pathdict[id].xs[-1])
            print("x: ",temp_traj.xs[-1])
            print("y: ",self.pathdict[id].ys[-1])
            print("y: ",temp_traj.ys[-1])            
            print(".............................")
            plt.plot(self.pathdict[id].xs,self.pathdict[id].ys)
            plt.plot(temp_traj.points[:,0],temp_traj.points[:,1])
            plt.scatter(temp_traj.points[:,0],temp_traj.points[:,1],color='orange')
            plt.show()              
                       

        
    def plot(self):
        
        plt.axis([-50000,50000.0,-50000.0,50000.0]) # xmin, xmax, ymin, ymax
        plt.gca().set_autoscale_on(False)
        
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
                if newtrajectory.get_length() > 1000:
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

readcsvfile(200)
trajs.interpol_points2()



