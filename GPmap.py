import numpy as np
import statistics as ss
import pandas as ps
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import random as rdm
import copy
import string
from scipy.spatial import distance as dst
from sklearn import gaussian_process
#from sklearn.gaussian_process import 

class trajectory:
    def __init__(self):
        self.xs = np.array([], (float))
        self.ys = np.array([], (float))
        self.points = np.empty([0,3])#np.array([], (float), ndmin=2)
        
        self.timestamp = np.array([], (float))
        
    def add_point(self,time,x,y):
        self.xs = np.append(self.xs,x)
        self.ys = np.append(self.ys,y)
        self.points = np.append(self.points, [[x,y,time]], axis=0)
        self.timestamp = np.append(self.timestamp,time)
    def remove_point(self, time, x,y):
        #self.xs = np.delete(self.xs,np.where(self.xs == x))
        #self.ys = np.delete(self.ys,np.where(self.ys == y))
        #self.timestamp = np.delete(self.timestamp,np.where(self.timestamp == time))
        self.points = np.delete(self.points, np.where(self.points == [[x,y,time]])[0], axis=0)
        if len(self.points) < 3: return True
        else: return False
        
    def get_length(self):
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1,p2 in zip(self.points, self.points[1:])])
        
        
        
    def get_trajectory():     
        print("not implemented")
        
class trajectories:
     
    def __init__(self):
        self.pathdict = {}
        
    def add_trajectory(self,id, trajectory):
        self.pathdict[id] = trajectory
       
    def filter_noise(self, treshold = 500, plotit = False):
        minlength = treshold - 1
        isdead = False
        while minlength < treshold:
            for key in self.pathdict:
                backup = copy.deepcopy(self.pathdict[key])
                for point, nextpoint in zip(self.pathdict[key].points,self.pathdict[key].points[1:]):
                    if np.linalg.norm(nextpoint - point) < treshold:
                        if set(nextpoint) != set(self.pathdict[key].points[-1]):
                            isdead = self.pathdict[key].remove_point(nextpoint[2], nextpoint[0], nextpoint[1])
                        else:
                            isdead = self.pathdict[key].remove_point(point[2], point[0], point[1])
                        #if not isdead:
                        #    self.plotcompare([backup, self.pathdict[key]], [0x00FFFF, 0xFF00FF])
                if isdead:
                    break
                elif plotit is True:    
                    self.plotcompare([backup, self.pathdict[key]], [0x00FFFF, 0xFF00FF])
            if isdead: 
                self.pathdict.pop(key)
                isdead = False
            else:
                minlength = min([np.linalg.norm(p1-p2) for p1,p2 in zip(self.pathdict[key].points, self.pathdict[key].points[1:])])
                
    def kmeansclustering(self, k, treshold = 200000):
        #Generate k centroids
        rdmkeys = rdm.sample(list(self.pathdict.keys()), k)
        
        #fix for centroids that is to close
        istoclose = True
        while istoclose:
            istoclose = False
            for key1, key2 in zip(rdmkeys, rdmkeys[1:]):
                if self.calc_distance(self.pathdict[key1], self.pathdict[key2]) < treshold:
                    istoclose = True
                    rdmkeys.remove(key1)
                    treshold -= 100
                    if treshold < 0:
                        treshold = 0
                    rdmkeys.append(rdm.choice([newkey for newkey in list(self.pathdict.keys()) if newkey not in rdmkeys]))
        
        centroids = {}
        for key in rdmkeys:
            centroids[''.join(rdm.choices(string.ascii_uppercase + string.digits, k=5))] = copy.deepcopy(self.pathdict[key])
        self.plotwx(centroids)
        
        #centroidsold = {}
        ischanging = True
        while ischanging:
            centroidsold = copy.deepcopy(centroids)
            #connect centroids to closest trajectories to form clusters        
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
        
            #reassign centroids with means of respective cluster
            for key in centroids:
                centroids[key] = self.calc_mean_traj(clusters[key])
            
            self.plotwx(centroids)
            totdist = 0
            for key in centroidsold:
                totdist += self.calc_distance(centroids[key], centroidsold[key])
            if totdist < 5:
                ischanging = False
                self.plotclusters(clusters)
                return clusters
            
    def calc_mean_traj(self, keys):
        samplecount = len(keys)
        if samplecount == 1:
            return self.pathdict[keys[0]]
        elif samplecount == 0:
            return []
        newmeantraj = trajectory()
        pointsum = sum([self.pathdict[value].points for value in keys])
        [newmeantraj.add_point(point[2]/samplecount, point[0]/samplecount, point[1]/samplecount) for point in pointsum]
        return newmeantraj
        
    def calc_distance(self,traj1 , traj2): #calculates an abstract distance beetween two trajectories
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1,p2 in zip(traj1.points, traj2.points)])

    def get_next_point(self, last_point, next_point, movement): 
        v = next_point-last_point
        return last_point+(v/(np.linalg.norm(v)))*movement
    
    def interpol_test(self, N):
        for id in tqdm(self.pathdict):
            WL = self.pathdict[id].get_length()/(N-1)
            L = [np.linalg.norm(p1-p2) for p1,p2 in zip(self.pathdict[id].points, self.pathdict[id].points[1:])]
            newtraj = trajectory();
            newtraj.add_point(self.pathdict[id].points[0][2], self.pathdict[id].points[0][0], self.pathdict[id].points[0][1])
            pointofinterest=0
            sumofpassedpoints = 0
            disttonextpoint = copy.deepcopy(L)
            
            for i in range(1, N-1):
                
                while WL*i > disttonextpoint[0]:
                    pointofinterest+=1
                    temp = disttonextpoint.pop(0)
                    disttonextpoint[0] += temp
                    sumofpassedpoints = sum(L[0:pointofinterest])
                
                newpoint = self.get_next_point(self.pathdict[id].points[pointofinterest], self.pathdict[id].points[pointofinterest+1], WL*i-sumofpassedpoints)
                newtraj.add_point(newpoint[2], newpoint[0], newpoint[1]) 
            newtraj.add_point(self.pathdict[id].points[-1][2], self.pathdict[id].points[-1][0], self.pathdict[id].points[-1][1])
            plt.plot(self.pathdict[id].points[:,0], self.pathdict[id].points[:,1], "b*")
            plt.plot(self.pathdict[id].points[:,0],self.pathdict[id].points[:,1])
            plt.plot(newtraj.points[:,0],newtraj.points[:,1])
            plt.scatter(newtraj.points[:,0],newtraj.points[:,1],color='orange')
            plt.show()    
            self.pathdict[id] = newtraj
            
    
            
   
            
    def get_length_to_each_point(self,id):
        last_object = [self.pathdict[id].points[0]]
        dist = 0.0
        li = [0.0]
        i = 0
        for point in self.pathdict[id].points:
            if i != 0:
                dist += np.linalg.norm(point[:2]-last_object[:2])
                li.append(dist)
            last_object = point                
            i+=1
        print(li)
        return li    
            
    def interpol_points(self, number_of_segments= 10, plotresult = False):
        for id in tqdm(self.pathdict):    
            #Calculate total length of the function
            funcion_length = self.pathdict[id].get_length()
    
            #Calculate how much to move at each time step
            segment_length = funcion_length/number_of_segments 
            
            #Calculate distance from first point to each other point
            li = self.get_length_to_each_point(id)
            
            temp_traj = trajectory()            
            total_length = 0.0
            j = 1
            for i in range(0,number_of_segments+1):
                while i*segment_length > li[j]+0.0002:
                    if  j < len(li)-1:
                        j+=1
                last_point = self.pathdict[id].points[j-1]
                next_point = self.pathdict[id].points[j]
                rest_segment = i*segment_length-li[j-1]
                curr_interpol_point = self.get_next_point(last_point, next_point,rest_segment) 
                temp_traj.add_point(curr_interpol_point[2] ,curr_interpol_point[0],curr_interpol_point[1])
            if plotresult:
                self.plotcompare([self.pathdict[id], temp_traj])
            self.pathdict[id] = temp_traj
                    
    
    
    def generate_guassian_processes(self):
        params = gaussian_process.kernels.Hyperparameter('theta',float,3,3) #testing stage
        
    
    def plotcompare(self, listoftrajs, colors = []):
        
        #= rdm.sample(range(colorrange), len(listoftrajs))
        count = 0
        if len(colors) is 0:
            for i in range(len(listoftrajs)):
                colors.append('%06X' % rdm.randint(0, 0xEEEEEE))
        else:
            for idx,value in enumerate(colors):
                colors[idx] = '%06X' % value
       
        for traj in listoftrajs:     
            print(".............................")    
            print("Goal:")
            print("x: ",traj.points[-1,0])
           #print("x: ",traj.points[-1,0])
            print("y: ",traj.points[-1,1])
           # print("y: ",temp_traj.points[-1,1])            
            print(".............................")
            paint = (colors[count])
            count += 1
            
            plt.plot(traj.points[:,0],traj.points[:,1], '#'+paint)
            plt.scatter(traj.points[:,0],traj.points[:,1], color='#'+paint)
            
           # plt.plot(temp_traj.points[:,0],temp_traj.points[:,1])
           # plt.scatter(temp_traj.points[:,0],temp_traj.points[:,1],color='orange')
        plt.show()
        colors.clear()
        
    def plotclusters(self, clusters):
        plt.axis([-50000,50000.0,-50000.0,50000.0]) # xmin, xmax, ymin, ymax
        
        numofclusters = len(clusters)
        colorrange = 10000
        colors = rdm.sample(range(colorrange), numofclusters)
        count = 0
        colors = []
       
        for i in range(10):
            colors.append('%06X' % rdm.randint(0, 0xFFFFFF))
           
        for element in clusters:
            color = (colors[count])
           #color = cm.hot(float(colors[count])/colorrange)
           #color = cm.autumn(float(colors[count])/colorrange)
            for key in clusters[element]:
                plt.plot(self.pathdict[key].xs, self.pathdict[key].ys, '#'+color)
            count += 1
        plt.show()         
    
    def plotwx(self, x):
        for id in x:
            plt.plot(x[id].xs, x[id].ys, "*")
        
        self.plot()
        
    def plot(self):
        
        plt.axis([-50000,50000.0,-50000.0,50000.0]) # xmin, xmax, ymin, ymax
        #plt.gca().set_autoscale_on(False)
        
        for id in self.pathdict:
            plt.plot(self.pathdict[id].points[:,0], self.pathdict[id].points[:,1])
            
        plt.show();

trajs = trajectories()
    
def readcsvfile(numoftrajstoread=0):
    global trajs;
    
    with open('testfile.csv') as data:
        data = csv.reader(data, delimiter=',')
        trajnr = 0
        
        isnewtrajectory = True
        id = 0
        for row in data:
            if(row[0] == '###'):
                if(newtrajectory.get_length() > 5000):
                    trajs.add_trajectory(id,newtrajectory)
                    trajnr = trajnr + 1;
                if numoftrajstoread is not 0 and trajnr >= numoftrajstoread:
                    break;
                newtrajectory = trajectory()
                isnewtrajectory = True
                
            else:
                if not isnewtrajectory:
                    newtrajectory.add_point(float(row[0]), int(row[2]),int(row[3])) 
                    
                else:
                    
                    id = row[1]
                    newtrajectory = trajectory()
                    newtrajectory.add_point(float(row[0]), int(row[2]),int(row[3])) 
                    isnewtrajectory = False                    



readcsvfile(10)
trajs.filter_noise()
trajs.interpol_points(10)
#trajs.interpol_test(10)

#keys = []
#for key in trajs.pathdict:
#    keys.append(key)    
#trajs.calc_distance(trajs.pathdict[keys[0]], trajs.pathdict[keys[1]])
#trajs.plot()

trajs.kmeansclustering(3)