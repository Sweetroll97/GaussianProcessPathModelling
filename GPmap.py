import numpy as np
import statistics as ss
import pandas as ps
import itertools
import csv
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import random as rdm
import copy
import string
from scipy.spatial import distance as dst
from sklearn import gaussian_process
import sklearn.mixture as mixture
#from sklearn.gaussian_process import 
from matplotlib.colors import LogNorm

class trajectory:
    def __init__(self):
        self.points = np.empty([0,3])#np.array([], (float), ndmin=2)
        self.em_belongs_to_cluster = ""
        self.current_cluster_probability = -15000000
        self.probability = []
        
    def add_point(self,time,x,y):
        self.points = np.append(self.points, [[x,y,time]], axis=0)
    def get_probability(self,index):
        return self.probablility[index]/sum(self.probablility)
    def remove_point(self, time, x,y):
        self.points = np.delete(self.points, np.where(self.points == [[x,y,time]])[0], axis=0)
        if len(self.points) < 3: return True
        else: return False
        
    def get_length(self):
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1,p2 in zip(self.points, self.points[1:])])
    def get_lengths(self):
        return [np.linalg.norm(p1-p2) for p1,p2 in zip(self.points, self.points[1:])]
    def get_lengths_from_first_point(self):
        lengths = self.get_lengths()
        for idx,value in enumerate(lengths[:-1]):
            lengths[idx+1] = lengths[idx] + lengths[idx+1]
        return lengths
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
        #self.plotwx(centroids)
        
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
            
            #self.plotwx(centroids)
            totdist = 0
            for key in centroidsold:
                totdist += self.calc_distance(centroids[key], centroidsold[key])
            if totdist < 5:
                ischanging = False
                #self.plotclusters(clusters)
                return clusters
        
            
    def calc_mean_traj(self, keys):
        samplecount = len(keys)
        if samplecount == 1:
            return self.pathdict[keys[0]]
        elif samplecount == 0:
            raise ValueError('empty trajectory list')
        newmeantraj = trajectory()
        pointsum = sum([self.pathdict[value].points for value in keys])
        [newmeantraj.add_point(point[2]/samplecount, point[0]/samplecount, point[1]/samplecount) for point in pointsum]
        return newmeantraj
        
    def calc_distance(self,traj1 , traj2): #calculates an abstract distance beetween two trajectories
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1,p2 in zip(traj1.points, traj2.points)])

    def get_next_point(self, last_point, next_point, movement): 
        v = next_point-last_point
        return last_point+(v/(np.linalg.norm(v)))*movement
            
    def interpol_points(self, number_of_points = 10, plotresult = False):
        for id in tqdm(self.pathdict):    
            #Calculate total length of the function
            function_length = self.pathdict[id].get_length()
    
            #Calculate how much to move at each time step
            segment_length = function_length/(number_of_points-1) 
            
            #Calculate distance from first point to each other point
            li = self.pathdict[id].get_lengths_from_first_point()
            
            temp_traj = trajectory()            
            temp_traj.add_point(self.pathdict[id].points[0,2], self.pathdict[id].points[0,0],self.pathdict[id].points[0,1])
            j = 0
            passedlength = 0
            for i in range(1,number_of_points-1):
                while i*segment_length > li[j]:
                    if  j < len(li):
                        passedlength = li[j]
                        j+=1
                    else: raise ValueError('Interpolation error')
                last_point = self.pathdict[id].points[j]
                next_point = self.pathdict[id].points[j+1]
                rest_segment = i*segment_length-passedlength
                
                curr_interpol_point = self.get_next_point(last_point, next_point,rest_segment) 
                temp_traj.add_point(curr_interpol_point[2] ,curr_interpol_point[0],curr_interpol_point[1])
            temp_traj.add_point(self.pathdict[id].points[-1,2], self.pathdict[id].points[-1,0],self.pathdict[id].points[-1,1])
            
            if plotresult:
                self.plotcompare([self.pathdict[id], temp_traj], [0xFF0000, 0x00FF00])
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
                plt.plot(self.pathdict[key].points[:,0], self.pathdict[key].points[:,1], '#'+color)
            count += 1
        plt.show()         
    
    def plotwx(self, x):
        for key in x:
            plt.plot(x[key].points[:,0], x[key].points[:,1], "*")
        
        self.plot()
        
    def plot(self):
        
        plt.axis([-50000,50000.0,-50000.0,50000.0]) # xmin, xmax, ymin, ymax
        #plt.gca().set_autoscale_on(False)
        
        for id in self.pathdict:
            plt.plot(self.pathdict[id].points[:,0], self.pathdict[id].points[:,1])
            
        plt.show();
        
class GaussianMixtureModel():
        

    def __init__(self, trajs,K,sigma,M,treashold):
        self.trajs = copy.deepcopy(trajs)
        self.T = len(self.trajs.pathdict[next(iter(self.trajs.pathdict.items()))[0]].points)
        self.M = M
        self.beta = self.T/K
        self.K = K
        self.covariance = sigma
        self.EM_clusters = {}
        self.set_centroids(treashold)

        
    def calc_distance(self,traj1 , traj2): #calculates an abstract distance beetween two trajectories
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1,p2 in zip(traj1.points, traj2.points)])    
        
    def set_centroids(self,treshold = 200000):
        for t_key in self.trajs.pathdict:
            for cluster in range(self.M):
                self.trajs.pathdict[t_key].probability.append(0.0)
        #Generate M centroids
        counter = 0
        rdmkeys = rdm.sample(list(self.trajs.pathdict.keys()), self.M)                   
        istoclose = True
        while istoclose:
            istoclose = False
            for key1, key2 in zip(rdmkeys, rdmkeys[1:]):
                if self.calc_distance(self.trajs.pathdict[key1], self.trajs.pathdict[key2]) < treshold:
                    if counter > 100:
                        counter = 0
                        treashold-= 100
                        treashold = max(treashold, 0)
                    counter += 1
                    istoclose = True
                    rdmkeys.remove(key1)
                    treshold -= 100
                    if treshold < 0:
                        treshold = 0
                    rdmkeys.append(rdm.choice([newkey for newkey in list(self.trajs.pathdict.keys()) if newkey not in rdmkeys]))

        for i,key in enumerate(rdmkeys):
            number = str(i+1)
            self.EM_clusters["cluster "+number] = [key]
        self.plotclusters(self.EM_clusters)
        
        
    def plot_cluster(self,key= "##"):
        if key == "##": #Print all trajectories
            for traj_keys in self.EM_clusters:
                mean = np.array(self.get_mean_cluster_iT(traj_keys))
                plt.plot(mean[:,0],mean[:,1])
        else: #Print specific trajectory
            for traj in self.trajs.pathdict[key]:
                plt.plot(traj.points[:,0],trajecory.points[:,1]) 
        
    
    def get_mean_cluster_iT(self,key): #To be able to plot the whole cluster
        keys = self.EM_clusters[key]
        cluster_size = len(keys)
        mean_trajectory = []
        
        for i in range(0,self.T):
            sum_from_points = [0,0,0]
            for t_key in keys:
                sum_from_points[0] = (sum_from_points[0] + self.trajs.pathdict[t_key].points[i][0])
                sum_from_points[1] = (sum_from_points[1] + self.trajs.pathdict[t_key].points[i][1])
                sum_from_points[2] = (sum_from_points[2] + self.trajs.pathdict[t_key].points[i][2])
            sum_from_points[0] = sum_from_points[0]/len(keys)
            sum_from_points[1] = sum_from_points[1]/len(keys)
            sum_from_points[2] = sum_from_points[2]/len(keys)           
            mean_trajectory.append(sum_from_points)
        return mean_trajectory
    
    def plot_mean_clusters(self):
        for key in self.EM_clusters:
            mean_cluster = np.array(self.get_mean_cluster_iT(key))
            plt.plot(mean_cluster[:,0],mean_cluster[:,1])
        plt.show()
    
        
    def get_mean_cluster_i(self,key): # Get the "beta" cluster
        keys = self.EM_clusters[key]
        cluster_size = len(keys)        
        mean_trajectory = []
        sum_from_points = np.array([0.0,0.0,0.0])
        for i in range(0,self.T):
            for t_key in keys:
                a = self.trajs.pathdict[t_key].points[i]
                sum_from_points += self.trajs.pathdict[t_key].points[i]
            if i%(self.beta) == (self.beta - 1) and cluster_size > 0:
                sum_from_points = sum_from_points/(cluster_size*self.beta)
                mean_trajectory.append(sum_from_points)
                sum_from_points = np.array([0.0,0.0,0.0])
        
        return mean_trajectory
    
    #Get mu for the right point to compare with
    def get_mu(self,cluster_key,i,j):
        arr = np.array([self.get_mean_cluster_i(cluster_key)[j]])
        return [arr[0][0],arr[0][1]]
 
    #((x-mu)^T)*(Covariance^-1)*(x-mu)
    def get_scalar(self,matrix,vec):
        res = []
        for row in matrix.transpose():
            sum =0
            for col in row:
                for obj in vec:
                    sum += (col*obj)
            res.append(sum)
            
        return (res[0]**2+res[1]**2)

    def gaussian_2_dimension(self,point, mu):
        div = point[:-1] - mu[:-1]
        div_T = np.transpose(div)
        inv_covariance = np.linalg.inv(self.covariance)
        scalar = self.get_scalar(inv_covariance, div)
        determinant = abs(np.linalg.det(self.covariance))
        res = (1/(2*math.pi))*(1/math.sqrt(determinant))*math.exp((-1/2)*(scalar))
        return res
    
    def gaussian_1_dimension(self,point, mu):
        a = (np.linalg.norm(point[:-1]-mu)**2)
        b = (-1/(2*(self.covariance)**2))
        c = math.exp((-1/(2*(self.covariance)**2))*(np.linalg.norm(point[:-1]-mu)**2))
        res = math.log(c,math.e)
        return res    
        
            
    def nearest_cluster(self,t_key):
        cluster_key = ""
        highest_prob = self.trajs.pathdict[t_key].current_cluster_probability
        for i,key in enumerate(self.EM_clusters.keys()): 
            higest_prob = -150000
            probability = 0.0
            j = 0               
            for t in range(self.T):
                # The M-step
                mu = np.array(self.get_mu(key,t,j))
                if len(mu) == 0:
                    break    
                point = self.trajs.pathdict[t_key].points[t]
                # The E-step
                probability += self.gaussian_1_dimension(point,mu)
                if (t%(self.beta)) == (self.beta - 1):                        
                    j+=1
            self.trajs.pathdict[t_key].probability[i] = math.exp(probability)
            #print(math.exp(probability))
            if probability > highest_prob:
                highest_prob = probability
                cluster_key = key
            p =sum(self.trajs.pathdict[t_key].probability)
        return cluster_key
    

    def make_swap(self, old_cluster, new_cluster, t_key, traj):
        length = 1
        if old_cluster !="":
            length = len(self.EM_clusters[old_cluster])
            if length > 1:
                self.EM_clusters[old_cluster].remove(t_key)
                traj.em_belongs_to_cluster = new_cluster
        if length > 0:
            self.EM_clusters[new_cluster].append(t_key)
            traj.em_belongs_to_cluster = new_cluster
        self.EM_clusters[new_cluster].append(t_key)

        
    def GMM(self):
        swap = True
        while swap == True:
            swap = False
            number_of_swaps = 0
            for t_key in self.trajs.pathdict:
                traj = self.trajs.pathdict[t_key]
                old_cluster = traj.em_belongs_to_cluster
                new_cluster = self.nearest_cluster(t_key)
                if(old_cluster != new_cluster and new_cluster != ""):
                    self.make_swap(old_cluster, new_cluster, t_key, traj)
                    swap = True
                    number_of_swaps+=1
            #self.plotclusters(self.EM_clusters)
            self.plot_mean_clusters()
            print("done")
            print(number_of_swaps)
        

    def plotclusters(self, clusters):
        plt.axis([-50000,50000.0,-50000.0,50000.0]) # xmin, xmax, ymin, ymax
        
        numofclusters = len(clusters)
        colorrange = 10000
        colors = rdm.sample(range(colorrange), numofclusters)
        count = 0
        colors = []
       
        for i in range(1000):
            colors.append('%06X' % rdm.randint(0, 0xFFFFFF))
           
        for element in clusters:
            color = (colors[count])
           #color = cm.hot(float(colors[count])/colorrange)
           #color = cm.autumn(float(colors[count])/colorrange)
            for key in clusters[element]:
                plt.plot(self.trajs.pathdict[key].points[:,0], self.trajs.pathdict[key].points[:,1], '#'+color)
            count += 1
        plt.show()
    
            

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
                if(newtrajectory.get_length() > 8000):
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



readcsvfile(100)
trajs.filter_noise()
trajs.interpol_points(10)
#trajs.interpol_test(10)
trajs.plot()
#test = trajs.kmeansclustering(25)
K = 5 #"Hyperparameter"
M = 11 #number of clusters
treashold = 200000
var_x = 53601
var_y = 52012
#covariance = np.array([[var_x,math.sqrt(var_x)*math.sqrt(var_y)],[math.sqrt(var_x)*math.sqrt(var_y),var_y]])
covariance = 25100
GMM = GaussianMixtureModel(trajs,K,covariance,M,treashold)
GMM.GMM()
