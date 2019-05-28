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
        self.probability = []
        self.weighted_probability = []
        
    def get_highest_model_index(self):
        return self.probability.index(min(self.probability))
    
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
       
    def filter_noise(self, treshold = 1100, plotit = False):
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
        
        
class Model():
    def __init__(self,list_of_points):
        self.traj = copy.deepcopy(list_of_points)
class GaussianMixtureModel():
        

    def __init__(self, trajs,K,covariance,M,treashold, init_points = None):
        self.trajs = copy.deepcopy(trajs)
        self.T = len(self.trajs.pathdict[next(iter(self.trajs.pathdict.items()))[0]].points)
        self.M = M
        self.beta = self.T/K
        self.K = K
        self.covariance = covariance
        self.colors = []
        self.set_cluster_colours()
        self.importance = []
        self.models = []
        self.set_centroids(treashold)
        self.plotta()
        
      
    def set_cluster_colours(self):
        for i in range(self.M):
            self.colors.append('%06X' % rdm.randint(0, 0xFFFFFF))       

        
    def calc_distance(self,traj1 , traj2): #calculates an abstract distance beetween two trajectories
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1,p2 in zip(traj1.points, traj2.points)])    
        
    def set_centroids(self,treshold = 20000):
        #Set all
        num_of_trajectories = len(self.trajs.pathdict)
        for i in range(self.M):
            for t_key in self.trajs.pathdict:
                self.trajs.pathdict[t_key].weighted_probability.append(0.0)
                self.trajs.pathdict[t_key].probability.append(0.0)                                
             
        #Generate M centroids
        counter = 0
        rdmkeys = rdm.sample(list(self.trajs.pathdict.keys()), self.M)                   
        istoclose = True
        while istoclose:
            istoclose = False
            for key1, key2 in zip(rdmkeys, rdmkeys[1:]):
                if self.calc_distance(self.trajs.pathdict[key1], self.trajs.pathdict[key2]) < treshold:
                    if counter > 50:
                        counter = 0
                        treshold-= 1
                        treshold = max(treshold, 0)
                    counter += 1
                    istoclose = True
                    rdmkeys.remove(key1)
                    treshold -= 1
                    if treshold < 0:
                        treshold = 0
                    rdmkeys.append(rdm.choice([newkey for newkey in list(self.trajs.pathdict.keys()) if newkey not in rdmkeys]))
        #Set initial probability for the centroids
        diviates = [[155.0,230.0,0.0],[-125.0,-230.0,0.0],[-125.0,-230.0,0.0]]
        
        for i,key in enumerate(rdmkeys):
                self.trajs.pathdict[key].probability[i] = 0.0
                self.trajs.pathdict[key].weighted_probability[i] = 0.0
                #model_i = Model(self.trajs.pathdict[key].points+diviate)
                model_i = Model(self.trajs.pathdict[key].points)
                self.models.append(model_i.traj)
            
        #global models_created
        #for i in range(self.M):
            #model_i = Model(models_created[i].points)
            #self.models.append(model_i.traj+np.array(diviates[i]))
        self.plotclusters(True)


     
    def get_mean_cluster_i(self,m): # Get the "beta" cluster       
        mean_trajectory = np.empty([0,3])
        for i in range(0,self.T):
            sum_from_points = np.array([0.0,0.0,0.0])
            for t_key in self.trajs.pathdict:
                point = self.trajs.pathdict[t_key].points[i]
                #point_prob = self.trajs.pathdict[t_key].weighted_probability[m]
                point_prob = self.trajs.pathdict[t_key].probability[m]
                sum_from_points += point_prob*point
                
            if (i % self.beta) == (self.beta - 1):
                sum_from_points = sum_from_points/self.beta
                mean_trajectory = np.append(mean_trajectory,[sum_from_points], axis=0)
                
        return mean_trajectory[:,:2]/(len(self.trajs.pathdict)/self.M)
        #return mean_trajectory[:,:2]/self.M
        #return mean_trajectory[:,:2]#*len(self.trajs.pathdict)/self.M/(len(self.trajs.pathdict)/self.M)
        
        
    def get_mu_point(self,m,j):
        arr = np.array([self.get_mean_cluster_i(m)[j]])
        return [arr[0][0],arr[0][1]] 

    
    def gaussian_1_dimension(self,point,m,j):        
        c = math.exp((-1/(2*(self.covariance)**2))*(np.linalg.norm(point[:-1]-self.models[m][j,:2])**2))
        if np.isnan(c) or c < (2.2250738585072014e-308)**(1/self.T):
            res = (2.2250738585072014e-308)**(1/self.T)
        else:
            res = c
        return res   
    
            

        #return max_diviation,num_of_changes
  
    def set_weights(self,bayesian_matrix):
        max_diviation = 0.0
        sum_ti_importance = sum(bayesian_matrix)
        sum_c = sum(sum_ti_importance)# The total sum of all values in the matrix P(Cm)+P(Cm+1)+...+P(CM)
        p_Cms = []
        for m in range(self.M):#Compute all P(C_m)
            sum_c_m= sum(bayesian_matrix[m]) # sum C_m
            P_Cm = (sum_c_m/sum_c)
            p_Cms.append(P_Cm)
        p_Cms = np.array(p_Cms)  
        for t_i,t_key in enumerate(self.trajs.pathdict):
            """Calculate 
            P(Cm\t_i) = P(Cm)*P(Cm\ti)/ (P(Cm)*P(Cm\ti)+P(Cm+1)*P(Cm+1\ti)+...+P(CM)*P(CM\ti))
            for each cluster
            """
            traj = self.trajs.pathdict[t_key]
            for m in range(self.M):  #P(Cm\traj_i)
                numerator = p_Cms[m]*bayesian_matrix[m][t_i]         #P(Cm)*P(Cm\ti)
                denominator = sum((bayesian_matrix[:,t_i]*p_Cms))    #P(Cm)*P(Cm\ti)+P(Cm+1)*P(Cm+1\ti)+...+P(CM)*P(CM\ti)
                quote = numerator/denominator
                diviation = abs(traj.probability[m]-quote)
                if diviation > max_diviation:
                    max_diviation = diviation
                traj.probability[m] = quote       #P(Cm\ti)
                #bayesian_matrix[m][t_i] = quote
                
                

        new_C_sum = []
        for m,row in enumerate(bayesian_matrix):
            new_C_sum = sum(row)
            if new_C_sum < (2.2250738585072014e-308):
                new_C_sum = (2.2250738585072014e-308)
            for i,t_key in enumerate(self.trajs.pathdict):
                new_weighted_prob = row[i]/new_C_sum
                if new_weighted_prob < (2.2250738585072014e-308):
                    new_weighted_prob = 0.0                               
                traj.weighted_probability[m] = new_weighted_prob
                
                
                
        return max_diviation
            

    
    def normalize(self,bayesian_matrix):
        bayesian_matrix = np.array(bayesian_matrix)
        bayesian_matrix /= (sum(bayesian_matrix))
        return bayesian_matrix
            
    
    def compute_probability(self, max_div):
        bayesian_matrix = []
        for m in range(self.M): # For each Model
            trajs_prob = np.empty([0,1])
            for t_key in self.trajs.pathdict: #For each trajectory
                j = 0
                probability = 1.0
                #*****************************************************************#
                # The E-step                                                      
                #-----------------------------------------------------------------#
                #Calculate the Expectations
                for t in range(self.T):#For each Time spep in 
                    
                    probability *= self.gaussian_1_dimension(self.trajs.pathdict[t_key].points[t],m,j)
                #*****************************************************************#
                    if (t%(self.beta)) == (self.beta - 1):                        
                        j+=1
                trajs_prob= np.append(trajs_prob,(probability))# P(t_key| Cm)            
            bayesian_matrix.append(trajs_prob)
        
        #*****************************************************************#
        # The M-step
        # Calculate the Maximization
        #-----------------------------------------------------------------#
        normalised_bayesian_matrix = self.normalize(bayesian_matrix)
        max_div = self.set_weights(np.array(normalised_bayesian_matrix))
        #max_div = self.set_weights(np.array(bayesian_matrix))
        for c_i in range(self.M):
            self.models[c_i] = self.get_mean_cluster_i(c_i)
        #*****************************************************************#
        return max_div
    
    def plotta(self):
        
        
        for c_i in range(self.M):
            clusters = np.empty([0,2])
            colors = ["red","green","blue","black","orange","White","darkgreen","darkblue","gold","brown","yellow","pink","purple","grey"]
            for i,t_key in enumerate(self.trajs.pathdict):
                if np.isnan(self.trajs.pathdict[t_key].probability[c_i]):
                    clusters = np.append(clusters,[[i,0.0]],axis=0)
                else:
                    clusters = np.append(clusters,[[i,self.trajs.pathdict[t_key].probability[c_i]]],axis=0)
            plt.scatter(clusters[:,0], clusters[:,1], c=colors[c_i])
        plt.title("Models probability")
        plt.xlabel("Trajectorys")
        plt.ylabel("Probability")
        plt.show()
        self.plotclusters()      


        
    def GMM(self, max_div = 0.0):
        swap = True
        laps = 0
        latest_value = 0.0
        i = 0
        while swap == True:
            movement = self.compute_probability(max_div)
            if movement < (0.09):
                swap = False
            laps+= 1
            print(movement)
            if laps >= 200:
                swap = False
            print("----------") 
            if laps% 20 == 0:
                self.plotta()
        print("-DONE-")    
        self.plotta()
        

    def plotclusters(self, first_run = False):# xmin, xmax, ymin, ymaxset_centroids
        colors = ["red","green","blue","black","orange","White","darkgreen","darkblue","gold","brown","yellow","pink","purple","grey"]
        for t_key in self.trajs.pathdict:
            if first_run:
                plt.plot(self.trajs.pathdict[t_key].points[:,0],self.trajs.pathdict[t_key].points[:,1],color= "lightblue")
            else:
                traj_color = self.trajs.pathdict[t_key].get_highest_model_index()
                print(colors[traj_color])
                plt.plot(self.trajs.pathdict[t_key].points[:,0],self.trajs.pathdict[t_key].points[:,1],color= colors[traj_color]) 
            #print(self.trajs.pathdict[t_key].probability)
        
        for c_i in range(self.M):
            plt.scatter(self.models[c_i][:,0], self.models[c_i][:,1], alpha=0.5, marker=r'$\clubsuit$',color=colors[c_i])
            plt.plot(self.models[c_i][:,0], self.models[c_i][:,1], c = colors[c_i])
        plt.show()

    
    
def readcsvfile(numoftrajstoread=0):
    global trajs;
    
    with open('testfile.csv') as data:
        data = csv.reader(data, delimiter=',')
        trajnr = 0
        maxim = 0
        num = 0
        isnewtrajectory = True
        id = 0
        for row in data:
            if(row[0] == '###'):
                if maxim < num:
                    maxim = num
                num = 0
                if(newtrajectory.get_length() > 0):
                    trajs.add_trajectory(id,newtrajectory)
                    trajnr = trajnr + 1;
                if numoftrajstoread is not 0 and trajnr >= numoftrajstoread:
                    break;
                newtrajectory = trajectory()
                isnewtrajectory = True
                
            else:
                num += 1
                if not isnewtrajectory:
                    newtrajectory.add_point(float(row[0]), float(row[2]),float(row[3])) 
                    
                else:
                    
                    id = row[1]
                    newtrajectory = trajectory()
                    newtrajectory.add_point(float(row[0]), float(row[2]),float(row[3])) 
                    isnewtrajectory = False  
        print(num)

def generate_data(M,T, init_points, noise, Number_of_trajs):
    global trajs
    global models_created
    #Generate trajectories
    i = 0
    j = 0
    for m in range(M):
        #ran = int( Number_of_trajs/(M+j))
        ran = Number_of_trajs
        tr = []
        for j in range(ran):
            new_traj = trajectory()
            dist = 0.0
            for t in range(T):
                point = []
                for x_i in range(2):
                    a = dist+rdm.randint((init_points[m]-noise), (dist+init_points[m]-noise))
                    point.append(a)
                dist+= noise
                new_traj.add_point(i,point[0],point[1])
            tr.append(new_traj)
            trajs.add_trajectory(i,new_traj)
            i += 1
        models_created.append(tr[-1])

        
    

trajs = trajectories()

M = 6 #number of clusters
T = 5
K = T 
init_points = [0, 455, 1050]
noise = 10
Number_of_trajs = 100
models_created = []
#generate_data(M,T, init_points, noise, Number_of_trajs)
readcsvfile(40)
trajs.filter_noise()
trajs.interpol_points(T)
#trajs.interpol_test(10)
#trajs.plot()

#test = trajs.kmeansclustering(25)

treashold = 20000
#covariance = np.array([[var_x,math.sqrt(var_x)*math.sqrt(var_y)],[math.sqrt(var_x)*math.sqrt(var_y),var_y]])
covariance = 1700
GMM = GaussianMixtureModel(trajs,K,covariance,M,treashold, init_points)
GMM.GMM()
