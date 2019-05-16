"""Docstring
    blablabla
"""
import itertools
import csv
import math
import matplotlib as mpl
import random as rdm
import copy

import numpy as np
#import statistics as ss
import sklearn.mixture as mixture
import matplotlib.pyplot as plt
from tqdm import tqdm
#import string
#from scipy.spatial import distance as dst
from sklearn import gaussian_process as gp
#from sklearn.cluster import KMeans
#from sklearn.gaussian_process import
from matplotlib.colors import LogNorm

class trajectory:
    """class docstring"""
    def __init__(self):
        """init trajectory"""
        self.points = np.empty([0, 3])#np.array([], (float), ndmin=2)

    def add_point(self, point):
        """add point x,y timestamp"""
        self.probability = []
        self.number_of_clusters = 0
        self.points = np.append(self.points, [point], axis=0)

    def get_probability(self, index):
        return self.probability[index]/sum(self.probability)
    def remove_point(self, time, x, y):
        """remove point as x,y, timestamp if this results to a empty trajectory return True"""
        self.points = np.delete(self.points, np.where(self.points == [[x, y, time]])[0], axis=0)
        return bool(len(self.points) < 3)

    def calc_distance(self, traj):
        """calculates an abstract distance beetween self and a trajectory"""
        if not traj:
            return None
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1, p2 in zip(self.points, traj.points)])
    #Frechet distance #avg distance

    def get_length(self):
        """returns the function length with the sum of all lengths beetween the points"""
        return sum(self.get_lengths())
    def get_lengths(self):
        """returns a list of distances beetween points"""
        return [np.linalg.norm(p1-p2) for p1, p2 in zip(self.points, self.points[1:])]
    def get_lengths_from_first_point(self):
        """get all function lengths from first point to each other point"""
        lengths = self.get_lengths()
        for idx in enumerate(lengths[:-1]):
            lengths[idx[0]+1] = lengths[idx[0]] + lengths[idx[0]+1]
        return lengths

    def normailise_timestamps(self):
        """shift all timestamps so they all begin at 0"""
        normaliser = self.points[0, 2]
        for idx in enumerate(self.points):
            self.points[idx[0], 2] -= normaliser

class trajectories:
    """contains many trajectories, the logic for calculating gaussian mixture model
    and gaussian proccesses
    """
    def __init__(self):
        self.pathdict = {}

    def add_trajectory(self, _id, _trajectory):
        """adds a trajectory _id for the key and the second parameter is the trajectory object
        to pathdict unordered dictionary
        """
        self.pathdict[_id] = _trajectory
    def normalize_timestamps(self):
        """shifts all trajectories's timestamps so they all start at 0"""
        for _, traj in self.pathdict.items():
            traj.normailise_timestamps()

    def filter_noise(self, threshold=500, plotit=False):
        """Rough filter that filters all points that is closer than threshold"""
        minlength = threshold - 1 #just to start loop
        isdead = False
        key = None
        while minlength < threshold:
            for key in tqdm(self.pathdict, total=len(self.pathdict.keys())):
                backup = copy.deepcopy(self.pathdict[key])
                for point, nextpoint in zip(self.pathdict[key].points,
                                            self.pathdict[key].points[1:]):
                    if np.linalg.norm(nextpoint[:2] - point[:2]) < threshold:
                        if self.pathdict[key].points.size > 0 and \
                            set(nextpoint) != set(self.pathdict[key].points[-1]):
                            isdead = self.pathdict[key].remove_point(
                                nextpoint[2], nextpoint[0], nextpoint[1])
                        else:
                            isdead = self.pathdict[key].remove_point(point[2], point[0], point[1])
                        #if not isdead:
                        #    self.plotcompare([backup, self.pathdict[key]], [0x00FFFF, 0xFF00FF])
                if isdead:
                    tqdm.external_write_mode()
                    print("\n removed: " + key)
                    break
                elif plotit is True:
                    self.plotcompare([backup, self.pathdict[key]], [0x00FFFF, 0xFF00FF])
            if isdead and key:
                self.pathdict.pop(key)
                isdead = False
            else:
                minlength = min([np.linalg.norm(p1-p2) for p1, p2 in
                                 zip(self.pathdict[key].points, self.pathdict[key].points[1:])])

    def elbow_method_for_k(self, initk, maxk, threshold=200000):
        """Decide k for the clustering"""
        percentage_variance = []
        #datavariance = sum([self.pathdict[key].calc_distance(self.pathdict[nextkey]) for
        #                    key, nextkey in zip(list(self.pathdict.keys()), list(self.pathdict.keys())[1:])])/len(self.pathdict)
        #datavariance = self.calc_mean_traj(self.pathdict.keys())
        for i in tqdm(range(initk, maxk-3)):
            clusters, threshold = self.kmeansclustering(i, threshold)
            clustercentroids = [self.calc_mean_traj(clusters[cluster]) for cluster in clusters]
            clustervariance = sum([traj.calc_distance(nexttraj) for traj, nexttraj in zip(clustercentroids, clustercentroids[1:]) if traj and nexttraj])/len(clustercentroids)
            #percentage_variance.append((datavariance/clustervariance))
            newlist = []
            #for _, val in clusters.items():
                #maxval = [self.pathdict[key].calc_nex]
            anothervariance = max([max([self.pathdict[key].calc_distance(self.pathdict[nextkey]) for key,nextkey in zip(keylist, keylist[1:])], default=0) for _, keylist in clusters.items()], default=0)
            percentage_variance.append((clustervariance/i))
        #plt.figure(figsize=(2,1))
        plt.xticks(range(len(percentage_variance)))
        plt.plot(range(initk, initk+len(percentage_variance)), percentage_variance)
        print(percentage_variance)

    def generate_centroids(self, k, threshold=200000, plotit=False):
        """Generate k centroids with closest distance threshold"""
        rdmkeys = rdm.sample(list(self.pathdict.keys()), k)

        #fix for centroids that is to close
        counter = 0
        while True:
            istoclose = False
            for key1, key2 in tqdm(zip(rdmkeys, rdmkeys[1:]), total=len(rdmkeys)):
                if self.pathdict[key1].calc_distance(self.pathdict[key2]) < threshold:
                    if counter > 100:
                        counter = 0
                        threshold -= 100
                        threshold = max(threshold, 0)

                    counter += 1
                    istoclose = True
                    rdmkeys.remove(key1)
                    rdmkeys.append(rdm.choice([newkey for newkey in list
                                               (self.pathdict.keys()) if newkey not in rdmkeys]))
                    break
            if not istoclose:
                break

        centroids = {}
        for idx, key in enumerate(tqdm(rdmkeys)):
            centroids["cluster_"+str(idx+1)] = copy.deepcopy(self.pathdict[key])
        if plotit:
            self.plotwx(centroids)
        return centroids

    def kmeansclustering(self, k, threshold=200000, acceptible_distance=5, plotit=False):
        """generates k clusters with centroids that is further away than threshold"""
        centroids = self.generate_centroids(k, threshold, plotit)

        ischanging = True
        while ischanging:
            centroidsold = copy.deepcopy(centroids)
            #connect centroids to closest trajectories to form clusters
            clusters = {}
            for key in centroids:
                clusters[key] = []

            for pathkey in tqdm(self.pathdict):
                minkey = next(iter(centroids.items()))[0]
                mindistance = centroids[minkey].calc_distance(self.pathdict[pathkey])

                for key in centroids:
                    newdistance = centroids[key].calc_distance(self.pathdict[pathkey])
                    if  newdistance < mindistance:
                        minkey = key
                        mindistance = newdistance
                clusters[minkey].append(pathkey)

            #reassign centroids with means of respective cluster
            for key in tqdm(centroids):
                if clusters[key]:
                    centroids[key] = self.calc_mean_traj(clusters[key])
                else:
                    tqdm.external_write_mode()
                    print("error")
            if plotit:
                self.plotwx(centroids)
            totdist = 0
            for key in centroidsold:
                if centroidsold[key]:
                    totdist += centroids[key].calc_distance(centroidsold[key])
            if totdist < acceptible_distance:
                ischanging = False
                if plotit:
                    self.plotclusters(clusters)
                return clusters, threshold


    def calc_mean_traj(self, keys):
        """returns a mean trajectory given keys from pathdict"""
        if not keys:
            return None

        samplecount = len(keys)
        #if samplecount == 0:
        #    raise ValueError('empty trajectory list')
        if samplecount == 1:
            return self.pathdict[keys[0]]
        newmeantraj = trajectory()

        pointsum = sum([self.pathdict[value].points for value in keys])
        #if isinstance(pointsum, int):
        #    raise ValueError("empty key list")

        for point in pointsum:
            newmeantraj.add_point(point/samplecount)
        return newmeantraj

    def get_next_point(self, last_point, next_point, movement):
        """returns a point between two points with the distance from first point as movement"""
        v = next_point-last_point
        return last_point+(v/(np.linalg.norm(v)))*movement

    def interpol_points(self, number_of_points=10, plotresult=False):
        """interpolates all trajectories so all get n number_of_points, plot opion avaliable"""
        for key in tqdm(self.pathdict):
            #Calculate total length of the function
            function_length = self.pathdict[key].get_length()

            #Calculate how much to move at each time step
            segment_length = function_length/(number_of_points-1)

            #Calculate distance from first point to each other point
            li = self.pathdict[key].get_lengths_from_first_point()

            temp_traj = trajectory()
            temp_traj.add_point(self.pathdict[key].points[0])
            j = 0
            passedlength = 0
            for i in range(1, number_of_points-1):
                while i*segment_length > li[j]:
                    if  j < len(li):
                        passedlength = li[j]
                        j += 1
                    else: raise ValueError('Interpolation error')
                last_point = self.pathdict[key].points[j]
                next_point = self.pathdict[key].points[j+1]
                rest_segment = i*segment_length-passedlength

                curr_interpol_point = self.get_next_point(last_point, next_point, rest_segment)
                temp_traj.add_point(curr_interpol_point)
            temp_traj.add_point(self.pathdict[key].points[-1])

            if plotresult:
                self.plotcompare([self.pathdict[key], temp_traj], [0xFF0000, 0x00FF00])
            self.pathdict[key] = temp_traj

    def generate_guassian_processes(self, clusters):
        """generates two gaussian proccesses for x and y for each cluster"""
        #params = gp.kernels.Hyperparameter('theta',float,3,3) #testing stage
        self.normalize_timestamps()
        for cluster, value in clusters.items():
            if len(cluster) > 1:


                pointsxt = np.empty([0,2])
                pointsyt = np.empty([0,2])

                for key in value:
                    pointsxt = np.append(pointsxt,self.pathdict[key].points[:,0:3:2], axis=0)
                    pointsyt = np.append(pointsyt,self.pathdict[key].points[:,1:3], axis=0)




                xs = pointsxt[:,0]
                #print(xs)

                #for i in range(0, 23):
                #    xs[i] += rdm.random()*5000

                #for i, _ in enumerate(xs):
                #    xs[i] += rdm.random()*20000
                #print(xs)
                #T, X coords

                #ys = pointsyt[:,0]

                tx = pointsxt[:,1]
                #ty = pointsyt[:,1]

                #tx = np.tile(np.linspace(0, len(self.pathdict[value[0]].points)-1,len(self.pathdict[value[0]].points)),len(value))

                lengthscale = 1.0
                noise = 1.0
                n = 20
                constant = 2

                theta_x,theta_y = (316.0,1.0)

                kx = theta_x*gp.kernels.RBF(lengthscale,(10.0,1e5)) + gp.kernels.WhiteKernel(noise, (1e-5,1.0)) + constant
                print("x kernel before:", kx)
                process_x = gp.GaussianProcessRegressor(kernel=kx, n_restarts_optimizer=n, normalize_y=False, alpha=0)

                #ky = theta_y*gp.kernels.RBF(lengthscale) + gp.kernels.WhiteKernel(noise)
                #print("y kernel before:", ky)
                #process_y = gp.GaussianProcessRegressor(kernel=ky, n_restarts_optimizer=n, normalize_y=True, alpha=0)

                process_x.fit(tx.reshape(-1,1), xs.reshape(-1,1))

                print("")
                print("x kernel after:", process_x.kernel_)

                #process_y.fit(ty.reshape(-1,1), ys.reshape(-1,1))

                #print("y kernel after:", process_y.kernel_)
                #print("")

                mt = self.calc_mean_traj(value).points[:,2].reshape(-1,1)

                seqx = np.sort(tx).reshape(-1,1)[len(value)-1:]
                seqx = np.atleast_2d(np.linspace(min(tx), max(tx), len(tx))).reshape(-1,1)
                #seqx = np.atleast_2d(self.calc_mean_traj(value).points[:,2]).reshape(-1,1)
                predict_x,std_x = process_x.predict(seqx, return_std=True)
                #predict_y,std_y = process_y.predict(ty.reshape(-1,1), return_std=True)
                #print(std_x)
                #plt.imshow(std_x)


                for x,y in zip(np.split(tx,len(value)), np.split(xs, len(value))):
                    plt.plot(x, y, c='black')

                for _ in range(10):
                    test = process_x.sample_y(seqx)
                    plt.plot(seqx,test.squeeze())
                plt.scatter(seqx, predict_x, c='m')

                #print(predict_x)
                #print("")
                #print(tx)
                #print("")
                #print(std_x)

                plt.figure()
                plt.title(cluster)

                for idx,(x,y) in enumerate(zip(np.split(tx,len(value)), np.split(xs, len(value)))):
                    plt.plot(x, y, label='x\'s traj: '+str(idx+1))

                #plt.plot(tx, xs, 'm', label='$x-values$')
                #plt.plot(ty, ys, 'c', label='$y-values$')

                plt.xlabel('$time$')
                plt.ylabel('$value$')
                plt.legend(loc='upper left')

                plt.figure()
                plt.title(cluster)

                #plt.scatter(mt, predict_x, c='m')
                #plt.scatter(tx, predict_x, c='m')


                plt.fill(np.concatenate([seqx, seqx[::-1]]),
                         np.concatenate([predict_x - np.square(std_x),
                                        (predict_x + np.square(std_x))[::-1]]),
                alpha=.5, fc='b', ec='None', label='95% confidence interval')

                #plt.fill(np.concatenate([mt, mt[::-1]]),
                #         np.concatenate([predict_x - std_x,
                #                        (predict_x + std_x)[::-1]]),
                #alpha=.5, fc='b', ec='None', label='95% confidence interval')

                for x,y in zip(np.split(tx,len(value)), np.split(xs, len(value))):
                    plt.plot(x, y, c='black')

                plt.scatter(seqx, predict_x, c='m')
                #plt.plot(tx, xs, 'black')
                #plt.scatter(std_x, std_y)

                #plt.scatter(ty, predict_y, c='c')

                #plt.fill(np.concatenate([ty, ty[::-1]]),
                #         np.concatenate([predict_y - 1.96 * std_y,
                #                        (predict_y + 1.96 * std_y)[::-1]]),
                #alpha=.5, fc='b', ec='None', label='95% confidence interval')

                #plt.plot(ty, ys, 'black')


                plt.figure()
                plt.title(cluster)
                [plt.plot(self.pathdict[key].points[:,0], self.pathdict[key].points[:,1], 'g') for key in clusters[cluster]]
                plt.show()
                #with open(cluster,mode='wt') as data:
                #    data = csv.writer(data, delimiter=',')
                #    for x, t in zip(xs, tx):
                #        data.writerow([str(x), str(t)])

    def plotcompare(self, listoftrajs, colors):

        #= rdm.sample(range(colorrange), len(listoftrajs))
        count = 0
        if colors.count() == 0:
            for _ in range(len(listoftrajs)):
                colors.append('%06X' % rdm.randint(0, 0xEEEEEE))
        else:
            for idx, value in enumerate(colors):
                colors[idx] = '%06X' % value

        for traj in listoftrajs:
            print(".............................")
            print("Goal:")
            print("x: ", traj.points[-1,0])
           #print("x: ",traj.points[-1,0])
            print("y: ", traj.points[-1,1])
           # print("y: ",temp_traj.points[-1,1])
            print(".............................")
            paint = (colors[count])
            count += 1

            plt.plot(traj.points[:, 0], traj.points[:, 1], '#'+paint)
            plt.scatter(traj.points[:, 0], traj.points[:, 1], color='#'+paint)

           # plt.plot(temp_traj.points[:,0],temp_traj.points[:,1])
           # plt.scatter(temp_traj.points[:,0],temp_traj.points[:,1],color='orange')
        plt.show()
        colors.clear()

    def plotclusters(self, clusters):
        """plots all clusters in the parameter"""
        plt.axis([-50000, 50000.0, -50000.0, 50000.0])# xmin, xmax, ymin, ymax

        numofclusters = len(clusters)
        colorrange = 10000
        colors = rdm.sample(range(colorrange), numofclusters)
        count = 0
        colors = []

        for _ in range(1000):
            colors.append('%06X' % rdm.randint(0, 0xFFFFFF))

        for element in clusters:
            color = (colors[count])
           #color = cm.hot(float(colors[count])/colorrange)
           #color = cm.autumn(float(colors[count])/colorrange)
            for key in clusters[element]:
                plt.plot(self.pathdict[key].points[:, 0],
                         self.pathdict[key].points[:, 1], '#'+color)
            count += 1
        plt.show()

    def plotwx(self, x):
        """plots with x"""
        for key in x:
            plt.plot(x[key].points[:, 0], x[key].points[:, 1], "*")

        self.plot()

    def plot(self):
        """plot of all trajectories"""
        plt.axis([-50000, 50000.0, -50000.0, 50000.0]) # xmin, xmax, ymin, ymax
        #plt.gca().set_autoscale_on(False)

        for key in self.pathdict:
            plt.plot(self.pathdict[key].points[:, 0], self.pathdict[key].points[:, 1])

        plt.show()


class Model():
    def __init__(self,list_of_points):
        self.traj = copy.deepcopy(list_of_points)
class GaussianMixtureModel():


    def __init__(self, trajs,K,covariance,M,treashold):
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


    def set_cluster_colours(self):
        for i in range(self.M):
            self.colors.append('%06X' % rdm.randint(0, 0xFFFFFF))


    def calc_distance(self,traj1 , traj2): #calculates an abstract distance beetween two trajectories
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1,p2 in zip(traj1.points, traj2.points)])

    def set_centroids(self,treshold = 200000):
        #Set all
        for i in range(self.M):
            all_trajs = [trajectory]
            self.importance.append(1.0)
            for t_key in self.trajs.pathdict:
                self.trajs.pathdict[t_key].probability.append(0.0)
                all_trajs.append(self.trajs.pathdict[t_key])



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
                        treashold -= 100
                        treashold = max(treashold, 0)
                    counter += 1
                    istoclose = True
                    rdmkeys.remove(key1)
                    treshold -= 100
                    if treshold < 0:
                        treshold = 0
                    rdmkeys.append(rdm.choice([newkey for newkey in list(self.trajs.pathdict.keys()) if newkey not in rdmkeys]))
        #Set initial probability for the centroids
        for i,key in enumerate(rdmkeys):
            self.trajs.pathdict[key].probability[i] = (1.0)
            model_i = Model(self.trajs.pathdict[key].points)
            self.models.append(model_i.traj)
        self.plotclusters()


    def get_mean_cluster_i(self,m): # Get the "beta" cluster
        mean_trajectory = np.empty([0,3])
        for i in range(0,self.T):
            sum_from_points = np.array([0.0,0.0,0.0])

            for t_key in self.trajs.pathdict:
                point = self.trajs.pathdict[t_key].points[i]
                point_prob = self.trajs.pathdict[t_key].probability[m]
                sum_from_points += point_prob*point

            if (i % self.beta) == (self.beta - 1):
                sum_from_points = sum_from_points/self.beta
                mean_trajectory = np.append(mean_trajectory,[sum_from_points], axis=0)

        return mean_trajectory[:,:2]

    #Get mu for the right point to compare with
    def get_mu_point(self,m,j):
        arr = np.array([self.get_mean_cluster_i(m)[j]])
        return [arr[0][0],arr[0][1]]


    def gaussian_1_dimension(self,point,m,j):
        #a = (np.linalg.norm(point[:-1]-mu)**2)
        #b = (-1/(2*(self.covariance)**2))

        c = math.exp((-1/(2*(self.covariance)**2))*(np.linalg.norm(point[:-1]-self.models[m][j,:2])**2))
        #c = math.exp((-1/(2*(self.covariance)**2))*(np.linalg.norm(point[:-1]-mu)**2))
        if np.isnan(c) or c <= 0.001:
            res = 0.001
        else:
            res = c
        return res

    def set_new_weights(self,prob_matrix, max_diviation):
        num_of_changes = 0
        sum_from_trajs = sum(prob_matrix)
        trajs = []
        for i,t_key in enumerate(self.trajs.pathdict):
            traj = self.trajs.pathdict[t_key]
            for m in range(self.M):
                sum_t = sum_from_trajs[i]
                if sum_t ==0.0  or np.isnan(sum_t):
                    sum_t = 0.0001

                prob_t = prob_matrix[m][i]
                new_prob= prob_t/sum_t
                if np.isnan(new_prob):
                    new_prob = 0.0
                new_diviation = abs(new_prob - prob_matrix[m][i])
                if max_diviation < new_diviation:
                    max_diviation = new_diviation
                    num_of_changes += 1
                traj.probability[m] = new_prob
            trajs.append(traj.probability)


        return max_diviation,num_of_changes

    #def set_importance(self,importance):
        #sum_ti_importance = sum(importance)
        #for m in range(self.M):
            #importance_i = importance[m]
            #self.importance[m] = (importance_i/sum_importance)

    def set_importance(self,bayesian_matrix):

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
            for m in range(self.M):  #P(Cm\traj_i)
                numerator = p_Cms[m]*bayesian_matrix[m][t_i]         #P(Cm)*P(Cm\ti)
                denominator = sum((bayesian_matrix[:,t_i]*p_Cms))    #P(Cm)*P(Cm\ti)+P(Cm+1)*P(Cm+1\ti)+...+P(CM)*P(CM\ti)
                self.trajs.pathdict[t_key].probability[m] = numerator/denominator       #P(Cm\ti)









    def set_new_probabilities(self, t_probs, c_i,max_div = 0.05):
        prob_sumation = sum(t_probs)
        num_of_changes = 0
        for i,t_key in enumerate(self.trajs.pathdict):
            new_prob = t_probs[i]/prob_sumation
            aa = abs(new_prob - self.trajs.pathdict[t_key].probability[c_i])
            if max_div < abs(new_prob - self.trajs.pathdict[t_key].probability[c_i]):
                max_div = new_prob
                num_of_changes += 1
            self.trajs.pathdict[t_key].probability[c_i] = new_prob
        return max_div,num_of_changes


    def compute_probability(self, max_div):
        num_of_changes = 0
        number_of_trajectories = len(self.trajs.pathdict)
        c_trajs = []
        bayesian_matrix = []
        for m in range(self.M): # For each Cluster
            trajs_prob = np.empty([0,1])

            #mu = np.array(self.get_mean_cluster_i(m))
            for t_key in self.trajs.pathdict: #For each trajectory
                j = 0
                probability = 1.0
                #*****************************************************************#
                # The E-step
                #-----------------------------------------------------------------#
                #Calculate the probability
                for t in range(self.T):#For each Time spep in

                    probability *= self.gaussian_1_dimension(self.trajs.pathdict[t_key].points[t],m,j)
                    if (t%(self.beta)) == (self.beta - 1):
                        j+=1

                #probability *= self.importance[m]

                #*****************************************************************#
                P_Cm_t_key = self.trajs.pathdict[t_key].probability[m]
                trajs_prob= np.append(trajs_prob,(math.exp(probability)*P_Cm_t_key))# P(C_m\t_key)

            #max_div,temp_num_of_changes = self.set_new_probabilities(trajs_prob,m,max_div)

            bayesian_matrix.append(trajs_prob)
            c_trajs.append(trajs_prob)
        self.set_importance(np.array(bayesian_matrix))
        #*****************************************************************#
        # The M-step
        #-----------------------------------------------------------------#
        max_div,num_of_changes = self.set_new_weights(np.array(c_trajs),max_div)
        for c_i in range(self.M):
            self.models[c_i] = self.get_mean_cluster_i(c_i)
        #*****************************************************************#
        return max_div,num_of_changes



    def GMM(self, max_div = 0.3):
        swap = True
        laps = 0
        while swap == True:
            swap = False
            #E-step
            movement,num_of_changes = self.compute_probability(max_div)
            if movement >0.03:
                swap = True


            laps+= 1
            #if laps % 1 == 0:
                #clusters = np.empty([0,2])
                #for c_i in range(self.M):
                    #clusters = np.empty([0,2])
                    #for i,t_key in enumerate(self.trajs.pathdict):
                        ##plt.scatter(i+1,self.trajs.pathdict[t_key].probability[c_i],c ='#'+self.colors[c_i])
                        ##if np.isnan(self.trajs.pathdict[t_key].probability[c_i]):
                            ##clusters = np.append(clusters,[[i+1,0.0]],axis=0)
                        ##else:
                            ##clusters = np.append(clusters,[[i+1,self.trajs.pathdict[t_key].probability[c_i]]],axis=0)
                        #plt.plot(clusters[:,0], clusters[:,1], '#'+self.colors[c_i])
                        ##plt.plot(clusters[:,0], clusters[:,1])
                #plt.title("Cluster probability")
                #print("Max div" + str(movement))
                ##plt.show()
                #print(num_of_changes)
                #self.plotclusters()
            print("PLOT")
            print("-Movement-")
            print(movement)
            print("----------")
            for model in self.models:
                plt.plot(model[:,0], model[:,1])
            plt.show()
            print(laps)




    def plotclusters(self):# xmin, xmax, ymin, ymaxset_centroids
        colorrange = 10000
        colors = rdm.sample(range(colorrange), self.M)
        count = 0
        colors = []

        for i in range(1000):
            colors.append('%06X' % rdm.randint(0, 0xFFFFFF))
        for c_i in range(self.M):
            color = self.colors[c_i]
            #cluster = self.get_mean_cluster_i(c_i)
            plt.plot(self.models[c_i][:,0], self.models[c_i][:,1], '#'+color)
            #plt.plot(cluster[:,0], cluster[:,1])
        plt.show()




trajs = trajectories()

def countrows(file):
    """counts all rows of a csv file"""
    with open(file) as data:
        data = csv.reader(data, delimiter=',')
        return sum([1 for row in data if row[0] != '###'])

def readcsvfile(filename='testfile.csv', numoftrajstoread=0):
    """reads a csv file and imports it into a trajectories object"""
    global trajs

    with open(filename) as data:
        data = csv.reader(data, delimiter=',')
        trajnr = 0
        rows = countrows(filename)
        if numoftrajstoread > 0:
            rows = numoftrajstoread
        #data.seek(0)

        isnewtrajectory = True
        newtrajectory = None

        key = 0
        for row in tqdm(data, total=rows):
            if row[0] == '###':
                if newtrajectory is not None and newtrajectory.get_length() > 5000:
                    trajs.add_trajectory(key, newtrajectory)
                    trajnr = trajnr + 1
                if numoftrajstoread != 0 and trajnr >= numoftrajstoread:
                    break
                newtrajectory = trajectory()
                isnewtrajectory = True

            else:
                if not isnewtrajectory:
                    newtrajectory.add_point([int(row[2]), int(row[3]),float(row[0])])

                else:
                    key = row[1]
                    newtrajectory = trajectory()
                    newtrajectory.add_point([int(row[2]), int(row[3]),float(row[0])])
                    isnewtrajectory = False

trajstoread = 10
readcsvfile('testfile.csv', trajstoread)
trajs.filter_noise()
n = sum([len(trajs.pathdict[key].points) for key in trajs.pathdict])/len(trajs.pathdict)
print(n)
#n = 37 #for 100 trajs
#n = 36 #for 50 trajs
n = 23  #for 10 trajs
trajs.interpol_points(int(n))

pointsxt = np.empty([0,2])
pointsyt = np.empty([0,2])

for traj in trajs.pathdict:
    pointsxt = np.append(pointsxt,trajs.pathdict[traj].points[:,0:3:2], axis=0)
    pointsyt = np.append(pointsyt,trajs.pathdict[traj].points[:,1:3], axis=0)
trajs.plot()
xs = pointsxt[:,0]
ys = pointsyt[:,0]

tx = pointsxt[:,1]
ty = pointsyt[:,1]

plt.figure()
for idx,(x,y) in enumerate(zip(np.split(tx,len(trajs.pathdict)), np.split(xs, len(trajs.pathdict)))):
    plt.plot(x, y)

#test = trajs.kmeansclustering(25)
K = n #"Hyperparameter"
M = 2 #number of clusters
treashold = 20000
#covariance = np.array([[var_x,math.sqrt(var_x)*math.sqrt(var_y)],[math.sqrt(var_x)*math.sqrt(var_y),var_y]])
covariance =2500
GMM = GaussianMixtureModel(trajs,K,covariance,M,treashold)
GMM.GMM()


plt.figure()
for idx,(x,y) in enumerate(zip(np.split(tx,len(trajs.pathdict)), np.split(xs, len(trajs.pathdict)))):
    plt.plot(x, y)
plt.show()

trajs.plot()

#K=11 #for 100 trajs
K=2 #for 10 trajs
#trajs.elbow_method_for_k(2, trajstoread-20)
#26 for 50

#CLUSTERS = trajs.kmeansclustering(K, 600000, 5, False)[0]
#trajs.generate_guassian_processes(CLUSTERS)