"""Docstring
    blablabla
"""
import csv
import random as rdm
import copy

import numpy as np
#import statistics as ss
#import pandas as ps
import matplotlib.pyplot as plt
from tqdm import tqdm
#import string
#from scipy.spatial import distance as dst
from sklearn import gaussian_process as gp
#from sklearn.cluster import KMeans
#from sklearn.gaussian_process import
from typing import NamedTuple

class trajectory:
    """class docstring"""
    def __init__(self):
        """init trajectory"""
        self.points = np.empty([0, 3])#np.array([], (float), ndmin=2)

    def add_point(self, point):
        """add point x,y timestamp"""
        self.points = np.append(self.points, [point], axis=0)

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
        self.numofpoints = 0
        self.clusters = {}
        self.gaussian_processes = {}

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
                self.clusters = clusters
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
            if number_of_points < 2:
                raise ValueError('not enaugh points to interpolate')
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
            self.numofpoints = number_of_points

    def generate_guassian_processes(self, clusters, plotit = False, numofsamples=100, plotcov=False):
        """generates two gaussian proccesses for x and y for each cluster"""
        #params = gp.kernels.Hyperparameter('theta',float,3,3) #testing stage
        class struct(NamedTuple):
            process_x: gp.GaussianProcessRegressor
            process_y: gp.GaussianProcessRegressor
            
        self.normalize_timestamps()
        for cluster, value in clusters.items():
            if len(cluster) > 1:
                if plotit:
                    plt.figure()
                    plt.title(cluster)
                    for key in clusters[cluster]:
                        plt.plot(self.pathdict[key].points[:, 0], self.pathdict[key].points[:, 1], 'g')
                    plt.show()

                #pointsxt = np.empty([0, 2])
                #pointsyt = np.empty([0, 2])

                #for key in value:
                    #pointsxt = np.append(pointsxt, self.pathdict[key].points[:, 0:3:2], axis=0)
                    #pointsyt = np.append(pointsyt, self.pathdict[key].points[:, 1:3], axis=0)
                #    pointsxt = np.append(pointsxt, self.pathdict[key].points[:, 0], axis=0)
                #    pointsyt = np.append(pointsyt, self.pathdict[key].points[:, 1], axis=0)
                

                #xs = pointsxt[:, 0]
                xs = np.array([self.pathdict[key].points[:,0] for key in value]).flatten()
                ys = np.array([self.pathdict[key].points[:,1] for key in value]).flatten()
                #print(xs)

                #for i in range(0, 23):
                #    xs[i] += rdm.random()*5000

                #for i, _ in enumerate(xs):
                #    xs[i] += rdm.random()*20000
                #print(xs)
                #T, X coords

                #ys = pointsyt[:,0]

                #tx = pointsxt[:, 1]
                #ty = pointsyt[:,1]

                #tx = np.tile(np.linspace(0, len(self.pathdict[value[0]].points)-1,len(self.pathdict[value[0]].points)),len(value))
                if self.numofpoints <= 0:
                    raise ValueError("Not interpolated")
                #tx = np.tile(np.linspace(0, self.numofpoints-1, self.numofpoints), len(value)).reshape(-1,1)
                #ty = tx
                
                tx = np.array([self.pathdict[key].points[:,2] for key in value]).flatten().reshape(-1,1)
                ty = tx

                lengthscale = 1.0
                noise = 1.0
                n = 20
                (constant_x, constant_y) = (2.0, 2.0)

                theta_x,theta_y = (316.0,1.0)

                kx = theta_x*gp.kernels.RBF(lengthscale,(10.0,1e5)) + gp.kernels.WhiteKernel(noise, (1e-5,1.0)) + constant_x
                print("x kernel before:", kx)
                process_x = gp.GaussianProcessRegressor(kernel=kx, n_restarts_optimizer=n, normalize_y=False, alpha=0, copy_X_train=True)

                ky = theta_y*gp.kernels.RBF(lengthscale, (10.0, 1e5)) + gp.kernels.WhiteKernel(noise, (1e-5, 1.0)) + constant_y
                print("y kernel before:", ky)
                process_y = gp.GaussianProcessRegressor(kernel=ky, n_restarts_optimizer=n, normalize_y=False, alpha=0, copy_X_train=True)

                process_x.fit(tx, xs)

                print("")
                print("x kernel after:", process_x.kernel_)

                process_y.fit(ty, ys)

                print("y kernel after:", process_y.kernel_)
                print("")
               
                if plotit:
                    plt.figure()
                    plt.title("process x")
                    self.plotprocess(process_x, numofsamples, plotcov)
                    
                    plt.figure()
                    plt.title("process y")
                    self.plotprocess(process_y, numofsamples, plotcov)
                    #seqx = np.atleast_2d(np.linspace(min(tx), max(tx), len(tx))).reshape(-1,1)
                    #seqy = np.atleast_2d(np.linspace(min(ty), max(ty), len(ty))).reshape(-1,1)
                
                    #for x,y in zip(np.split(tx,len(value)), np.split(xs, len(value))):
                    #    plt.plot(x, y, c='black')

                
                    #samples = process_x.sample_y(seqx,numofsamples)
                    #for sample in samples.T:
                    #    plt.plot(seqx, sample)

                #mt = self.calc_mean_traj(value).points[:,2].reshape(-1,1)
                #seqx = self.calc_mean_traj(value).points[:, 0:3:2]
                
                #covariance = process_x.kernel.__call__(tx, xs)
                
                #predict_x,std_x = process_x.predict(seqx, return_std=True)
                #predict_y,std_y = process_y.predict(ty.reshape(-1,1), return_std=True)
                #print(std_x)
                #plt.imshow(std_x)

                #plt.scatter(seqx, predict_x, c='m')
                #print(predict_x)
                #print("")
                #print(tx)
                #print("")
                #print(std_x)

                #plt.figure()
                #plt.title(cluster)

                #for idx,(x, y) in enumerate(zip(np.split(tx, len(value)), np.split(xs, len(value)))):
                #    plt.plot(x, y, label='x\'s traj: '+str(idx+1))

                #plt.plot(tx, xs, 'm', label='$x-values$')
                #plt.plot(ty, ys, 'c', label='$y-values$')

                #plt.xlabel('$time$')
                #plt.ylabel('$value$')
                #plt.legend(loc='upper left')

                #plt.figure()
                #plt.title(cluster)

                #plt.scatter(mt, predict_x, c='m')
                #plt.scatter(tx, predict_x, c='m')
                #plt.fill_between(seqx.flatten(), predict_x.flatten()-np.square(std_x).flatten(), predict_x.flatten()+np.square(std_x).flatten(), color='b', alpha=0.2)
                #plt.errorbar(seqx, predict_x, np.square(std_x))
                #plt.fill(np.concatenate([seqx, seqx[::-1]]),
                #         np.concatenate([predict_x - np.square(std_x),
                #                        (predict_x + np.square(std_x))[::-1]]),
                #alpha=.5, fc='b', ec='None', label='95% confidence interval')

                #plt.fill(np.concatenate([mt, mt[::-1]]),
                #         np.concatenate([predict_x - std_x,
                #                        (predict_x + std_x)[::-1]]),
                #alpha=.5, fc='b', ec='None', label='95% confidence interval')

                #for x, y in zip(np.split(tx, len(value)), np.split(xs, len(value))):
                #    plt.plot(x, y, c='black')

                #plt.scatter(seqx, predict_x, c='m')
                #plt.plot(tx, xs, 'black')
                #plt.scatter(std_x, std_y)

                #plt.scatter(ty, predict_y, c='c')

                #plt.fill(np.concatenate([ty, ty[::-1]]),
                #         np.concatenate([predict_y - 1.96 * std_y,
                #                        (predict_y + 1.96 * std_y)[::-1]]),
                #alpha=.5, fc='b', ec='None', label='95% confidence interval')

                #plt.plot(ty, ys, 'black')


                #with open(cluster,mode='wt') as data:
                #    data = csv.writer(data, delimiter=',')
                #    for x, t in zip(xs, tx):
                #        data.writerow([str(x), str(t)])        
                self.gaussian_processes[cluster] = struct(process_x, process_y)
                
    def gaussian_process_prediction(self, observations = None):
        """predicts the future path of observations"""
        observations = np.array([
            [26636.0, 0.0],
            [25563.269545667717, 1.5219380855560303],
            [24427.892169907624, 2.952316999435425],
            [23262.136978459646, 4.499892473220825],
            [22137.60101487511, 6.001432180404663],
            [21071.31276888807, 7.722446441650391],
            [20104.233448941595, 9.700586080551147],
            [19148.348601746715, 11.554569959640503],
            [18157.061861841583, 13.13681936264038],
            [17078.007119093843, 14.117368459701538],
            [16283.555851030129, 15.54201364517212],
            [15422.862936149199, 17.151325225830078],
            [14528.944531411305, 18.715941905975342],
            [13676.601709931612, 20.377726078033447],
            [12936.961101377226, 21.87928009033203],
            [12047.812383706154, 23.147696256637573],
            [11201.079379086052, 24.791006088256836],
            [10281.543248305608, 26.36141848564148],
            [9651.478001735219, 27.848175525665283],
            [9084.964958646353, 29.18827986717224],
            [8658.649625048833, 30.658584117889404],
            [8198.830420704067, 31.98782181739807]])
        for cluster in self.gaussian_processes.values():
            processx = cluster.process_x#.fit(observations[:5, 1].reshape(1,-1), observations[:5, 0].reshape(1,-1))
            mean, std = processx.predict(observations[:5,1].reshape(1,-1), return_std=True)
            seq = np.atleast_2d(np.linspace(min(observations[:,1]), max(observations[:,1]), 5)).reshape(-1,1)
            plt.fill_between(seq.flatten(), mean.flatten()-np.square(std).flatten(), mean.flatten()+np.square(std).flatten(), color='b', alpha=0.2)
            
            plt.scatter(seq, mean, c='m')
        print("in progress")
    def plotprocess(self, process, numofsamples=100, plotcov=False):
        values = process.y_train_
        t = process.X_train_
        seq = np.atleast_2d(np.linspace(min(t), max(t), len(t))).reshape(-1,1)
        
        num_of_parallel_trajs = np.count_nonzero(t==0)
        for x,y in zip(np.split(t, num_of_parallel_trajs), np.split(values, num_of_parallel_trajs)):
            plt.plot(x, y, c='black')

    
        #samples = process.sample_y(seq, numofsamples)
        #for sample in samples.T:
        #    plt.plot(seq, sample)
            
        if plotcov:
            predict,std = process.predict(seq, return_std=True)
            plt.scatter(seq, predict, c='m')
            plt.fill_between(seq.flatten(), predict.flatten()-np.square(std).flatten(), 
                             predict.flatten()+np.square(std).flatten(), color='b', alpha=0.2)

    def plotproceses(self):
        for _, cluster in self.gaussian_processes.items():
            process_x = cluster.process_x
            process_y = cluster.process_y
            
            xs = process_x.alpha_
            ys = process_y.alpha_
            
            tx = process_x.X_train_
            ty = process_y.X_train_
        
            seqx = np.atleast_2d(np.linspace(min(tx), max(tx), len(tx))).reshape(-1,1)
            seqy = np.atleast_2d(np.linspace(min(ty), max(ty), len(ty))).reshape(-1,1)
            
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

trajs = trajectories()

def countrows(file, numoftrajs=0):
    """counts all rows of a csv file"""
    with open(file) as data:
        data = csv.reader(data, delimiter=',')
        
        if numoftrajs == 0:
            return sum([1 for row in data if row[0] != '###'])
        counter=0
        Sum = 0
        for row in data:
            if row[0] == '###':
                counter+=1
            else:
                if counter >= numoftrajs:
                    break;
                else:
                    Sum += 1


def readcsvfile(filename='testfile.csv', numoftrajstoread=0):
    """reads a csv file and imports it into a trajectories object"""
    global trajs

    with open(filename) as data:
        data = csv.reader(data, delimiter=',')
        trajnr = 0
        rows = countrows(filename, numoftrajstoread)
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
trajs.plot()

pointsxt = np.empty([0,2])
pointsyt = np.empty([0,2])

for key in trajs.pathdict:
    pointsxt = np.append(pointsxt,trajs.pathdict[key].points[:,0:3:2], axis=0)
    pointsyt = np.append(pointsyt,trajs.pathdict[key].points[:,1:3], axis=0)

xs = pointsxt[:,0]
ys = pointsyt[:,0]

tx = pointsxt[:,1]
ty = pointsyt[:,1]

plt.figure()
for idx,(x,y) in enumerate(zip(np.split(tx,len(trajs.pathdict)), np.split(xs, len(trajs.pathdict)))):
    plt.plot(x, y)

plt.figure()
for idx,(x,y) in enumerate(zip(np.split(tx,len(trajs.pathdict)), np.split(xs, len(trajs.pathdict)))):
    plt.plot(x, y)
plt.show()



#K=11 #for 100 trajs
K=2 #for 10 trajs
#trajs.elbow_method_for_k(2, trajstoread-20)
#26 for 50

CLUSTERS = trajs.kmeansclustering(K, 600000, 5, False)[0]
trajs.generate_guassian_processes(CLUSTERS, True, plotcov=True)
trajs.plotproceses()
#trajs.gaussian_process_prediction()

#trajs.pathdict[next(iter(trajs.pathdict))]. #test row for a n trajectory
#trajs.calc_mean_traj([])