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
#from sklearn.gaussian_process import

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
        return sum([np.linalg.norm(p1[:2]-p2[:2]) for p1, p2 in zip(self.points, traj.points)])

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

    def filter_noise(self, threshold=500, plotit=False):
        """Rough filter that filters all points that is closer than threshold"""
        minlength = threshold - 1
        isdead = False
        while minlength < threshold:
            for key in tqdm(self.pathdict, total=len(self.pathdict.keys())):
                backup = copy.deepcopy(self.pathdict[key])
                for point, nextpoint in zip(self.pathdict[key].points,
                                            self.pathdict[key].points[1:]):
                    if np.linalg.norm(nextpoint - point) < threshold:
                        if set(nextpoint) != set(self.pathdict[key].points[-1]):
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
            if isdead:
                self.pathdict.pop(key)
                isdead = False
            else:
                minlength = min([np.linalg.norm(p1-p2) for p1, p2 in
                                 zip(self.pathdict[key].points, self.pathdict[key].points[1:])])

    def kmeansclustering(self, k, treshold=200000, plotit=False):
        """generates k clusters with centroids that is further away than threshold"""
        #Generate k centroids
        rdmkeys = rdm.sample(list(self.pathdict.keys()), k)

        #fix for centroids that is to close
        istoclose = True
        while istoclose:
            istoclose = False
            for key1, key2 in tqdm(zip(rdmkeys, rdmkeys[1:]), total=len(rdmkeys)):
                if self.pathdict[key1].calc_distance(self.pathdict[key2]) < treshold:
                    istoclose = True
                    rdmkeys.remove(key1)
                    treshold -= 100
                    if treshold < 0:
                        treshold = 0
                    rdmkeys.append(rdm.choice([newkey for newkey in list
                                               (self.pathdict.keys()) if newkey not in rdmkeys]))

        centroids = {}
        for idx, key in enumerate(tqdm(rdmkeys)):
            centroids["cluster_"+str(idx+1)] = copy.deepcopy(self.pathdict[key])
        if plotit:
            self.plotwx(centroids)

        #centroidsold = {}
        ischanging = True
        while ischanging:
            centroidsold = copy.deepcopy(centroids)
            #connect centroids to closest trajectories to form clusters
            clusters = {}
            for key in centroids:
                clusters[key] = []

            for pathkey in tqdm(self.pathdict):
                mindistance = centroids[next(iter(centroids))].calc_distance(self.pathdict[pathkey])
                minkey = next(iter(centroids.items()))[0]
                for key in centroids:
                    newdistance = centroids[key].calc_distance(self.pathdict[pathkey])
                    if  newdistance < mindistance:
                        minkey = key
                        mindistance = newdistance
                clusters[minkey].append(pathkey)

            #reassign centroids with means of respective cluster
            for key in tqdm(centroids):
                centroids[key] = self.calc_mean_traj(clusters[key])
            if plotit:
                self.plotwx(centroids)
            totdist = 0
            for key in centroidsold:
                totdist += centroids[key].calc_distance(centroidsold[key])
            if totdist < 5:
                ischanging = False
                if plotit:
                    self.plotclusters(clusters)
                return clusters

    def calc_mean_traj(self, keys):
        """returns a mean trajectory given keys from pathdict"""
        samplecount = len(keys)
        #if samplecount == 0:
        #    raise ValueError('empty trajectory list')
        if samplecount == 1:
            return self.pathdict[keys[0]]
        newmeantraj = trajectory()
        pointsum = sum([self.pathdict[value].points for value in keys])
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
        for cluster, value in clusters.items():
            if len(cluster) > 1:
                #extracl all points from a cluster
                #points = np.array([point for sublist in [self.pathdict[key].points for key in clusters[cluster]] for point in sublist])
                points = np.empty([0,len(value),3])
                for i in range(0, len(self.pathdict[value[0]].points)):
                    temp = np.empty([0,3])
                    for key in value:
                        temp = np.append(temp, [self.pathdict[key].points[i,:]], axis=0)
                    #test = np.atleast_2d([temp])
                    points = np.append(points, np.atleast_2d([temp]), axis=0)

                #xs = sorted(points[:,0])
                #ys = sorted(points[:,1])
                
                #points = np.array([self.pathdict[key].points for key in value])
                [self.pathdict[key].normailise_timestamps() for key in value]
                #xs = np.array([self.pathdict[key].points[:,0:3:2] for key in value])#.T.reshape(-1,1)
                #ys = np.array([self.pathdict[key].points[:,1:3] for key in value])#.T.reshape(-1,1)
                
                pointsxt = np.empty([0,2])
                pointsyt = np.empty([0,2])
                
                for key in value:
                    pointsxt = np.append(pointsxt,self.pathdict[key].points[:,0:3:2], axis=0)
                    pointsyt = np.append(pointsyt,self.pathdict[key].points[:,1:3], axis=0)
                
                xs = pointsxt[:,0]
                ys = pointsyt[:,0]
                
                tx = pointsxt[:,1]
                ty = pointsyt[:,1]
                
                #xs = points[:,:,0]
                #ys = points[:,:,1] 
                
                #E = self.calc_mean_traj(clusters[cluster]) #expected values is E(X) calculated as the mean function
                
                lengthscale = 1.0
                n = 10
                
                #meanfuncs = trajectories()
                #meanfuncs.pathdict["1"] = trajectory()
                #meanfuncs.pathdict["1"].add_point([0,0,0])
                #meanfuncs.pathdict["1"].add_point([5,2,0])
                #meanfuncs.pathdict["1"].add_point([3,3,0])
                
                #meanfuncs.pathdict["2"] = trajectory()
                #meanfuncs.pathdict["2"].add_point([0, 0.5,0])
                #meanfuncs.pathdict["2"].add_point([5, 2.5,0])
                #meanfuncs.pathdict["2"].add_point([3, 2.5,0])
                
                #meanE = meanfuncs.calc_mean_traj(["1", "2"])
                
                theta_x,theta_y = (1.0,1.0)
                
                kx = theta_x*gp.kernels.RBF(lengthscale,(1e-2,1e2)) + gp.kernels.WhiteKernel(1.0)
                print("x kernel before:", kx)
                process_x = gp.GaussianProcessRegressor(kernel=kx, normalize_y=True, n_restarts_optimizer=n)
                
                ky = theta_y*gp.kernels.RBF(lengthscale,(1e-2,1e2)) + gp.kernels.WhiteKernel(1.0)
                print("y kernel before:", ky)
                process_y = gp.GaussianProcessRegressor(kernel=ky, normalize_y=True, n_restarts_optimizer=n)
                
                #seq = np.linspace(1, 1000, len(points))
                #seq_x = np.atleast_2d(np.linspace(0,  10, 1000)).T
                #process_x.fit(points[:,0].reshape(-1,1), seq_x.T)
                #test = np.interp(points[:,0], E.points[:,0], E.points[:,1])
                #process_x.fit(points[:,0].reshape(-1,1), test.T)
                #process_x.fit(E.points[:,0].reshape(-1,1), points[:,1].T)
                #process_x.fit(seq_x, points[:,0].T)
                #process_x.fit(np.atleast_2d(E.points[:,0]).T, points[:,0].T)
               
                #seq_x = np.atleast_2d(np.linspace(min(points[:,0]),  max(points[:,0]), len(points))).reshape(-1,1)
                #seq_x = np.atleast_2d(np.linspace(min(xs.flatten()),  max(xs.flatten()), len(xs))).reshape(-1,1)
                #seq_x = np.atleast_2d(np.linspace(0,  10, len(xs))).reshape(-1,1)
                #seq_x = np.atleast_2d(np.linspace(1,  10, len(points))).reshape(-1,1)
                #process_x.fit(seq_x, points[:,0])
                #process_x.fit(seq_x, xs)
                #process_x.fit(xs, seq_x)
                #process_x.fit(seq_x, xs)
                process_x.fit(tx.reshape(-1,1), xs.reshape(-1,1))
                #process_x.fit(E.points[:,0].reshape(-1,1), xs)
                #process_x.fit(xs, E.points[:,0])
                #for key,nextkey in zip(value, value[1:]):
                #    process_x.fit(self.pathdict[key].points[:,0].reshape(-1,1), self.pathdict[nextkey].points[:,0].reshape(-1,1))
                #[process_x.fit(self.pathdict[key].points[:,0].reshape(-1,1), self.pathdict[nextkey].points[:,0].reshape(-1,1)) for key, nextkey in zip(value, value[1:])]
                print("")
                print("x kernel after:", process_x.kernel_)
               
                #seq = np.linspace(1, 1000, len(points))
                #seq = np.linspace(min(points[:,1]),  max(points[:,1]), len(points))
                #process_y.fit(points[:,1].reshape(-1,1), seq.reshape(-1,1))
                #process_y.fit(points[:,1].reshape(-1,1), points[:,0].reshape(-1,1))

                #seq_y = np.atleast_2d(np.linspace(min(points[:,1]),  max(points[:,1]), len(points))).reshape(-1,1)
                #seq_y = np.atleast_2d(np.linspace(min(ys.flatten()),  max(ys.flatten()), len(ys))).reshape(-1,1)
                #seq_y = np.atleast_2d(np.linspace(0,10,len(ys))).reshape(-1,1)
                #seq_y = np.atleast_2d(np.linspace(1,  10, len(points))).reshape(-1,1)
                #process_y.fit(seq_y,points[:,1])
                #process_y.fit(ys, seq_y)
                #process_y.fit(seq_y, ys)
                process_y.fit(ty.reshape(-1,1), ys.reshape(-1,1))
                #process_y.fit(E.points[:,1].reshape(-1,1), ys)
                #process_y.fit(ys, E.points[:,1])
                #for key,nextkey in zip(value, value[1:]):
                #    process_y.fit(self.pathdict[key].points[:,1].reshape(-1,1), self.pathdict[nextkey].points[:,1].reshape(-1,1))
                #[process_y.fit(self.pathdict[key].points[:,1].reshape(-1,1), self.pathdict[nextkey].points[:,1].reshape(-1,1)) for key, nextkey in zip(cluster, cluster[1:])]
                print("y kernel after:", process_y.kernel_)
                print("")
                
               
                #predict_x,std_x = process_x.predict(xs, return_std=True);
                #predict_y,std_y = process_y.predict(ys, return_std=True);

                #predict_x,std_x = process_x.predict(E.points[:,0].reshape(-1,1), return_std=True);
                #predict_y,std_y = process_y.predict(E.points[:,1].reshape(-1,1), return_std=True);

                predict_x,std_x = process_x.predict(tx.reshape(-1,1), return_std=True);
                predict_y,std_y = process_y.predict(ty.reshape(-1,1), return_std=True);
                
                #xepoints = [cov_x+ex for ex in E.points[:,0]]
                #yepoints = [cov_y+ey for ey in E.points[:,1]]
                
                #nxepoints = [cov_x-ex for ex in E.points[:,0]]
                #nyepoints = [cov_y-ey for ey in E.points[:,1]]

                #test = process_x.kernel_.diag(points[:,0])
                
                fig = plt.figure()
                #plt.plot(seq_x[:,0], points[:,0], 'm', label='x-values')
                #xval = plt.plot(seq_x[:,0], xs, 'm', label='$x-values$')
                #plt.plot(seq_y[:,0], points[:,1], 'c', label='y-values')
                #yval = plt.plot(seq_y[:,0], ys, 'c', label='$y-values$')
                
                plt.plot(tx, xs, 'm', label='$x-values$')
                plt.plot(ty, ys, 'c', label='$y-values$')
                
                
                #plt.plot(predict_x, predict_y, 'blue', label='$mean-values$')
                
                plt.xlabel('$time$')
                plt.ylabel('$value$')
                plt.legend(loc='upper left')
               
                fig = plt.figure() 
                plt.scatter(tx, predict_x, c='m')
                #[plt.scatter(x+std, x-std, c='black') for x,std in zip(predict_x, std_x)]
                plt.plot(tx, xs, 'black')
               
                plt.scatter(ty, predict_y, c='c')
                #[plt.scatter(y+std, y-std, c='black') for y,std in zip(predict_y, std_y)]
                plt.plot(ty, ys, 'black')
                
                #plt.plot(E.points[:,0], E.points[:,1], 'b-', label=u'expected')
                #[plt.scatter(p, q, c='r') for p,q in zip(xepoints, yepoints)]
                #[plt.scatter(p, q, c='r') for p,q in zip(nxepoints, nyepoints)]
                #plt.scatter(points[:,0], seq, c='g')
                #plt.plot(cov_x[0], cov_x[1], 'r:', markersize=10, label='covariance')
                #[plt.plot(cov, nextcov, 'r:', markersize=10) for cov,nextcov in zip(cov_x[2:], cov_x[3:])]
                #[plt.plot(cov, nextcov, 'r:', markersize=10) for cov,nextcov in zip(cov_y, cov_y[1:])]
                #test =  cov_x*points[:,0].T
                #test2 = cov_y*points[:,1].T
                #test = []
                #for ((covx, covy), (px,py)) in zip(zip(cov_x, cov_y), zip(points[:,0], points[:,1])):
                #    test.append([px+covx*px, py+covy*py])
                #test = np.array([[px + covx*px, py + covy*py] for ((covx,covy),(px,py)) in zip(zip(cov_x, cov_y),zip(points[:,0], points[:,1]))])
                #test = np.array([[px + covx*px, py + covy*py] for ((covx,covy),(px,py)) in zip(zip(cov_x, cov_y),zip(points[:,0], points[:,1]))])
                #[plt.plot(var[0], var[1], c='r') for var in test]
                #[plt.scatter(var, nextvar, c='r') for var,nextvar in zip(test,test2)]
                #[plt.plot(cov*points[:,0], nextcov*points[:,1], markersize=10) for cov,nextcov in zip(cov_x, cov_y)]
                #plt.plot(x, y_pred, 'b-', label=u'Prediction')                
                #x = np.atleast_2d(np.linspace(0, 1000, len(cov_x)))
                
                #plt.plot(points[:,0], f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
                #plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
                #plt.plot(points[:,0], y_pred, 'b-', label=u'Prediction')
                
                #plt.errorbar(predict_x, predict_y, yerr=std_y,xerr=std_x, capsize=0) 
                #plt.fill(np.concatenate([tx, tx[::-1]]),
                #    np.concatenate ([
                #    predict_x - std_x,(
                #    predict_x + std_x)[::-1]]),
                #   alpha=.5, fc='b', ec='None')
                
                #plt.fill(np.concatenate([ty, ty[::-1]]),
                #    np.concatenate ([
                #    predict_y - std_x,(
                #    predict_y + std_x)[::-1]]),
                #    alpha=.5, fc='b', ec='None')
                
                
                #plt.xlabel('$x$')
                #plt.ylabel('$y$')
                
                #plt.ylim(-10, 20)
                #plt.scatter(points[:,0], points[:,1])

                #plt.legend(loc='upper left')                
                plt.figure()
                [plt.plot(self.pathdict[key].points[:,0], self.pathdict[key].points[:,1], 'g') for key in clusters[cluster]]                
                plt.show()
                #process.fit(trainingdata,E.points)
                #x_pred = np.linspace(-6,6)
                #y_pred, sigma = process.predict(x_pred, return_std=True)
                #plt.plot(x_pred, y_pred)
                #gp.GaussianProcessRegressor(,,"theta",,,,)

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

        for _ in range(10):
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



readcsvfile('testfile.csv',10)
trajs.filter_noise()
trajs.interpol_points(10)
#trajs.interpol_test(10)

#trajs.plot()

CLUSTERS = trajs.kmeansclustering(3, 600000)
trajs.generate_guassian_processes(CLUSTERS)

#trajs.pathdict[next(iter(trajs.pathdict))]. #test row for a n trajectory
#trajs.calc_mean_traj([])