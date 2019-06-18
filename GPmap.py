import csv
import math
import random as rdm
import copy
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import gaussian_process as gp

class trajectory:
    """A set of points which forms a trajectory"""
    def __init__(self):
        """init trajectory"""
        self.points = np.empty([0, 3])
        self.probability = []
        self.weighted_probability = []

    def get_highest_model_index(self):
        """returns model index of highest probability """
        return self.probability.index(max(self.probability))

    def add_point(self, point):
        """add point x,y timestamp"""
        self.probability = []
        self.points = np.append(self.points, [point], axis=0)

    def remove_point(self, point):
        """remove point as x,y, timestamp if this results to a empty trajectory return True"""
        self.points = np.delete(self.points, np.where(self.points == point)[0], axis=0)
        return bool(len(self.points) < 3)

    def calc_distance(self, traj):
        """calculates an abstract distance beetween self and a trajectory"""
        if not traj:
            return None
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
        """Rough filter that filters all points that is closer than threshold
        if a trajectory ends up having under 3 points it is removed
        """
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
                            isdead = self.pathdict[key].remove_point(nextpoint)
                        else:
                            isdead = self.pathdict[key].remove_point(point)
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
        """Decide k for the k-means clustering in early testing stages"""
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
        rdmkeys = [rdm.choice(list(self.pathdict.keys()))]
        counter = 0

        for _ in range(1, k):
            while True:
                newrdmkey = rdm.choice([newkey for newkey in list(self.pathdict.keys()) if newkey not in rdmkeys])
                mindist = min([self.pathdict[key].calc_distance(self.pathdict[newrdmkey]) for key in rdmkeys])
                if mindist > threshold:
                    rdmkeys.append(newrdmkey)
                    break;
                counter += 1
                if counter > 100:
                   counter = 0
                   threshold -= 100
                   threshold = max(threshold, 0)
                   tqdm.external_write_mode()
                   print("Generating centroids: Warning lowering threshold to ", threshold)

        centroids = {}
        for idx, key in enumerate(tqdm(rdmkeys)):
            centroids["cluster_"+str(idx+1)] = copy.deepcopy(self.pathdict[key])
        if plotit:
            self.plotwx(centroids)
        return centroids

    def kmeansclustering(self, k, threshold=200000, acceptible_distance=5, plotit=False, centroids=None):
        """generates k clusters with centroids that is further away than threshold"""
        if not centroids or len(centroids) != k:
            if len(centroids) != k:
                print("kmeans: Warning given centroids does not match k, generating new centroids!")
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

    def interpol_points(self, number_of_points = 10, plotresult = False):
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

    def generate_guassian_processes(self, clusters, plotit=False, numofsamples=100, plotcov=False,
                                    x_noise_bounds=(1e-5, 1.0), x_lengthscale_bounds=(10.0, 1e5),
                                    y_noise_bounds=(1e-5, 1.0), y_lengthscale_bounds=(10.0, 1e5)):
        """generates two gaussian proccesses for x and y for each cluster"""
        class struct(NamedTuple):
            """simple struct for the two procesess"""
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

                xs = np.array([self.pathdict[key].points[:,0] for key in value]).flatten()
                ys = np.array([self.pathdict[key].points[:,1] for key in value]).flatten()

                #tx = np.tile(np.linspace(0, len(self.pathdict[value[0]].points)-1,len(self.pathdict[value[0]].points)),len(value))
                if self.numofpoints <= 0:
                    raise ValueError("Not interpolated")
                #tx = np.tile(np.linspace(0, self.numofpoints-1, self.numofpoints), len(value)).reshape(-1,1)
                #ty = tx

                tx = np.array([self.pathdict[key].points[:,2] for key in value]).flatten().reshape(-1,1)
                ty = tx

                lengthscale_x = 1.0
                lengthscale_y = 1.0
                
                noise_x = 1.0
                noise_y = 1.0
                
                n_x = 10
                n_y = 10
                
                constant_x = 2.0 
                constant_y = 2.0

                theta_x,theta_y = (316.0,1.0)

                kx = theta_x*gp.kernels.RBF(lengthscale_x, x_lengthscale_bounds) + gp.kernels.WhiteKernel(noise_x, x_noise_bounds) + constant_x
                print("x kernel before:", kx)
                process_x = gp.GaussianProcessRegressor(kernel=kx, n_restarts_optimizer=n_x, normalize_y=False, alpha=0, copy_X_train=True)

                ky = theta_y*gp.kernels.RBF(lengthscale_y, y_lengthscale_bounds) + gp.kernels.WhiteKernel(noise_y, y_noise_bounds) + constant_y
                print("y kernel before:", ky)
                process_y = gp.GaussianProcessRegressor(kernel=ky, n_restarts_optimizer=n_y, normalize_y=False, alpha=0, copy_X_train=True)

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
                    plt.show()
                self.gaussian_processes[cluster] = struct(process_x, process_y)

    def gaussian_process_prediction(self, clusters, observations = None, plotit=False):
        """In early testing stages mostly the same as generate gaussian procesess"""
        numofsamples=100
        plotcov=True
        """predicts the future path of observations"""
        class struct(NamedTuple):
            """simple struct for the two procesess"""
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
                xs = np.empty([0, 2])
                ys = np.empty([0, 2])

                for key in value:
                    xs = np.append(xs, self.pathdict[key].points[:, 0:3:2], axis=0)
                    ys = np.append(ys, self.pathdict[key].points[:, 1:3], axis=0)

                xs = xs.T
                ys = ys.T
                #xs = np.array([self.pathdict[key].points[:,0:1] for key in value]).squeeze()
                #ys = np.array([self.pathdict[key].points[:,1] for key in value]).flatten()

                #tx = np.tile(np.linspace(0, len(self.pathdict[value[0]].points)-1,len(self.pathdict[value[0]].points)),len(value))
                if self.numofpoints <= 0:
                    raise ValueError("Not interpolated")
                #tx = np.tile(np.linspace(0, self.numofpoints-1, self.numofpoints), len(value)).reshape(-1,1)
                #ty = tx

                E = self.calc_mean_traj(value).points

                tx = E[:, 0:3:2].T
                ty = E[:, 1:3].T
                #tx = np.array([self.pathdict[key].points[:,2] for key in value]).flatten().reshape(-1,1)
                #ty = tx

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
                self.gaussian_processes[cluster] = struct(process_x, process_y)

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
        t = process.X_train_.flatten()
        seq = np.atleast_2d(np.linspace(min(t), max(t), len(t))).reshape(-1,1)

        num_of_parallel_trajs = np.count_nonzero(t==0)
        for x,y in zip(np.split(t, num_of_parallel_trajs), np.split(values, num_of_parallel_trajs)):
            plt.plot(x, y, c='black')


        #samples = process.sample_y(seq, numofsamples)
        #for sample in samples.T:
        #    plt.plot(seq, sample)

        if plotcov:
            predict,std = process.predict(seq, return_std=True)
            plt.scatter(seq, predict, c='m', zorder=len(values)+1, s=2)
            plt.fill_between(seq.flatten(), predict.flatten()-np.square(std).flatten(),
                             predict.flatten()+np.square(std).flatten(), color='b', alpha=0.2, zorder=len(values)+2)

    def plotproceses_in_xy_plane(self):
        for _, cluster in self.gaussian_processes.items():
            process_x = cluster.process_x
            process_y = cluster.process_y

            xs = process_x.y_train_
            ys = process_y.y_train_

            tx = process_x.X_train_
            ty = process_y.X_train_

            seq_x = np.atleast_2d(np.linspace(min(tx), max(tx), len(tx))).reshape(-1,1)
            seq_y = np.atleast_2d(np.linspace(min(ty), max(ty), len(ty))).reshape(-1,1)

            predict_x,std_x = process_x.predict(seq_x, return_std=True)
            predict_y,std_y = process_y.predict(seq_y, return_std=True)
            #plt.scatter(seq_x, predict_x, c='m', zorder=len(xs)+1)
            #plt.fill_between(predict_x.flatten(), predict_y.flatten()-np.square(std_y).flatten(),
            #                 predict_y.flatten()+np.square(std_y).flatten(), color='b', alpha=0.2)


            #plt.scatter(seq_y, predict_y, c='m', zorder=len(ys)+1)
            #plt.fill_between(predict_y.flatten(), predict_x.flatten()-np.square(std_x).flatten(),
            #                 predict_x.flatten()+np.square(std_x).flatten(), color='yellow', alpha=0.2)

            plt.scatter(predict_x, predict_y, c='m', zorder=len(self.pathdict)+1, s=2)
            #plt.text(predict_x[-1], predict_y[-1], "x_score: " + str(process_x.log_marginal_likelihood()) +
            #         " y_score: " + str(process_y.log_marginal_likelihood()))
        #plt.show()
        self.plot('black')

    def plotcompare(self, listoftrajs, colors):
        count = 0
        if not colors:
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
            paint = '#' + colors[count]
            count += 1

            plt.plot(traj.points[:, 0], traj.points[:, 1], paint)
            plt.scatter(traj.points[:, 0], traj.points[:, 1], color=paint)
        plt.show()
        colors.clear()

    def plotclusters(self, clusters):
        """plots all clusters in the parameter"""
        #plt.axis([-50000, 50000.0, -50000.0, 50000.0])# xmin, xmax, ymin, ymax
        plt.title("Clusters")
        numofclusters = len(clusters)
        colorrange = 10000
        colors = rdm.sample(range(colorrange), numofclusters)
        count = 0
        colors = []

        for _ in range(1000):
            colors.append('%06X' % rdm.randint(0, 0xFFFFFF))
        colors = ["red", "green", "blue", "black", "orange", "darkgreen",
                  "darkblue", "gold", "brown", "yellow", "pink", "purple", "grey"]

        for element in clusters:
            color = (colors[count])
           #color = cm.hot(float(colors[count])/colorrange)
           #color = cm.autumn(float(colors[count])/colorrange)
            for key in clusters[element]:
                plt.plot(self.pathdict[key].points[:, 0],
                         self.pathdict[key].points[:, 1], color)
            count += 1
        plt.show()

    def plotwx(self, x):
        """plots with x"""
        for key in x:
            plt.plot(x[key].points[:, 0], x[key].points[:, 1], "*", zorder=len(self.pathdict)+1)
        self.plot('lightgrey')

    def plot(self, color = None):
        """plot of all trajectories in x-y space"""
        plt.title("trajs in x-y space")
        for key in self.pathdict:
            if color:
                plt.plot(self.pathdict[key].points[:, 0], self.pathdict[key].points[:, 1], color)
            else:
                plt.plot(self.pathdict[key].points[:, 0], self.pathdict[key].points[:, 1])
        plt.show()

    def plot_in_timespace(self, color = None):
        """plots all trajectories in timestamp space"""

        pointsxt = np.empty([0,2])
        pointsyt = np.empty([0,2])

        for traj in self.pathdict:
            pointsxt = np.append(pointsxt,trajs.pathdict[traj].points[:,0:3:2], axis=0)
            pointsyt = np.append(pointsyt,trajs.pathdict[traj].points[:,1:3], axis=0)

        xs = pointsxt[:,0]
        ys = pointsyt[:,0]

        tx = pointsxt[:,1]
        ty = pointsyt[:,1]

        plt.figure()
        plt.title("x values in timepace")
        for idx,(x,y) in enumerate(zip(np.split(tx,len(trajs.pathdict)), np.split(xs, len(trajs.pathdict)))):
            plt.plot(x, y)

        plt.figure()
        plt.title("y values in timepace")
        for idx,(x,y) in enumerate(zip(np.split(ty,len(trajs.pathdict)), np.split(ys, len(trajs.pathdict)))):
            plt.plot(x, y)
        plt.show()
        
class Model():
    def __init__(self, list_of_points):
        self.traj = copy.deepcopy(list_of_points)

class GaussianMixtureModel():

    def __init__(self, trajs,K,covariance,M, init_model_keys ,init_points = None,):
        self.trajs = copy.deepcopy(trajs)
        self.T = len(self.trajs.pathdict[next(iter(self.trajs.pathdict.items()))[0]].points)
        self.M = M
        self.beta = self.T/K
        self.K = K
        self.covariance = covariance
        self.init_model_keys = init_model_keys
        self.models = []
        self.set_models()

    def set_models(self):
        #Set all
        num_of_trajectories = len(self.trajs.pathdict)
        for i in range(self.M):
            for t_key in self.trajs.pathdict:
                self.trajs.pathdict[t_key].weighted_probability.append(0.0)
                self.trajs.pathdict[t_key].probability.append(0.0)

        diviates = [[155.0,230.0,0.0],[-125.0,-230.0,0.0],[-125.0,-230.0,0.0]]
        for t_key in self.init_model_keys:
            self.models.append(self.init_model_keys[t_key].points)

        self.plot_models(True)
        
        
    def plot_models(self, first_run = False):
        colors = ["red", "green", "blue", "black", "orange", "darkgreen",
                  "darkblue", "gold", "brown", "yellow", "pink", "purple", "grey"]
        for t_key in self.trajs.pathdict:
            if first_run:
                plt.plot(self.trajs.pathdict[t_key].points[:,0],self.trajs.pathdict[t_key].points[:,1],color= "lightblue")
            else:
                traj_color = self.trajs.pathdict[t_key].get_highest_model_index()
                plt.plot(self.trajs.pathdict[t_key].points[:,0],self.trajs.pathdict[t_key].points[:,1],color= colors[traj_color])
                
        for c_i in range(self.M):
            plt.scatter(self.models[c_i][:,0], self.models[c_i][:,1], alpha=0.5, marker=r'*',color=colors[c_i], zorder = len(self.trajs.pathdict)+1)
            #plt.plot(self.models[c_i][:,0], self.models[c_i][:,1], c = colors[c_i])
        plt.show()     

    def plot(self):
        for c_i in range(self.M):
            clusters = np.empty([0,2])
            colors = ["red", "green", "blue", "black", "orange", "darkgreen",
                      "darkblue", "gold", "brown", "yellow", "pink", "purple", "grey"]
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
        self.plot_models()    

    def get_mean_model_i(self,m): # Get the "beta" cluster
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


    def gaussian_1_dimension(self,point,m,j):
        c = math.exp((-1/(2*(self.covariance)**2))*(np.linalg.norm(point[:-1]-self.models[m][j,:2])**2))
        if np.isnan(c) or c < (2.2250738585072014e-308)**(1/self.T):
            res = (2.2250738585072014e-308)**(1/self.T)
        else:
            res = c
        return res

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
                bayesian_matrix[m][t_i] = quote



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
        for c_i in range(self.M):
            self.models[c_i] = self.get_mean_model_i(c_i)
        #*****************************************************************#
        return max_div
    
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
            if laps% 40 == 0:
                self.plot()
        print("-DONE-")
        self.plot()

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

def generate_data(M,T, init_points, noise, Number_of_trajs):
    global trajs
    global models_created
    rdm_trajs =[]
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
                new_traj.add_point([point[0],point[1], t])
            tr.append(new_traj)
            trajs.add_trajectory(i,new_traj)
            i += 1
        models_created.append(tr[-1])
            #j += 3


"""Global Variables"""
trajs = trajectories()


"""Main calls
here you can tweak different parameters
"""
trajstoread = 100
readcsvfile('testfile.csv', trajstoread)
trajs.filter_noise(plotit=False)
#avarage amount of points simple estimate for the interpolation
#n = sum([len(trajs.pathdict[key].points) for key in trajs.pathdict])/len(trajs.pathdict)

n = 37 #for 100 trajs
#n = 36 #for 50 trajs
#n = 23  #for 10 trajs
#n = 10

trajs.interpol_points(int(n))
#trajs.plot()
#trajs.plot_in_timespace()
#trajs.normalize_timestamps()
#trajs.plot_in_timespace()

T = n
K = T #"Hyperparameter"
M = 4 #number of clusters

CENTROIDS = trajs.generate_centroids(M, plotit=True, threshold=252000)#152000
#odels_created = [traj for _, traj in CENTROIDS.items()]
#init_points = [0, 455, 1050]
#noise = 10
#Number_of_trajs = 40
#generate_data(M,T, init_points, noise, Number_of_trajs)

covariance = 1930
GMM = GaussianMixtureModel(trajs,K,covariance,M, CENTROIDS)
GMM.GMM()

#K=11 #for 100 trajs
#K=2 #for 10 trajs
#trajs.elbow_method_for_k(2, trajstoread-20)
#26 for 50
K=M

CLUSTERS = trajs.kmeansclustering(K, 16000, 5, True, centroids=CENTROIDS)[0]
#CLUSTERS = trajs.kmeansclustering(K, 6500, 5, True)[0]
#trajs.numofpoints = T
trajs.generate_guassian_processes(CLUSTERS, True, plotcov=True, x_noise_bounds=(1e-5, 10000.0), y_noise_bounds=(1e-5, 10000.0),
                                  x_lengthscale_bounds=(10.0, 1e5), y_lengthscale_bounds=(10.0, 1e5))
#trajs.gaussian_process_prediction(clusters=CLUSTERS)
trajs.plotproceses_in_xy_plane()

