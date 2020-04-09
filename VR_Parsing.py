""" 
    OculusQuest VR Headset - Data Parsing
    By: Malik Salvosa

    The contents of this script are to be used in visualizing the data of an OculusQuest VR headset
    I have sectioned off the parts of the script that perform different tasks (uncomment as desired to test functionality)
    Note: Other headsets can likely be used, although you may have to change how the .csv data output file is parsed

"""


import csv
import numpy as np
from re import split
from mayavi import mlab
import circle_fit as cf
from mayavi.mlab import *
import matplotlib.pyplot as plt
import scipy.spatial.distance as sp
from Radar_Plot import radar_factory
from mpl_toolkits.mplot3d import axes3d, Axes3D


def plot_with_quiver3d(x, y, z):
    """
        This function is used to visualize headset data using mayavi's Quiver3D

        Args:
            x: numpy array of data arranged along the x-axis
            y: numpy array of data arranged along the y-axis
            z: numpy array of data arranged along the z-axis

    """
    # Component vectors used to calculate the direction 
    # of the quiver arrows (standard calculation, see mayavi docs)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 4)
    u = y * np.sin(r) / (r + 0.001)
    v = -x * np.sin(r) / (r + 0.001)
    w = np.zeros_like(z)
    quiver3d(x, y, z, u, v, w, line_width=3, scale_factor=1)
    mlab.show() 


def plot_with_triangular_mesh(x, y, z, n):
    """
        This function is used to visualize headset data using mayavi's triangular mesh

        Args:
            x: numpy array of data arranged along the x-axis
            y: numpy array of data arranged along the y-axis
            z: numpy array of data arranged along the z-axis
            n: number of triangles to create
            
    """
    # Scalars calculation to describe the extremity of the triangles
    t = np.linspace(-np.pi, np.pi, n)
    # Creating the list of tuples for triangle vertices
    triangles = [(0, i, i + 1) for i in range(1, n)]
    x = np.r_[0, x]
    y = np.r_[0, y]
    z = np.r_[1, z]
    t = np.r_[0, t]
    triangular_mesh(x, y, z, triangles, scalars=t)
    mlab.show()


if __name__ == "__main__":
    # Data used for processing has the following characteristics:
    # 0-12000 (forward rowing)
    # 12000-23000 (backwards rowing)
    # 23000-25000 (nothing)
    # 25000-31000 (forward right-hand rowing only)
    # 31000-end (forward left-hand rowing only)
    with open('C:/Users/Malik/Documents/University of Waterloo/3A/URA/2020-03-13/Malik Rowing.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        time = []; lposition = []; rposition = []; lvelocity = []; rvelocity = []; laccel = []; raccel = []
        for row in csv_reader:
            # Skip the first line since it does not contain useful data
            if line == 0:
                line += 1
                continue
            # Parsing the raw data from the csv file at the correct indices
            # Converted to float where possible, partitioned tabs, and removed extra spacing
            time.append(split(r'\D+', row[0].rpartition('\t')[0][12:-3]))
            lposition.append([float(row[0].rpartition('\t')[2][1:]), float(row[1]), float(row[2].rpartition('\t')[0][1:-1])])
            rposition.append([float(row[2].rpartition('\t')[2][1:]), float(row[3]), float(row[4].rpartition('\t')[0][1:-1])])
            lvelocity.append([float(row[4].rpartition('\t')[2][1:]), float(row[5]), float(row[6].rpartition('\t')[0][1:-1])])
            rvelocity.append([float(row[6].rpartition('\t')[2][1:]), float(row[7]), float(row[8].rpartition('\t')[0][1:-1])])
            laccel.append([float(row[8].rpartition('\t')[2][1:]), float(row[9]), float(row[10].rpartition('\t')[0][1:-1])])
            raccel.append([float(row[10].rpartition('\t')[2][1:]), float(row[11]), float(row[12][1:-1])])
            line += 1

    # Amount of data you want to use for visualization
    data_points = 12000
    
    # Array initialization to store the number of desired data points
    xv = []; yv = []; zv = []; xp = []; yp = []; zp = []
    lp = []; rp = []; lv = []; rv = []; la = []; ra = []
    for i in range(0, data_points):
        # Appending the position and velocity (currently left controller, but could be either left or right)
        # These are used when calculating rps and then determining if the user changes circle diameter drastically
        xp = np.append(xp, lposition[i][0]); yp = np.append(yp, lposition[i][1]); zp = np.append(lposition[i][2])
        xv = np.append(xp, lvelocity[i][0]); yv = np.append(yp, lvelocity[i][1]); zv = np.append(lvelocity[i][2])

        # Appending the magnitude of the x, y, z data for position, velocity, and acceleration
        # These are used when crafting features for the spider plot
        lp = np.append(lp, np.linalg.norm(lposition[i])); rp = np.append(rp, np.linalg.norm(rposition[i]))
        lv = np.append(lv, np.linalg.norm(lvelocity[i])); rv = np.append(rv, np.linalg.norm(rvelocity[i]))
        la = np.append(la, np.linalg.norm(laccel[i])); ra = np.append(ra, np.linalg.norm(raccel[i]))

    # # Spider Plot Parameters (types of features can change as desired)
    # lp = max(lp); rp = max(rp)
    # lv = sum(lv)/len(lv); rv = sum(rv)/len(rv)
    # la = sum(la)/len(la); ra = sum(ra)/len(ra)

    # # 3D Quiver Plotting
    # plot_with_quiver3d(x, y, z)

    # # Triangular Mesh
    # plot_with_triangular_mesh(x, y, z, data_points)

    # # Spider Plot Visualization
    # N = 6
    # theta = radar_factory(N, frame='polygon')
    # data = [['Max LP', 'Max RP', 'Avg LV', 'Avg RV', 'Avg LA', 'Avg RA'], ('OculusQuest VR Data', [
    #     [lp, rp, lv, rv, la, ra]
    #     ])]
    # spoke_labels = data.pop(0)
    # title, case_data = data[0]

    # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(top=0.85, bottom=0.05)
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    # ax.set_title(title,  position=(0.5, 1.1), ha='center')

    # for d in case_data:
    #     ax.plot(theta, d)
    #     ax.fill(theta, d, alpha=0.25)
    # ax.set_varlabels(spoke_labels)

    # plt.show()

    # # Scatter visualization
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(xp, yp, zp, marker='o')
    # plt.show()

    # # 3D Plane Wire Frame Graph Chart
    # xx, yy = np.meshgrid(x,y)
    # def z_function(x,y):
    #     return np.sqrt(x**2 + y**2)
    # LP = np.asarray(lposition)
    # z = np.array([z_function(x,y) for (x,y) in zip(np.ravel(xx), np.ravel(yy))])
    # zz = z.reshape(xx.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(xx,yy,zz, rstride=10, cstride=10)
    # plt.show()

    # # Calculation of Circle Radius
    # # Started by idealizing data to use Z and Y data points and excluding slight variations due to arm motion in X direction.
    # data = np.column_stack((zp,yp))
    # # Calculation is best with positional data, since this more accurately defines the perimeter of the circle
    # xc, yc, r, variance = cf.least_squares_circle(data)
    # print(r)
    
    # # To visualize the data points used for radius calculation
    # plt.scatter(x, y)
    # plt.show()

    # # Calculation of revolutions per second
    # # Calculation of angle between x, y points
    # def angle_between(p1, p2):
    #     deltax = p2[0] - p1[0]
    #     deltay = p2[1] - p1[1]
    #     return np.arctan2(deltay, deltax)
    
    # angularData = []
    # for i in range(0, len(data)-1):
    #     angularData.append(angle_between((data[i][0], data[i][1]), (data[i+1][0], data[i+1][1])))
    
    # angularVelocity = []
    # for i in range(0, len(angularData)-1):
    #     angularVelocity.append((angularData[i+1]-angularData[i])/((float(time[i+1][1])+float(time[i+1][2])/1000)-(float(time[i][1])+float(time[i][2])/1000)))
    # angularVelocity = [x for x in angularVelocity if str(x) != 'nan' and str(x) != 'inf' and str(x) != '-inf']
    # rps = abs((sum(angularVelocity)/len(angularVelocity))/(0.104719755*60))
    # print(rps)

    # # Identifying changing diameters of the circles made by the user
    # start = 0; prev = 0
    # # Frequency of data collection (calculated around 220Hz)
    # end = int(220/rps); curr = 0
    # while end <= data_points:
    #     xp = []; yp = []; zp = []
    #     for i in range(start, end):
    #         xp.append(lposition[i][0])
    #         yp.append(lposition[i][1])
    #         zp.append(lposition[i][2])
    #     xp = np.asarray(xp); yp = np.asarray(yp); zp = np.asarray(zp)
        
    #     data = np.column_stack((zp,yp))
    #     curr = np.nanmax(sp.pdist(data))

    #     if prev != 0 and curr > prev + 0.25:
    #         print(prev, curr)
    #     prev = curr
    #     start = end
    #     end += int(220/rps)
        