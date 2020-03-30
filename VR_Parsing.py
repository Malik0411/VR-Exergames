import numpy as np
import scipy.spatial.distance as sp
from mayavi.mlab import *
from mayavi import mlab
import csv
import circle_fit as cf
from re import split
import matplotlib.pyplot as plt
from Radar_Plot import radar_factory

def plot_with_quiver3d(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 4)
    u = y * np.sin(r) / (r + 0.001)
    v = -x * np.sin(r) / (r + 0.001)
    w = np.zeros_like(z)
    return quiver3d(x, y, z, u, v, w, line_width=3, scale_factor=1)

def plot_with_triangular_mesh(x, y, z, n):
    t = np.linspace(-np.pi, np.pi, n)
    triangles = [(0, i, i + 1) for i in range(1, n)]
    x = np.r_[0, x]
    y = np.r_[0, y]
    z = np.r_[1, z]
    t = np.r_[0, t]
    return triangular_mesh(x, y, z, triangles, scalars=t)

if __name__ == "__main__":
    # Data splits at 12000 (forward rowing), 25000 (backwards rowing), and alternating left/right rowing thereafter
    with open('C:/Users/Malik/Documents/University of Waterloo/3A/URA/2020-03-13/data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        time = []; lposition = []; rposition = []; lvelocity = []; rvelocity = []; laccel = []; raccel = []
        for row in csv_reader:
            # First line does not contain useful data
            if line == 0:
                line += 1
                continue
            time.append(split(r'\D+', row[0].rpartition('\t')[0][12:-3]))
            lposition.append([float(row[0].rpartition('\t')[2][1:]), float(row[1]), float(row[2].rpartition('\t')[0][1:-1])])
            rposition.append([float(row[2].rpartition('\t')[2][1:]), float(row[3]), float(row[4].rpartition('\t')[0][1:-1])])
            lvelocity.append([float(row[4].rpartition('\t')[2][1:]), float(row[5]), float(row[6].rpartition('\t')[0][1:-1])])
            rvelocity.append([float(row[6].rpartition('\t')[2][1:]), float(row[7]), float(row[8].rpartition('\t')[0][1:-1])])
            laccel.append([float(row[8].rpartition('\t')[2][1:]), float(row[9]), float(row[10].rpartition('\t')[0][1:-1])])
            raccel.append([float(row[10].rpartition('\t')[2][1:]), float(row[11]), float(row[12][1:-1])])
            line += 1

    x = []; y = []; z = []; xp = []; yp = []; zp = []
    data = [['LP', 'RP', 'LV', 'RV', 'LA', 'RA'], ('OculusQuest VR Data', [])]
    for i in range(770, 1000):
        xp.append(lposition[i][0]); yp.append(lposition[i][1]); zp.append(lposition[i][2])
        x.append(lvelocity[i][0]); y.append(lvelocity[i][1]); z.append(lvelocity[i][2])
        
        temp = []
        temp.append(np.linalg.norm(lposition[i])); temp.append(np.linalg.norm(rposition[i]))
        temp.append(np.linalg.norm(lvelocity[i])); temp.append(np.linalg.norm(rvelocity[i]))
        temp.append(np.linalg.norm(laccel[i])); temp.append(np.linalg.norm(raccel[i]))
        data[1][1].append(temp)

    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    xp = np.asarray(xp); yp = np.asarray(yp); zp = np.asarray(zp)
    
    # Spider Plot Visualization
    N = 6
    theta = radar_factory(N, frame='polygon')
    spoke_labels = data.pop(0)
    title, case_data = data[0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title,  position=(0.5, 1.1), ha='center')

    for d in case_data:
        ax.plot(theta, d)
        ax.fill(theta, d, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    plt.show()

    # # Scatter visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xp, yp, zp, marker='o')
    # plt.show()

    # Triangular mesh (mayavi)
    # plot_with_triangular_mesh(x, y, z, 12000)
    # mlab.show()

    # # 3D Quiver Plotting with mayavi (better velocity representation)
    # plot_with_quiver3d(x, y, z)
    # mlab.show()

    # # Make the direction data for the arrows
    # u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    # v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    # w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z))
    # ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
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
    # while end <= line:
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
        
    # # Check whether user is stopping/starting and changing directions
    # # Idea is to check when velocity/acceleration are 0 during motion, and then suddenly reverse
    # # Y or Z points could be used, but since y is negative at the top of the circle (when vector is parallel to axis) it is more convenient to use
    # no_change = True
    # for i in range(0, len(y)-1):
    #     if y[i] < 0.1 and y[i] > 0:
    #         above_zero = 1
    #         below_zero = 0
    #         continue
    #     elif y[i] > -0.1 and y[i] < 0:
    #         above_zero = 0
    #         below_zero = 1
    #         continue

    #     if (above_zero == 1 and y[i] < 2*r) or (below_zero == 1 and y[i] > 2*r):
    #         no_change = False

    #     if no_change is False and above_zero == 1 and y[i] < y[i-1]:
    #         print("Direction switched to backward at {}. Current: {}. Previous: {}".format(i, y[i], y[i-1]))
    #         above_zero = 0
    #         no_change = True

    #     if no_change is False and below_zero == 1 and y[i] > y[i-1]:
    #         print("Direction switched to forward at {}. Current: {}. Previous: {}".format(i, y[i], y[i-1]))
    #         below_zero = 0
    #         no_change = True
          