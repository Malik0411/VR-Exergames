import numpy as np
from mayavi.mlab import *
from mayavi import mlab
import csv

# Look at Quiver 3d from axes 3d
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def plot_with_quiver3d(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 4)
    u = y * np.sin(r) / (r + 0.001)
    v = -x * np.sin(r) / (r + 0.001)
    w = np.zeros_like(z)
    obj = quiver3d(x, y, z, u, v, w, line_width=3, scale_factor=1)
    return obj

def plot_with_contour3d(x, y, z):
    scalars = x * x * 0.5 + y * y + z * z * 2.0
    obj = contour3d(x, y, z, scalars, contours=4, transparent=True)
    return obj

if __name__ == "__main__":
    with open('C:/Users/Malik/Documents/University of Waterloo/3A/URA/2020-02-29/Left Side, Circular Motion.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        lposition = []; rposition = []; lvelocity = []; rvelocity = []; laccel = []; raccel = []
        for row in csv_reader:
            if line == 0:
                line += 1
                continue
            lposition.append([float(row[0].rpartition('\t')[2][1:]), float(row[1]), float(row[2].rpartition('\t')[0][1:-1])])
            rposition.append([float(row[2].rpartition('\t')[2][1:]), float(row[3]), float(row[4].rpartition('\t')[0][1:-1])])
            lvelocity.append([float(row[4].rpartition('\t')[2][1:]), float(row[5]), float(row[6].rpartition('\t')[0][1:-1])])
            rvelocity.append([float(row[6].rpartition('\t')[2][1:]), float(row[7]), float(row[8].rpartition('\t')[0][1:-1])])
            laccel.append([float(row[8].rpartition('\t')[2][1:]), float(row[9]), float(row[10].rpartition('\t')[0][1:-1])])
            raccel.append([float(row[10].rpartition('\t')[2][1:]), float(row[11]), float(row[12][1:-1])])
            line += 1

    x = []; y = []; z = []
    for i in range(0, line-1):
        x.append(lvelocity[i][0])
        y.append(lvelocity[i][1])
        z.append(lvelocity[i][2])
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    
    # 3D Quiver Plotting with mayavi (better velocity representation)
    plot_with_quiver3d(x, y, z)
    mlab.show()

    # # 3D Quiver Plotting with mplotlib (better position representation)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

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