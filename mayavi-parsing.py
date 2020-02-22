import numpy as np
from scipy import interpolate
from mayavi.mlab import *
from mayavi import mlab
import csv

def plot_with_quiver3d(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 4)
    u = y * np.sin(r) / (r + 0.001)
    v = -x * np.sin(r) / (r + 0.001)
    w = np.zeros_like(z)
    obj = quiver3d(x, y, z, u, v, w, line_width=3, scale_factor=1)
    return obj

if __name__ == "__main__":
    with open('C:/Users/Malik/Documents/University of Waterloo/3A/URA/Wrist-Flick Up x2, Wrist-Flick Side x2.csv') as csv_file:
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
        x.append(raccel[i][0])
        y.append(raccel[i][1])
        z.append(raccel[i][2])
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)

    plot_with_quiver3d(x, y, z)
    mlab.show()

    ## Method 1 (Fast) but makes z mesh nxn (very large for lots of data)
    # xx, yy = np.meshgrid(x,y)
    # def z_function(x,y):
    #     return np.sqrt(x**2 + y**2)
    # LP = np.asarray(lposition)
    # z = np.array([z_function(x,y) for (x,y) in zip(np.ravel(xx), np.ravel(yy))])
    # zz = z.reshape(xx.shape)

    ## Method 2 (Too slow)
    # zz = interpolate.griddata((LP[:,0], LP[:,1]), LP[:,2], (xx, yy), method='nearest')