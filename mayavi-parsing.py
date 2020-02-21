import numpy as np
from mayavi.mlab import *
from mayavi import mlab
import csv

def plot_with_quiver3d():
    x, y, z = np.mgrid[-2:3, -2:3, -2:3]
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
            lposition.append([row[0].rpartition('\t')[2][1:], row[1], row[2].rpartition('\t')[0][:-1]])
            rposition.append([row[2].rpartition('\t')[2][1:], row[3], row[4].rpartition('\t')[0][:-1]])
            lvelocity.append([row[4].rpartition('\t')[2][1:], row[5], row[6].rpartition('\t')[0][:-1]])
            rvelocity.append([row[6].rpartition('\t')[2][1:], row[7], row[8].rpartition('\t')[0][:-1]])
            laccel.append([row[8].rpartition('\t')[2][1:], row[9], row[10].rpartition('\t')[0][:-1]])
            raccel.append([row[10].rpartition('\t')[2][1:], row[11], row[12][1:-1]])
        print('{} {} {} {} {} {}'.format(lposition[0], rposition[0], lvelocity[0], rvelocity[0], laccel[0], raccel[0]))
    # plot_with_quiver3d()
    # mlab.show()