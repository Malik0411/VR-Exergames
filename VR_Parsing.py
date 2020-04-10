""" 
    OculusQuest VR Headset - Data Parsing
    By: Malik Salvosa

    The contents of this script are to be used in visualizing the data of an OculusQuest VR headset
    I have sectioned off the parts of the script that perform different tasks (uncomment as desired to test functionality)
    Note: Other headsets can likely be used, although you may have to change how the .csv data output file is parsed

"""


import csv
import warnings
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
    # Component vectors used to calculate the direction of the quiver arrows (standard calculation, see mayavi docs)
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
            n: number of triangles to create (must be the esame as the number of data points)
            
    """
    # Scalars calculation to describe the extremity of the triangles
    t = np.linspace(-np.pi, np.pi, n)

    # Creating the list of tuples for triangle vertices
    triangles = [(0, i, i + 1) for i in range(1, n)]
    
    # Producing the axial data and plotting it
    x = np.r_[0, x]
    y = np.r_[0, y]
    z = np.r_[1, z]
    t = np.r_[0, t]
    triangular_mesh(x, y, z, triangles, scalars=t)
    mlab.show()


def spider_plot(n, parameters, parameter_data):
    """
        This function is used to visualize headset data using a spider (radar) plot

        Args:
            n: number of parameters to be used on the spider plot            
            parameters (list): names of parameters to be used on the spider plot
            parameter_data (list): data on the parameter data plotting on the spider plot

    """
    # Creating the geometric shape based on number of parameters
    theta = radar_factory(n, frame='polygon')

    # Producing the data set based on given parameters
    data = [parameters, ('OculusQuest VR Data', [parameter_data])]
    spoke_labels = data.pop(0)
    title, case_data = data[0]

    # Creating the plot accoridng to the mplotlib definition of radar plots
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title,  position=(0.5, 1.1), ha='center')

    # Plotting each data point on the chart
    for d in case_data:
        ax.plot(theta, d)
        ax.fill(theta, d, alpha=0.25)
    ax.set_varlabels(spoke_labels)
    plt.show()

def scatter_plot(x, y, z):
    """
        This function is used to visualize a scatter plot of the headset data

        Args:
            x: numpy array of data arranged along the x-axis
            y: numpy array of data arranged along the y-axis
            z: numpy array of data arranged along the z-axis

    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, marker='o')
    plt.show()


def get_circle_radius(y, z):
    """
        This function uses the circle fit library to approximate the circle radius for a set of y and z data
        Note: used y and z because these are the axes of motion used when making circles with the controllers

        Args:
            y: numpy array of data arranged along the y-axis
            z: numpy array of data arranged along the z-axis
        
        Returns:
            The calculated radius for the provided dataset

    """
    # Started by idealizing data to use Y and Z data points and excluding slight variations due to arm motion in X direction.
    data = np.column_stack((yp,zp))
    # Calculation is best with positional data, since this more accurately defines the perimeter of the circle
    xc, yc, r, variance = cf.least_squares_circle(data)
    return r


def angle_between(p1, p2):
    """
        This is a helper function used to calculate the angle between two points

        Args:
            p1: first data point, consisting of (y, z) data at a current time
            p2: second data point, consisting of (y, z) data at the next time
        
        Returns:
            The calculated angle between the two data points

    """
    # Calculation of angle between y, z points
    deltay = p2[0] - p1[0]
    deltaz = p2[1] - p1[1]
    return np.arctan2(deltaz, deltay)


def get_rps(y, z, time):
    """
        This function is used to calculate the rps data based on the velocity difference between each data point
        Note: used y and z because these are the axes of motion used when making circles with the controllers

        Args:
            y: numpy array of data arranged along the y-axis
            z: numpy array of data arranged along the z-axis
            time: list of time data for each data point
        
        Returns:
            The calculated revolutions per second for the given dataset

    """
    # Ignore runtime warnings (occurs due to nan, and inf calculations)
    warnings.filterwarnings("ignore")

    # Started by idealizing data to use Y and Z data points and excluding slight variations due to arm motion in X direction.
    data = np.column_stack((y,z))
    
    angularData, angularVelocity = [], []
    for i in range(0, len(data)-2):
        # Calculating the angle between the next two data points
        angularData.append(angle_between((data[i][0], data[i][1]), (data[i+1][0], data[i+1][1])))
        angularData.append(angle_between((data[i+1][0], data[i+1][1]), (data[i+2][0], data[i+2][1])))

        # Calculating the angular velocity (rad/s) based on the next two angular data points
        # Division of time is next(minutes + seconds + milliseconds) - current(minutes + seconds + milliseconds)
        angularVelocity.append((angularData[i+1]-angularData[i])/((60*time[i+1][0]+time[i+1][1]+time[i+1][2]/1000)-(60*time[i][0]+time[i][1]+time[i][2]/1000)))
        
    # Parsing out the invalid values (typically due to negligible time difference between points)
    angularVelocity = [x for x in angularVelocity if str(x) != 'nan' and str(x) != 'inf' and str(x) != '-inf'] 
    
    # Averaging the calculated rate for each point
    # 1 rad/s = 6.2831853108075 rps
    return abs((sum(angularVelocity)/len(angularVelocity))/6.2831853108075)


def get_changed_circle_size(rps, data, time, tolerance=0.15):
    """
        This function is used to identify when the user drastically changes the size of the circles they are making
        Sensitivity of detection can be adjusted by setting the tolerance argument

        Args:
            rps: the number of revolutions per second, to determine the creation of each circle
            data: the set of x, y, z data
            time: list of time data for each data point
            tolerance: the tolerance to use when determining how 'drastically' the circles change
        
        Returns:
            The list of circles that changed as tuples (circle1, circle2)

    """
    # Frequency of data collection (total number of points / total time)
    freq = len(data)/((60*time[len(data)-1][0]+time[len(data)-1][1]+time[len(data)-1][2]/1000)-(60*time[0][0]+time[0][1]+time[0][2]/1000))
    
    # List to hold the radii of the circles that changed
    changes = []

    # End signifies the approximate number of data points for the player to make a circle
    start, prev, curr, end = 0, 0, 0, int(freq/rps)
    while end <= len(data):
        y, z = [], []
        # Compile y and z data points that make an approximate circle
        for i in range(start, end):
            y, z = np.append(y, data[i][1]), np.append(z, data[i][2])
        curr = get_circle_radius(y, z)

        # Determine if the circles differ by the given tolerance
        if prev != 0 and abs(curr - prev) > tolerance:
            changes.append((prev, curr))

        prev = curr
        start = end
        end += int(freq/rps)

    return changes


if __name__ == "__main__":
    # Data used for processing has the following characteristics:
    # 0-12000 (forward rowing)
    # 12000-23000 (backwards rowing)
    # 23000-25000 (nothing)
    # 25000-31000 (forward right-hand rowing only)
    # 31000-end (forward left-hand rowing only)
    with open('C:/Users/Malik/Documents/University of Waterloo/3A/URA/2020-03-13/Malik Rowing.csv') as csv_file:
        # Skip the first line of the file (not useful)
        next(csv_file)

        # Parse through the, denoting for csv files
        csv_reader = csv.reader(csv_file, delimiter=',')
        time, lposition, rposition, lvelocity, rvelocity, laccel, raccel = [], [], [], [], [], [], []

        for row in csv_reader:
            # Partitioned tabs, and removed extra spacing from time data, then converted to float
            time.append([float(x) for x in split(r'\D+', row[0].rpartition('\t')[0][12:-3])])

            # Parsing the raw data from the csv file at the correct indices
            lposition.append([float(row[0].rpartition('\t')[2][1:]), float(row[1]), float(row[2].rpartition('\t')[0][1:-1])])
            rposition.append([float(row[2].rpartition('\t')[2][1:]), float(row[3]), float(row[4].rpartition('\t')[0][1:-1])])
            lvelocity.append([float(row[4].rpartition('\t')[2][1:]), float(row[5]), float(row[6].rpartition('\t')[0][1:-1])])
            rvelocity.append([float(row[6].rpartition('\t')[2][1:]), float(row[7]), float(row[8].rpartition('\t')[0][1:-1])])
            laccel.append([float(row[8].rpartition('\t')[2][1:]), float(row[9]), float(row[10].rpartition('\t')[0][1:-1])])
            raccel.append([float(row[10].rpartition('\t')[2][1:]), float(row[11]), float(row[12][1:-1])])

    # Amount of data you want to use for visualization
    data_points = 12000
    
    # Array initialization to store the number of desired data points
    xv, yv, zv, xp, yp, zp = [], [], [], [], [], []
    lp, rp, lv, rv, la, ra = [], [], [], [], [], []

    for i in range(0, data_points):
        # Appending the position and velocity (currently left controller, but could be either left or right)
        # These are used when calculating rps and then determining if the user changes circle diameter drastically
        xp, yp, zp = np.append(xp, lposition[i][0]), np.append(yp, lposition[i][1]), np.append(zp, lposition[i][2])
        xv, yv, zv = np.append(xv, lvelocity[i][0]), np.append(yv, lvelocity[i][1]), np.append(zv, lvelocity[i][2])

        # Appending the magnitude of the x, y, z data for position, velocity, and acceleration
        # These are used when crafting features for the spider plot
        lp, rp = np.append(lp, np.linalg.norm(lposition[i])), np.append(rp, np.linalg.norm(rposition[i]))
        lv, rv = np.append(lv, np.linalg.norm(lvelocity[i])), np.append(rv, np.linalg.norm(rvelocity[i]))
        la, ra = np.append(la, np.linalg.norm(laccel[i])), np.append(ra, np.linalg.norm(raccel[i]))

    # # Quiver 3D plotting
    # plot_with_quiver3d(xv, yv, zv)

    # # Triangular Mesh
    # plot_with_triangular_mesh(xv, yv, zv, data_points)

    # # Spider plot visualization
    # N = 6
    # parameters = ['Max LP', 'Max RP', 'Avg LV', 'Avg RV', 'Avg LA', 'Avg RA']
    # parameter_data = [max(lp), max(rp), sum(lv)/len(lv), sum(rv)/len(rv), sum(la)/len(la), sum(ra)/len(ra)]
    # spider_plot(N, parameters, parameter_data)

    # # Scatter Plot of position data
    # scatter_plot(xp, yp, zp)

    # # Calculation of Circle Radius
    # print('{0:.2f}m'.format(get_circle_radius(yp, zp)))

    # # Calculation of revolutions per second (using velocity because it is more precise than position)
    # print('{0:.2f} rps'.format(get_rps(yv, zv, time)))

    # Identifying drastic change in diameters of the circles made by the user
    changes = get_changed_circle_size(get_rps(yv, zv, time), lposition, time, tolerance=0)
    print('The user drastically changed diameters {} times, from (circle1, circle2) in: {}'.format(len(changes), changes))
