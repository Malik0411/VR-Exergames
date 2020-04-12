# Virtual Reality Exergames
A collection of the code used to analyze data collected from an OculusQuest VR headset. The purpose of the research is to promote the use of exercise games.

# Dependencies
## Mayavi Installation
- To visualize some of what I'm doing here, I've decided to utilize a 3D plotting library, Mayavi. I'm using a Windows 10 device, and ran into quite a few issues initially trying to install the package.

### First Method
**Install using pip**: You can use a prompt such as PowerShell, or Git Bash to install the package. Make sure you have a newer version of python installed, I'm using 3.7.4 for the entirety of this analysis.

````$pip install mayavi````

**Note**: This process did not work for me

### Second Method
#### Install using Anaconda
1. Run the Anaconda Navigator as Administrator.
2. Install the mayavi environment and all dependencies.
3. Launch VSCode and attempt to utilize the mplot library.

**Note**: I was not able to simply install Mayavi from Anaconda and utilize the mplot library, I got numerous errors regarding the package dependencies that I could not resolve.

#### Install using mayavi-pip (through Anaconda)
1. Run the Anaconda Prompt as Administrator.
2. Execute the following commands:
````
conda create --name mayavi-pip python=3.7 vtk numpy traitsui configobj six
conda activate mayavi-pip
pip --no-cache-dir install mayavi
````
3. This should install all mayavi dependencies in this active environment.
4. Launch VSCode, make sure your python is set to the active version of python installed by mayavi-pip.
5. Attempt to utilize the mplot library (**this worked for me, just make sure you have a GUI toolkit installed too**).

## GUI Installation
- You can use any GUI toolkit you desire, for example: PyQt, wxWidgets, Tkinter, etc. I chose to utilize **PyQt5**
1. To install your GUI, I was able to utilize pip.
````$pip install PyQt5````

**Note**: If you run into errors, try installing a different one. If you utilize the **mayavi-pip** method for the Mayavi installation, try activating the environment.

## Other libraries
**Install using pip**: I was able to install all other dependencies using pip without trouble. You're going to need:
1. numpy
2. circle_fit
3. matplotlib
