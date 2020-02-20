# Virtual Reality Exergames
A collection of the code used to analyze data collected from an OculusQuest VR headset. The purpose of the research is to aid elderly individuals with dementia, promoting the use of exercise games to support their health journey.

# Dependencies
## Mayavi Installation
- To visualize some of what I'm doing here, I've decided to utilize a 3D plotting library, Mayavi. I'm using a Windows 10 device, and ran into quite a few issues initially trying to install the package.

### First Method
**Install using pip**: You can use a prompt such as PowerShell, or Git Bash to install the package. Make sure you have a newer version of python installed, I'm using 3.7.4 for the entirety of this analysis.

````$pip install mayavi```` 

````$pip install PyQt5````


**Note**: This process did not work for me

### Second Method
#### Install using Anaconda
1. Run the Anaconda Navigator as Administrator.
2. Install the **Mayavi** environment and all dependencies.
3. Launch VSCode and attempt to utilize the mplot library.

**Note**: I was not able to simply install Mayavi from Anaconda and utilize the mplot library, I got numerous errors regarding the package dependencies that I could not resolve.
