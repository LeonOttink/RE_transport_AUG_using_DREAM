# Estimation of fast electron transport during an ASDEX Upgrade thermal quench
This repository contains the python code used for my internship at KTH Royal Institute of Technology in association with Eindhoven University of Applied Sciences with the above title. Asside from standard python packages, the [DREAM code](https://github.com/chalmersplasmatheory/DREAM) is used. 

The code consists of three core files and some sublimentary code:
- `disrupt.py`: main file running disruption simulations,
- `setup.py`: setup of DREAM and other settings,
- `userInpur.py`: parameters and profiles to be tuned by the user based on the simulated scenario. Many of these parameters can also be changed for a single simulation by using terminal commands.

Additionally, `postProcess.py` contains the code used for making the plots in the internship report and for computing the SSF's given the proper DREAM output files, and finally `GeriMap.py` provides the matplotlib colormap used for many plots, which is copied from the [DREAM directory](https://github.com/chalmersplasmatheory/DREAM).
