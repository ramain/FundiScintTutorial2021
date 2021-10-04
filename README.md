# FundiScintTutorial2021
Intro to looking for scintillation in pulsar data


The main tutorial notebook goes through the steps of creating dynamic spectra and deriving scintillation parameters from a pulsar archive.  Some handy functions are found in dynspectools.py - as shown in the tutorial notebook, with these functions one can write a very simple, general script to run on any archive or dynspec without the intermediate steps


If you just want to take a quick look at a pulsar's dynamic spectrum, one can do this simply with psrplot:
eg.  psrplot -pj -D /xw archivename.ar

You can also create a dynamic spectrum directly uing psrflux, as eg.  psrflux archivename.ar.  This writes an ascii wile with the S/N in each subintegration, along with the time, frequency labels for each.  dynspectools.py contains simple functions to read / write psrflux files with python


The notebook Annual_Scintillation.ipynb shows an example of a fit to annual variations in the arc curvature, deriving screen distances and velocities.  I urge people who are interested in applying this to solve binary orbits to read Reardon et al 2020, and look at the scintools package.