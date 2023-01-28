# SENS - SALSA E Não Só (In English: SALSA and more)
Python implementation of optimization methods for inverse problems in imaging.

This repository contains python implementations of software originally developed in Matlab during my PhD and post-doctoral work at Instituto Superior Técnico, Lisbon, Portugal. For details about the algorithms, refer to the articles:

*M. Afonso, J. Bioucas-Dias, and M. Figueiredo, "Fast image recovery using variable splitting and constrained optimization," IEEE Transactions on Image Processing, vol. 19, no. 9, pp. 2345-2356, September, 2010

*M. Afonso, J. Bioucas-Dias, and M. Figueiredo, "An Augmented Lagrangian based Method for the Constrained Formulation of Imaging Inverse Problems", IEEE Transactions on Image Processing, Vol. 20, no. 3, pp 681 - 695, March, 2011.

*M. Afonso, J. Bioucas-Dias, and M. Figueiredo, "An augmented Lagrangian approach to linear inverse problems with compound regularization", IEEE International Conference on Image Processing ICIP'2010, Hong Kong, China, 2010.

Original Matlab version: http://cascais.lx.it.pt/~mafonso/salsa.html

*M. Afonso and J.M.R. Sanches, "Image reconstruction under multiplicative speckle noise using total variation," Neurocomputing, Volume 150, Part A, 20 February 2015, Pages 200-213, http://dx.doi.org/10.1016/j.neucom.2014.08.073.
Original Matlab version: https://github.com/manyaafonso/Rayleigh-Speckle-Denoising-and-Reconstruction

*M. Afonso and J.M.R. Sanches, "Blind Inpainting using L0 and Total Variation Regularization," Image Processing, IEEE Transactions on , Vol. 24, No. 7, July 2015, doi: 10.1109/TIP.2015.2417505

Original Matlab version: https://github.com/manyaafonso/Blind-Inpainting-l0-TV

# Installation and Usage

This code was developed and tested on a Linux Mint 20.0 system. To replicate the environment locally, use the following commands
1. git clone https://github.com/manyaafonso/pySENS.git

2. cd pySENS

3. conda create -n SENS python=3.8

4. conda activate SENS

5. pip install numpy matplotlib pillow opencv-python pywavelets jupyter notebook

OR use the requirements file:
pip install -r requirements.txt

6. jupyter notebook

Then execute the specific notebook.
