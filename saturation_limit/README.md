# [saturation_limit.ipynb](saturation_limit.ipynb) #
For a point source, under three LSST fiducial atmosphere, we look at the saturation limits in each band, and make plots of peak electron counts vs. AB magnitude. A flat SED is assumed for the point source.
# [saturation_limit_trail.ipynb](saturation_limit_trail.ipynb) #
Instead of a point source, we look at a moving satellite which flies by at angular velocity of 0.5 deg/second. The profile of the moving trail is assumed to be a Gaussain with FWHM of 1.5 arcsec. We again look at the saturation limits in each band, and make plots of peak electron counts vs. AB magnitude. We give the satellite solar color (even though for a fixed magnitude in a certain optical band, it gives the same ADU as a flat SED.)

The FWHM of the trail and the angular velocity are specified at the very top of the notebook, so that we can easily rerun this if different values for these are of interest.
