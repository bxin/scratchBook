# [saturation_limit.ipynb](saturation_limit.ipynb) #
For a point source, under three LSST fiducial atmosphere, we look at the saturation limits in each band, and make plots of peak electron counts vs. AB magnitude. A flat SED is assumed for the point source.

* Need to finalize typical bright sky values

This notebook produces these plots
* [Peak_count_by_band_darkSky.pdf](./plots/Peak_count_by_band_darkSky.pdf)
* [Peak_count_by_band_brightSky.pdf](./plots/Peak_count_by_band_brightSky.pdf)

# [saturation_limit_trail.ipynb](saturation_limit_trail.ipynb) #
Instead of a point source, we look at a moving satellite which flies by at angular velocity of 0.5 deg/second. The profile of the moving trail is assumed to be a Gaussain with FWHM of 1.5 arcsec. We again look at the saturation limits in each band, and make plots of peak electron counts vs. AB magnitude. We give the satellite solar color (even though for a fixed magnitude in a certain optical band, it gives the same ADU as a flat SED.)

The FWHM of the trail and the angular velocity are specified at the very top of the notebook, so that we can easily rerun this if different values for these are of interest.

* Need to finalize typical bright sky values

This notebook produces these plots
* [Peak_count_by_band_trail_darkSky.pdf](./plots/Peak_count_by_band_trail_darkSky.pdf)
* [Peak_count_by_band_trail_brightSky.pdf](./plots/Peak_count_by_band_trail_brightSky.pdf)


# [saturation_limit_pt_trail.ipynb](saturation_limit_pt_trail.ipynb) #
This notebook is almost identical to [saturation_limit_trail.ipynb](saturation_limit_trail.ipynb), we just changed the input and look at the results if a satellite is painted black except one bright spot. In this case the Gaussian has a FWHM of 0.7 arcsec.

* Need to finalize typical bright sky values

This notebook produces these plots
* [Peak_count_by_band_pt_trail_darkSky.pdf](./plots/Peak_count_by_band_pt_trail_darkSky.pdf)
* [Peak_count_by_band_pt_trail_brightSky.pdf](./plots/Peak_count_by_band_pt_trail_brightSky.pdf)

# [satLim_exploreHeight.ipynb](satLim_exploreHeight.ipynb)
In this notebook we explore the 2D parameter space formed by angular velocity and FWHM. There are three discrete values for the satellite height: 320km, 550km, and 1150km. For simplicity, we assume a zenith angle of 40 degrees (this is a parameter at the very top of the notebook so that is can be easily changed) and the orbit of the satellite goes through zenith. The angular velocity is then a function of the height. If the size of the satellite is 3m, the FWHM of the trail will also be a function of the height.

* Need to finalize typical bright sky values

This notebook produces these plots
* [Peak_count_by_band_550km_darkSky.pdf](./plots/Peak_count_by_band_550km_darkSky.pdf)
* [Peak_count_by_band_550km_brightSky.pdf](./plots/Peak_count_by_band_550km_brightSky.pdf)

# [saturation_limit_ucd.ipynb](saturation_limit_ucd.ipynb) #
For reference, we converted code provided by Tony Tyson and Craig Lage at UC Davis into a notebook. This notebook does same thing as [saturation_limit.ipynb](saturation_limit.ipynb), but using the Exposure Time Calculator (ETC), with out-of-date inputs, and they were hardcoded.
