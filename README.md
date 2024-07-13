FuSITSNet fuses image time series from two satellites - Landsat-8 and MODIS. It can be used for multiple earth observation applications

Additionally we can include meteorological data (if applicable to application)

%%%%%%%%% Data Download %%%%%%%%%%%%%%%%%%

Download geotiff images from GEE for both the satellites.

*This data needs pre-processing for missing values and preparation in the form suitable to be used in ML models i.e. conversion into .npz files to be used as input to ML/DL models

 

%%%%%%%%% Data Preparation %%%%%%%%%%%%%%%%%%

Data preparation process is done individually for each satellite and each crop. Convert individual tif images into .npz files representing image time series for a year for each location.

%%%%%%%%% Model-FuSITSNet %%%%%%%%%%%%%%%%%%

The kernel size of CNN module in encoders need to be changes as per satellite and the timeseries length is different depending on satellite and crop.
