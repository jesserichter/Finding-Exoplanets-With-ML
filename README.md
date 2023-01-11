# Finding Exoplanets Using a Machine Learning Model

Inspired by [arXiv](https://arxiv.org/abs/1706.04319)

The code that paper used was hard to modify and used old data, so I recreated it with new data from TESS and used the wrapper for TensorFlow I helped develop in high school.


Dependencies: 
  * [Python3](https://www.continuum.io/downloads)
  * [Numpy](http://www.numpy.org/)
  * [Keras](https://keras.io/)
  * [TensorFlow](https://www.tensorflow.org/)
  * [Matplotlib](https://matplotlib.org/)
  * [Scikit-learn](http://scikit-learn.org/stable/)
  * [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)
  * [Exoplanet Light Curve Analysis](https://github.com/pearsonkyle/Exoplanet-Light-Curve-Analysis)



## File Structure 
`generate_data.py` - generates over 300,000 training and test samples from the parameter space grid in Table 1.

`quasiperiodicity.py` - Generates the variability shape analysis plot (Figure 4)

`transit_shape_analysis.py` - Generates various figures showing how the light curve shape changes as a function of the planet and orbit parameters. This plot is particularly useful when creating training data sets for exoplanets because not all parameters are going to yield an observable signal (careful! if you're unfamiliar with transiting exoplanet geometry you could easily bias your training data)

`model_fit_history` - creation of each neural network and its training. The training performance per epoch is saved and used to create Figure 6. 

`ROC_auc_score.py` - Creates the receiver operating characteristic plot to assess the accuracy of each algorithm. 

`graph_featureloss.py` - Explorations into the accuracy of the neural network if we're missing input data (Figure 11)

`graph_interpolate.py` - One of my favorite plots because it shows the accuracy of each algorithm after a signal has been interpolated from either a high or low resolution state. This is particularly useful to understand because it will allow one to apply this algorithm to an arbitrary transit survey without needing to retrain the network. Instead, just transform the input data to match what the network needs. Since we're dealing with a time sorted signal we can get away with this for the most part. Just make sure the transit signal is greater than a few data points. 

`graph_sensitivity.py` - Explores how the detection accuracy of each algorithm changes with signal to noise ratio. 

`timeseries_eval.py` - a simple script that will evaluate a time series light curve that is larger than the input for the neural network by breaking it up into smaller lightcurves. Think of it like a sliding box-car evaluation along the light curve. 
