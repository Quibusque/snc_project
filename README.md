# Project for Software e Computing: recognition of regions from street-view images

## Authors
- Marco Cruciani
- Francesco Zenesini

Description
==============================
This project builds a simple CNN neural network that tries to recognize the location of images provided by a google street-view dataset on kaggle: https://www.kaggle.com/datasets/nikitricky/streetview-photospheres .
Images are classified based on climate regions provided by this github repository The Multi-MIP Climate Change ATLAS https://github.com/SantanderMetGroup/ATLAS .

Usage 
=============
The user can use the interactive_notebook.ipynb to train the model and see its performance on the dataset.
The dataset is downloaded from kaggle, the user needs to have a kaggle account and provide the username and key in the notebook. 
The notebook is built assuming to be run on google colab, and it is recommended to run it with a GPU runtime.
The notebook can be found at the following link: https://colab.research.google.com/drive/1Uo7sVzYFC-i7nlCmu-nu3_JVJr2HdcO8

Running the tests
--------------
For testing we used pytest, the tests are in the `source/tests` folder.
To run the tests, you can use the command `pytest` in the root folder of the project.

Documentation
===============
The documentation of the project is available in the docs folder, and it can be
viewed by opening the index.html file in a browser.
full path is `docs/_build/html/index.html`

Climate regions references
==========================
> Iturbide, M., Fernández, J., Gutiérrez, J.M., Bedia, J., Cimadevilla, E., Díez-Sierra, J., Manzanas, R., Casanueva, A., Baño-Medina, J., Milovac, J., Herrera, S., Cofiño, A.S., San Martín, D., García-Díez, M., Hauser, M., Huard, D., Yelekci, Ö. (2021) Repository supporting the implementation of FAIR principles in the IPCC-WG1 Atlas. Zenodo, DOI: 10.5281/zenodo.3691645. Available from: https://github.com/IPCC-WG1/Atlas 