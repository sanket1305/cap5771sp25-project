# Movie Recommendation System

Welcome to the Movie Recommendation System project! This repository contains the code for a Data Science project focused on building a movie recommendation system using data analysis and visualization techniques.

---

## Project Overview

The system is built on three datasets (detailed in the "Milestone 1.pdf" report). The code in this repository processes these datasets to perform data analysis, create insightful visualizations, and build a recommendation model.

To run the project, please review the following key files in the repository:

---

### Files Overview

1. **.ipynb**: Contains all the backend coe from data sourcing, model training to model evaluation
2. **Scripts/data_splits/**: This folder exists on drive link given below in `Data Files` section. This folder was saved during model training to keep track of data split
3. **Scripts/models**: contains all the models which were trained as part of project. Again this can be found in drive link or can be generated using the guide given below to `Run the Project`
4. **Images/**: Contains the graph images generated during the analysis. These images are saved for visual reference.
5. **Report/**: Includes milestone reports, including "Milestone 1.pdf," which provides an overview of the datasets and the project.
6. **Data/**: Includes the data files used for the project (Note: Large files are not included in this repository due to size constraints).
7. **m2_plots/** and **figures/**: Contains the plots which were generaed through .ipnb files during execution.

---

### Data Files

Some of the data files used in this project are too large to upload directly to GitHub due to file size limitations. However, you can download the full dataset (which contains the exact data files used in the project) from the following location:  
[Download Data Files](https://drive.google.com/drive/folders/1O3tv2h5cheKzi6Cub4i18PPCmK4Swqf_?usp=sharing)

[Download whole project setup here](https://drive.google.com/file/d/1zRe9nXSz_McriMku-5k5eAUtHhRLIype/view?usp=sharing)

### Running the Project

**Note:** It is recommended to have conda env setup and then install required dependencies in it. Once done you can launch below mentioned jypter labs from Anacoda UI

1. Start with `Milestone1.pynb`, this will load the data do some preprocessing and store the data in normalise format in `/Data` directory
2. Once, you have data in DB in normalise format. Run `Milestone2.ipynb`, which will train 4 models Linear Regression, Logistic Regression, K-Nearest Neighbor and XGBoost and save them in files in `models` folder
3. Now, run the `Milestone2_v2.ipynb`, which will train 3 more models KMeans, KNN and HDBScan models. This models along with theire respective features and labels will be store in `models` folder under nested directories
4. run the `Milestone3.ipynb`, which will perform testing for all 6 DISTINCT models and evaluate their performances
5. run `stremlit run dashboard.py` from terminal and it will open web UI in your deafult browser.

### Prerequisites

Make sure to have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- sqlite
- seaborn
- scikit-learn
- plotly
- hdbscan
- streamlit

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly hdbscan streamlit
```

**Note:** If asks to install more packages, then please install them too before runnning the application again.

### How to Run

- Download the dataset (instructions provided above).
- Place the dataset files in the Data/ folder.
- There are 2 .ipynb files
- Execute them in order of milestones

This will trigger the data analysis and generate plots, which will be saved in the Images/ folder.
