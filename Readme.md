- - - -
# 1. Project Title:
## Using PySpark for Image Classification on Satellite Imagery of Agricultural Terrains

**Course :** CS696 - Big Data Tools and Methods

**Team Members:**
- Shah, Saumil
- Naidu, Indraraj

- - - -

# 2. Description
The goal of this project is to use Big Data Tools and Methods for Image Classification on Satellite Imagery of Agricultural Terrains. For this project we have used dataset available on [DeepSat (SAT-6) Airborne Dataset on Kaggle]([DeepSat (SAT-6) Airborne Dataset | Kaggle](https://www.kaggle.com/crawford/deepsat-sat6)).  We have used python implementation of Apache Spark - PySpark and AWS Elastic MapReduce Instances (EMR) for setting up clusters on cloud and performing this experiment.

## 2.1 About the Dataset
This is a dataset contains images from the National Agriculture Imagery Program (NAIP) dataset. It is a subset of the large NAIP dataset covering terrains over the State of California. Original NAIP dataset contained images which were converted to ```28 * 28``` image patches by the author which contains total of ```4 channels```, namely *Red, Green Blue and Infrared*.  Hence, once image is flattened it will have total of ```28*28*4 = 3136``` features. The dataset from the Kaggle is already split into 80-20 ratio, where training data contains  ```324,000``` images and testing data contains  ```81,000``` images with their corresponding one-hot labels. To create one-hot labels there are total  ```6 categories``` —  ```‘water’, ‘road’, ‘grassland’, ‘trees’, ‘barren_land’, ‘building’``` each corresponding to class label 1 to 6 respectively.

## 2.2 Code Outline
**High-level Overview**
    - Load Data
    - Transform Data
    - Feature Extraction (PCA)
    - Model Training (Random Forest)
    - Model Testing
    - Model Evaluation (Performance Statistics, Confusion Matrix)

- - - -

# 3. Special Instructions
Default ```base_dir``` for dataset is ```deepsat-sat6```. Hence, try keeping the notebook and this directory in the same folder. It also the notebook assumes by default that data generated from AWS is stored under  ```fromAWS``` folder. For ease of use, data generated from our experiments from AWS is provided in that directory.

Since, the actual dataset is a space consuming ```(~5.6 GB)```, we have only provided a small set of that dataset, original dataset can be downloaded from the above link of Kaggle which only has ```first 200 rows``` with filenames having suffix ```_200```.

Following file structure is advised: (in case of errors, please consult this)
```
./
... Project-DeepSAT.ipynb
... project_deepsat_aws.py
... requirements.txt
... deepsat-sat6/
... ... sat6annotations.csv
... ... test_X_200.csv
... ... test_y_200.csv
... ... train_X_200.csv
... ... train_y_200.csv

... fromAWS/
... ... pca_200.model
... ... predictionAndLabels.csv
... ... random_forest.model
```

If downloading the full dataset use the following file structure:
```
... Project-DeepSAT.ipynb
... project_deepsat_aws.py
... requirements.txt
... deepsat-sat6/
... ... sat6annotations.csv
... ... X_test_sat6.csv
... ... X_train_sat6.csv
... ... y_test_sat6.csv
... ... y_train_sat6.csv

... fromAWS/
... ... pca_200.model
... ... predictionAndLabels.csv
... ... random_forest.model
```

**Note**
- Here, filename ```pca_200.model``` could be different, based on the number of pca components chosen, here it is ```k=200```.

- For running ```project_deepsat_aws.py```,  more information can be found inside the file.  If you want test the find, you can supply option ```--demo```, which will only take rows 5 rows. This behaviour can be changed in the code by supplying different values.

```
#To run locally
python project_deepsat_aws.py \
-bd "./deepsat-sat6/" \
--demo -p 10 -t 5

#To run on AWS
python s3://cs696-project-deepsat/project_deepsat_aws.py \
-bd "s3://cs696-project/" -od "s3://cs696-project-deepsat/" \
--demo -p 10 -t 5
```

- Also, ```AWS CLI Export``` is provided in the notebook for setting up clusters on AWS.

- Logs generated from the AWS clusters have been provided under ```./Logs``` directory.

- - - -

# 4. Additional Libraries
You can install necessary packages to run these codes by running the following:
```pip install -r requirements.txt```

- - - -

# 5. Known Issues
Loading entire dataset on local machine causes```Java Heap Memory ``` issues, hence use small toy dataset with suffix ```_200``` or consult the notebook for more details.
- - - -
