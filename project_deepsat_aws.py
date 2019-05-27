# !/usr/bin/env python
# coding: utf-8
#
# # Project - Using PySpark for Image Classification on Satellite Imagery of Agricultural Terrains
#
# **Course :** CS696 - Big Data Tools and Methods
#
# **Team Members:**
# - Shah, Saumil : 82319571
# - Naidu, Indraraj : 823383841
# ---

## 1. Imports
import os
import sys
import random
import numpy as np
import argparse
from itertools import chain


import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as fn

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.feature import PCA as PCA_Spark
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# from sklearn.metrics import confusion_matrix

### Helper Functions
def spark_shape(self):
    return (self.count(), len(self.columns))


def join_X_and_y(X, y):

    X_new = X.withColumn("col_index", fn.monotonically_increasing_id().alias("rowId"))
    y_new = y.withColumn("col_index", fn.monotonically_increasing_id().alias("rowId"))

    joined_X_y = X_new.join(y_new, "col_index", 'inner').drop("col_index")
    print(joined_X_y.shape())
    return joined_X_y


def aws_spark_demo(base_dir='./deepsat-sat6', output_dir='./deepsat-sat6', demo_mode=True, demo_rows_n=5, pca_k=5, num_trees=4):

    spark = SparkSession.builder.appName("CS696-Project-DeepSAT-AWS").getOrCreate()

    reader = spark.read
    reader.option("header", "false")
    reader.option("inferSchema", "true")
    pyspark.sql.dataframe.DataFrame.shape = spark_shape


    print("\nbase_dir: {}, output_dir: {}, demo_mode: {}, demo_rows_n: {}, pca_k: {}, num_trees: {}".format(base_dir, output_dir, demo_mode, demo_rows_n, pca_k, num_trees))

    ### 2. Define Base Directory and Sub File Paths
    ann_data_path = os.path.join(base_dir, 'sat6annotations.csv')

    X_train_data_path = os.path.join(base_dir, 'X_train_sat6.csv')
    y_train_path = os.path.join(base_dir, 'y_train_sat6.csv')

    X_test_data_path = os.path.join(base_dir, 'X_test_sat6.csv')
    y_test_path = os.path.join(base_dir, 'y_test_sat6.csv')
    # print(os.listdir(base_dir))


    ann_spark = reader.csv(ann_data_path, sep=',', header=False)
    ann_spark = ann_spark.orderBy(fn.asc("_c1"), fn.asc("_c2"), fn.asc("_c3"), fn.asc("_c4"), fn.asc("_c5"), fn.asc("_c6"))
    ann_spark.show()

    category_names = np.array([c['_c0'] for c in ann_spark.select('_c0').collect()])
    print(category_names)


    #### Mapping of One-hot Labels to Categories
    total_categories = len(category_names)
    one_hot_labels_dict = { '0'*(total_categories-i-1)+'1'+'0'*(i): float(i+1)  for i in range(total_categories) }

    mapping_expr = fn.create_map([fn.lit(x) for x in chain(*one_hot_labels_dict.items())])
    print(one_hot_labels_dict)


    ## 3. Read Data

    #### Training Data
    if demo_mode:
        X_train_spark = reader.csv(X_train_data_path, sep=',').limit(demo_rows_n)
    else:
        X_train_spark = reader.csv(X_train_data_path, sep=',')
    print("\nX_train_spark.shape(): {}".format(X_train_spark.shape()))

    if demo_mode:
        y_train = reader.csv(y_train_path, sep=',').limit(demo_rows_n)
    else:
        y_train = reader.csv(y_train_path, sep=',')
    print("y_train.shape(): {}".format(y_train.shape()))


    #### Testing Data
    if demo_mode:
        X_test_spark = reader.csv(X_test_data_path, sep=',').limit(demo_rows_n)
    else:
        X_test_spark = reader.csv(X_test_data_path, sep=',')
    print("\nX_test_spark.shape(): {}".format(X_test_spark.shape()))


    if demo_mode:
        y_test = reader.csv(y_test_path, sep=',').limit(demo_rows_n)
    else:
        y_test = reader.csv(y_test_path, sep=',')
    print("y_test.shape(): {}".format(y_test.shape()))


    #### Misc Attributes
    img_dim = 28
    channels = 3
    n_features = img_dim**2 * channels

    first_k_principal_components = pca_k
    num_forest_trees = num_trees
    category_cols = [fn.col('_c'+str(x)) for x in range(6)]

    ## 4. Data Transformation Pipeline Stages

    #### Stages of X (data)
    vec_assembler = VectorAssembler(inputCols=['_c'+str(x) for x in range(n_features)], outputCol="features")
    standard_scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

    pca_spark = PCA_Spark(k=first_k_principal_components, inputCol="scaledFeatures", outputCol="features")
    forest_classifier = RandomForestClassifier(numTrees = num_forest_trees)

    X_pipeline = Pipeline(stages=[vec_assembler, standard_scaler])


    #### Stages of y (labels)
    def transform_y(y_dataframe, category_cols_ids, mapping_expr):
        y_category = y_dataframe.select(fn.concat(*tuple(category_cols_ids)).alias('label'))
        y_category = y_category.withColumn('label', mapping_expr[y_category['label']])

        return y_category

    ### Feature Extraction

    #### Vector Assembly and Scaling
    X_train = X_pipeline.fit(X_train_spark).transform(X_train_spark).select("scaledFeatures")
    print("\nX_train.shape(): {}".format(X_train.shape()))

    y_train_category = transform_y(y_train, category_cols, mapping_expr)
    print("y_train_category.shape(): {}".format(y_train_category.shape()))

    X_test = X_pipeline.fit(X_test_spark).transform(X_test_spark).select("scaledFeatures")
    print("\nX_test.shape(): {}".format(X_test.shape()))

    y_test_category = transform_y(y_test, category_cols, mapping_expr)
    print("y_test_category.shape(): {}".format(y_test_category.shape()))


    #### PCA
    pca_model = pca_spark.fit(X_train)
    print("\n{:.2f}% Variance Captured by {} components out of {} features.".format(100*sum(pca_model.explainedVariance),
                                                                                    first_k_principal_components,
                                                                                    n_features))
    X_train_reduced = pca_model.transform(X_train).select("features")

    X_y_train = join_X_and_y(X_train_reduced, y_train_category)
    X_y_train.show(10)


    X_test_reduced = pca_model.transform(X_test).select("features")

    X_y_test = join_X_and_y(X_test_reduced, y_test_category)
    X_y_test.show(10)

    X_y_test_path = os.path.join(base_dir, 'X_y_test.csv')
    print("Saving X_y_test at {}...".format(X_y_test_path))

    X_y_test.toPandas().to_csv(X_y_test_path)
    print("X_y_test Saved at {}".format(X_y_test_path))

    sys.exit(3)

    ## 5. Classification and Prediction

    ### Random Forest

    #### Training Model
    random_forest_model = forest_classifier.fit(X_y_train)
    random_forest_model.trees

    model_save_path = os.path.join(output_dir, 'random_forest.model')
    print("\nModel Trained.")

    random_forest_model.write().overwrite().save(model_save_path)
    print("Model Saved at {}".format(model_save_path))


    #### Predict using Model
    load_forest = RandomForestClassificationModel.load(model_save_path)
    model_pred = load_forest.transform(X_y_test)
    model_pred.show(5)

    predictionAndLabels = model_pred.select(['prediction', 'label']).rdd.map(lambda line: (line[0], line[1]))
    pred_labels_path = os.path.join(output_dir, 'predictionAndLabels.csv')
    print("\nPredictions made on Test Data.")


    predictionAndLabels.toDF().coalesce(1).write.csv(pred_labels_path, sep=',', header='true', mode='overwrite')
    print("predictionAndLabels Saved at {}".format(pred_labels_path))



# ---
if __name__ == '__main__':
    """Satellite Data Classification

    Data Source URL: https://www.kaggle.com/crawford/deepsat-sat6
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-bd', '--base_dir', help='Base Directory for Satellite Data')
    parser.add_argument('-od', '--output_dir', help='Output Directory for Satellite Data')
    parser.add_argument('-d', '--demo', help='Demo Mode', action='store_true')
    parser.add_argument('-p', '--pca_k', help='Number of PCA Components', type=int)
    parser.add_argument('-t', '--num_trees', help='Number of Random Forest Trees', type=int)

    # python project_deepsat_aws.py
    # s3://cs696-project-deepsat/project_deepsat_aws.py

    # -bd "./deepsat-sat6/" - To Run Locally
    # -bd "s3://cs696-project/" - To Run on AWS

    # -od "s3://cs696-project-deepsat/" - To Run on AWS
    # --demo -p 10 -t 5

    """ Sample Usage """
    # python project_deepsat_aws.py \
    # -bd "./deepsat-sat6/" \
    # --demo -p 10 -t 5

    # python s3://cs696-project-deepsat/project_deepsat_aws.py \
    # -bd "s3://cs696-project/" -od "s3://cs696-project-deepsat/" \
    # --demo -p 10 -t 5

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)


    if not args.base_dir:
        satellite_data_dir = "./deepsat-sat6/"
    else:
        satellite_data_dir = args.base_dir

    if not args.output_dir:
        data_output_dir = satellite_data_dir
    else:
        data_output_dir = args.output_dir


    if not args.demo:
        demo = False
    else:
        demo = True

    if not args.pca_k:
        pca = 4
    else:
        pca = args.pca_k

    if not args.num_trees:
        trees = 3
    else:
        trees = args.num_trees

    # """ Breakpoint """
    # sys.exit(1)

    # AWS CLI Demo Function to run all calculations
    aws_spark_demo(base_dir=satellite_data_dir, output_dir=data_output_dir, demo_mode=demo, pca_k=pca, num_trees=trees)
