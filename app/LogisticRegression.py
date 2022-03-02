#!/usr/bin/env python
# coding: utf-8

# # Prepare the Data
# ## Set Constants
# Set constants

# In[1]:


import os

ROOT = r"./"
DATA_IN = os.path.join(ROOT, "data", "income-predictor")
SOURCE_DATASET = os.path.join(DATA_IN ,"adult.data")
SOURCE_ATTRIBUTES = os.path.join(DATA_IN, "attributes.csv")
SOURCE_TEST_DATA = os.path.join(DATA_IN, "adult.test")
COMBINED_DATA_OUT = f'{SOURCE_DATASET}.full'

LABEL_COL='>50K'


print(SOURCE_DATASET)
print(SOURCE_ATTRIBUTES)
print(SOURCE_TEST_DATA)


# ## Instantiate Spark Session

# In[2]:


from pyspark.sql import SparkSession, DataFrame

spark=SparkSession     .builder     .appName("Income Predictor")     .getOrCreate()


# ## Create Schema
# An attributes file was created from the [data's decription](http://archive.ics.uci.edu/ml/datasets/Census+Income)

# In[3]:


from pyspark.sql.types import StructType, StructField, IntegerType, DecimalType, StringType, BooleanType

attrs = spark.read.csv(SOURCE_ATTRIBUTES, header=True)
attrs.show()

def get_field(t: str):
    if t == "continuous":
        return DecimalType()
    elif t == "string":
        return StringType()
    elif t == "boolean":
        return BooleanType()
    else:
        raise Exception("Not expected: %s" % t)

def to_struct(row) -> StructField:
    return StructField(row['description'], get_field(row['type']), nullable=False)

struct_fields = attrs.rdd.map(to_struct).collect()

schema=StructType(struct_fields)


# ## Load the Training Data
# We'll also create a function `load_data(file: str)` which can be used to load the test data later.

# In[4]:


def load_data(file: str):
    return spark.read.load(file, format="csv", sep=",", header=False, ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True, schema=schema)

df = load_data(SOURCE_DATASET)
df.show(5)
df.printSchema()
df.count()


# # Clean Training Data
# As it is, the training data is not ready for planned analysis:

# ## Convert Label Column
# Our categorisation model will aim to predict whether an individual earns more than $50k per year. Since there are only two categories, we can express this as a binary integer or boolean type.
# 
# The function `convert_label_col(df: DataFrame)` will be created and used again later.

# In[5]:


import pyspark.sql.functions as f

def convert_label_col(df: DataFrame):
    df=df.withColumnRenamed('label', LABEL_COL)
    df=df.withColumn(LABEL_COL, f.when(f.col(LABEL_COL) == '>50K', 'True').otherwise('False'))
    return df.withColumn(LABEL_COL, df[LABEL_COL].cast(BooleanType()))

df = convert_label_col(df)

df.select(LABEL_COL).show(10)


# ## Handle Missing Data
# Missing values have been replaced with a '?' character. These values may make our model less accurate.
# 
# 

# In[6]:


def count_rows_with_missing_data(df: DataFrame):
    return df.select([f.sum(f.when(f.col(c) == '?', 1).otherwise(0)).alias(c) for c in df.columns])

count_rows_with_missing_data(df).show()


# There are a few ways to deal with them:
# * Remove entries with missing values (so long as this is not a significant number)
# * Impute missing values (if continuous)
# ```python
# NUMERIC_COLUMNS=['capital-gain','capital-loss']
# imputer = Imputer(
#     inputCols=NUMERIC_COLUMNS,
#     outputCols=["{}_imputed".format(c) for c in NUMERIC_COLUMNS]
# )
# df=imputer.fit(df).transform(df)
# ```
# * Assign them a string value (in this case '?') and include them in the model
# 
# Note that none of our numeric columns contain missing data:

# In[7]:


#create a list of the columns that are string typed
categoricalColumns = [item[0] for item in df.dtypes if item[1].startswith('string')]
numericColumns = [item[0] for item in df.dtypes if item[1].startswith('decimal')]

print(f'Numeric: {numericColumns}')
print(f'Categorical: {categoricalColumns}')

count_rows_with_missing_data(df).select(numericColumns).show()


# Therefore we do not need to impute these values. 
# 
# Filtering out all the missing values, we can see the impact this has on the dataset:

# In[8]:


def remove_rows_with_missing_categorical_data(df: DataFrame):
    categoricalColumns = [item[0] for item in df.dtypes if item[1].startswith('string')]
    for col in categoricalColumns:
        df=df.filter(df[col]!='?')
    return df

total_rows=df.count()
rows_with_missing_data=total_rows-remove_rows_with_missing_categorical_data(df).count()
percentage=rows_with_missing_data*100/total_rows

print(percentage)


# Removing these entries should be benificial for our model but it is probably worth keeping them for other forms of analysis.
# 
# We shall export the data for further analysis and clean the empty values when training our model:

# In[9]:




test_data=load_data(SOURCE_TEST_DATA)
# test_data=remove_superfluous_cols(test_data)
test_data=convert_label_col(test_data)

# Combine prepared test data and training data for use with Tableau later
df_combined = df.union(test_data)

df_combined.repartition(1).write.csv(COMBINED_DATA_OUT, header=True, sep=',', mode='overwrite')


# # Exploring Data Characteristics
# When representing the population, we must multiply any counts by fnlwgt to get the true proportion.

# In[10]:


print('Sex:')
df_combined.groupby('sex').count().show()

print('Education:')
df_combined.groupby('education').count().show()

print('Education_num:')
df_combined.groupby('education-num').count().sort('education-num', ascending=False).show()


# In[11]:



distinct_countries_count=df.select(['native-country']).distinct().count()
print(f'Number of distinct countries: {distinct_countries_count}')
df_combined.groupby('native-country').count().show(distinct_countries_count)


# # Preparing the Model
# We are going to train a Logistic Regression modal to classify the data. In order to do that, each row must be converted into numeric vector data.

# ## Irrelevant Data
# fnlwgt (Final Weight) does not relate to the individual entry; rather it describes the proportion of the population that this entry represents.
# 
# It has no use in our analysis and can be removed.
# 
# We can also remove 'education' as this is already indexed in the column 'education-num'.
# 
# The function `remove_superfluous_cols(df)` will be reused when preparing our test data.

# In[12]:


def remove_superfluous_cols(df: DataFrame):
    df=df.drop('fnlwgt')
    return df.drop('education')

df_clean = remove_superfluous_cols(df)

df_clean.show(5)


# ## Remove unknown values
# As previously discussed, we will remove the rows with unknown text values which cannot be imputed.

# In[13]:


df_clean=remove_rows_with_missing_categorical_data(df_clean)

count_rows_with_missing_data(df_clean).show()


# ## Create Feature Vector Data
# Now we can convert all string values into numeric index values using a string indexer:

# In[14]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

indexers = [StringIndexer(inputCol=column, outputCol=column+"_indexed").fit(df) for column in categoricalColumns ]
pipeline = Pipeline(stages=indexers)


def index_strings(df: DataFrame):
    dfi=pipeline.fit(df).transform(df)
    return dfi.drop(*categoricalColumns).withColumn(LABEL_COL, df[LABEL_COL].cast(IntegerType()))

df_indexed=index_strings(df_clean)
df_indexed.show(5)


# ## Check for Redundant Features
# 
# Now that we indexed all our string values, we can create a correlation matrix between each feature.
# 
# If two features have a high correlation, we can view one or more of these as redundant and remove it from our machine learning model.

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation


corr_graph=df_indexed.drop(LABEL_COL)

columns = corr_graph.columns
print(columns)
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=corr_graph.columns, 
                            outputCol=vector_col)
corr_graph_vector = assembler.transform(corr_graph).select(vector_col)
matrix = Correlation.corr(corr_graph_vector, vector_col)

matrix = Correlation.corr(corr_graph_vector, vector_col).collect()[0][0]
corrmatrix = matrix.toArray().tolist()
# print(corrmatrix)

df = spark.createDataFrame(corrmatrix,columns)

def plot_corr_matrix(correlations,attr,fig_no):
    fig=plt.figure(figsize=(15,15))
    ax=fig.add_subplot(111)
    ax.set_title("Correlation Matrix")
    ax.set_xticks(np.arange(len(attr)))
    ax.set_yticks(np.arange(len(attr)))
    ax.set_xticklabels(attr, rotation=90, fontsize=10)
    ax.set_yticklabels(attr, fontsize=10)
    cax=ax.imshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    plt.show()

plot_corr_matrix(corrmatrix, columns, 234)


# ## Create Feature Vectors

# In[16]:


from pyspark.ml.feature import VectorAssembler, Normalizer
VECTOR_COL='features'

features=list(df_indexed.drop(LABEL_COL).toPandas().columns)

assembler=VectorAssembler(inputCols=features, outputCol=VECTOR_COL)

def create_feature_vector(df):
    df_vector = assembler.transform(df)
    return df_vector.select(*[VECTOR_COL, LABEL_COL])

df_vector = create_feature_vector(df_indexed)
df_vector.show(5)


# ## Normalise Vectors
# 
# We to normalize the feature vectors using $L^1$ norm 

# In[17]:


normalizer = Normalizer().setInputCol(VECTOR_COL).setOutputCol(f'{VECTOR_COL}_n').setP(1.0)


def normalize_vector(df_vector):
    df_vector = normalizer.transform(df_vector)
    return df_vector.drop(VECTOR_COL).withColumnRenamed(f'{VECTOR_COL}_n',VECTOR_COL)

df_vector = normalize_vector(df_vector)
df_vector.show(5)


# # Classification
# 
# Logical Regression adapted from example in Spark documentation: https://spark.apache.org/docs/3.0.1/ml-pipeline.html
# 
# ## Configure Logistical Regression Instance
# 

# In[18]:


from pyspark.ml.classification import LogisticRegression

training_data=df_vector

# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=10, regParam=0.01, featuresCol=VECTOR_COL, labelCol=LABEL_COL)
# Print out the parameters, documentation, and any default values.
print(f"LogisticRegression parameters:")
lr.explainParams()


# ## Train the Model
# Train the model using the training data we prepared earlier

# In[19]:


training_data=df_vector
# Learn a LogisticRegression models. This uses the parameters stored in lr.
model1 = lr.fit(training_data)
print("Model 1 was fit using parameters: ")
model1.extractParamMap()


# ## Prepare the Test Data
# In a seperate file, additional values, with their classifications, is stored.
# 
# We will load this data and prepare it as we did our training data:

# In[20]:


test_data=load_data(SOURCE_TEST_DATA)
test_data=remove_superfluous_cols(test_data)
test_data=convert_label_col(test_data)
test_data=index_strings(test_data)
test_data=create_feature_vector(test_data)
test_data=normalize_vector(test_data)

test_data.show(5)


# ## Make Predictions
# Finally we can can put our model to the test.

# In[21]:


model=model1
# Make predictions on test data using the Transformer.transform() method.
# LogisticRegression.transform will only use the 'features' column.
prediction = model.transform(test_data)

prediction.select(['>50K', 'prediction' ]).show()


# ## Evaluate the Model
# 
# Finally, we can evaluate the success of our model with a classification evaluator:

# In[22]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName='accuracy', labelCol=LABEL_COL)
# compute the classification error on test data.
accuracy = evaluator.evaluate(prediction)

if not evaluator.isLargerBetter():
    accuracy = 1 - accuracy
    
print("Test Accuracy = %g" % accuracy)


# In[ ]:




