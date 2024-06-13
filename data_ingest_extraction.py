# Databricks notebook source
# MAGIC %md
# MAGIC # Online Transactional Processing (OLTP)
# MAGIC OLTP is the traditional relational database, like Azure SQL or Oracle. They are excellent at handling small transactions normally called CRUD - created, read, update, and delete. There are two primary flavors. SQL and NoSQL. 
# MAGIC
# MAGIC # Online Analytical Processing (OLAP)
# MAGIC These are data systems that tend to be either Data Lakes or Data Warehouses designed to handle large amounts of data. Data Warehouses tend to only hold structured data, where Data Lakes can hold almost any data type, including media like videos and audio. 
# MAGIC
# MAGIC # Data Lakes
# MAGIC One of the advantages of Data Lakes is that the storage and the compute are separated. In a system like Oracle, the storage and the compute are combined, meaning that that data must be brought to the engine, whereas with a data lake, you bring the right engine to the data. 
# MAGIC
# MAGIC # Common Data Platform Structure
# MAGIC Today's common data platform tends to be made up of the following parts:
# MAGIC
# MAGIC  - Data Governance Layer
# MAGIC  - Ingestion Layer
# MAGIC  - Analytics Layer
# MAGIC  - Consumption Layer
# MAGIC  - Serving Layer
# MAGIC  - Sentic View
# MAGIC  - Processing Layer
# MAGIC  - Storage Layer
# MAGIC
# MAGIC # Lambda and Kappa Architectures
# MAGIC
# MAGIC Lambda tends to have both a batch and streaming ingestion layer, typically with the ability to serve real-time streaming data. Kappa tends to only have streaming data, and will "batch" up data for systems that need to be served batch data at the end of the pipelines process.
# MAGIC
# MAGIC # Lakehouse Architecture
# MAGIC
# MAGIC There are seven central tenets to the lakehouse architecture:
# MAGIC
# MAGIC 1. Openness
# MAGIC 2. Data diversity
# MAGIC 3. Workflow diversity
# MAGIC 4. Processing diversity
# MAGIC 5. Language agnostic
# MAGIC 6. Decoupling storage and compute
# MAGIC 7. ACID transactions
# MAGIC
# MAGIC # Delta Architecture
# MAGIC
# MAGIC This is very similar to Kappa, except you aren't forced to stream data, you can run batch and stream from the same processing layer. 
# MAGIC
# MAGIC # Medallion data pattern
# MAGIC
# MAGIC  - Bronze is raw data most like its source. Data is only added here, not deleted in order to always have the option to return to it for additional processing.
# MAGIC  - Silver is clean, molded, transformed, and modelled data. This is the stage before creating data products.
# MAGIC  - Gold are the final data products. This is what is meant to be consumed. 
# MAGIC
# MAGIC # Data Mesh in Theory and Practice
# MAGIC
# MAGIC A Data Mesh is not a mix of Data Silos, but a method for brining data together aligned with Domains. It's DDD for Data. It follows four principles.
# MAGIC
# MAGIC 1. Data Ownership - If you produce the data you own it and are responsible for it. 
# MAGIC 2. Data as a Product - The team who owns the data treats the data like a product and offers it and services to consuming teams. 
# MAGIC 3. Data is Available - like microservices, data should be discoverable and accessible to the people who need it. 
# MAGIC 4. Data is Governed - Data policies are centrally defined and distributed to locally.
# MAGIC

# COMMAND ----------

from functools import partial
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, flatten, explode, struct, create_map, array
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DateType, TimestampType

# COMMAND ----------

spark = SparkSession.builder.appName("chap_2").master("local[*]").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleansing
# MAGIC
# MAGIC In this section we want to clean and prepare data. These represent several fundamental methods for cleansing and wrangling data.

# COMMAND ----------

data_frame = spark.createDataFrame(data = [("Brian", "Enger", 1),
                                           ("Nechama", "Engr", 2),
                                           ("Naava", "Engr", 3),
                                           ("Miri", "Engr", 4),
                                           ("Brian", "Enger", 1),
                                           ("Miri", "Engr", 3), ], schema=["name", "div", "ID"])

# COMMAND ----------

data_frame.distinct().show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC An alternative to the above is to use the `dropDuplicates`

# COMMAND ----------

data_frame.dropDuplicates().show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also look at a subset of a row using list of columns

# COMMAND ----------

data_frame.dropDuplicates(['ID']).show(truncate=False)

# COMMAND ----------

data_frame_2 = spark.createDataFrame(data = [("Brian," "Engr", 1),
                                             ("Nechama", "", None),
                                             ("Naava", "Engr", 3),
                                             ("Miri", "", 4),
                                             ("Brian", "Engr", None),
                                             ("Miri", "", 3),], schema = ["name", "div", "ID"])
data_frame = spark.createDataFrame(data = [("Brian", "Engr", 1),
                                           ("Nechama", "Engr", None),
                                           ("Naava", "Engr", 3),
                                           ("Miri", "Engr", 4),
                                           ("Brian", "Engr", None),
                                           ("Miri", "Engr", 3),], schema = ["name", "div", "ID"])

# COMMAND ----------

# MAGIC %md
# MAGIC More ways to limit nulls

# COMMAND ----------

data_frame.filter(col("ID").isNull()).show()

# COMMAND ----------

data_frame.filter(col("ID").isNotNull()).show()

# COMMAND ----------

# MAGIC %md
# MAGIC In this example, we are checking that both the ID and the name columns are not null:

# COMMAND ----------

data_frame.filter(col("ID").isNotNull() &  (col("name").isNotNull())).show()

# COMMAND ----------

# MAGIC %md
# MAGIC We also have the option of using the select method and creating a null label column. 
# MAGIC Using a label can be beneficial at times. One example could be to use that column as a feature for a machine learning model:

# COMMAND ----------

data_frame.select(col("*"), col("ID").isNull().alias("null")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Here is an example of using the select method to build a cleaned version of a column that incorrectly defined nulls as empty strings. It is very common to find cases where nulls are set to alternative values:
# MAGIC
# MAGIC ***This doesn't look like it's working***

# COMMAND ----------

# Assuming data_frame_2 is a DataFrame already defined
from pyspark.sql.functions import when, col

data_frame_2 = data_frame_2.withColumn(
    "cleaned_div", 
    when(col("div") == "", None).otherwise(col("div"))
)

data_frame_2.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use RegEx

# COMMAND ----------

data_frame.select(col("*"), regexp_replace('ID', '1', '10').alias("fixed_ID")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we are replacing all values in the name column that start with Mi to start with mi instead:

# COMMAND ----------

data_frame.select(col("*"), regexp_replace('name', '^Mi', 'mi').alias("cleaned_name")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outlier identification
# MAGIC
# MAGIC It can be common to look for outliers in columns, and this is often used in analysis and data science workflows.

# COMMAND ----------

data_frame_3 = spark.createDataFrame(data = [("Brian", "Engr", 1),("Nechama", "Engr", 1),("Naava", "Engr", 3),("Miri", "Engr", 5),("Brian", "Engr", 7),("Miri", "Engr", 9),], schema = ["name", "div", "ID"])

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we are using * to unpack a list and pass each element into the select method individually. This can be a very useful technique to use. We are also using the binary not operator to reverse the range identification and pass it to the when method:

# COMMAND ----------

data_frame_3.select(col("name"),
        *[
            when(
                ~col("ID").between(3, 10),
                "yes"
            ).otherwise("no").alias('outlier')
        ]
    ).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Casting columns
# MAGIC
# MAGIC Data columns can sometimes be set to the wrong data type when dealing with raw tables. This type of error ideally should be fixed as far upstream as possible.

# COMMAND ----------

data_frame_4 = spark.createDataFrame(data = [("Brian", "Engr", "1"),
                                             ("Nechama", "Engr", "1"),
                                             ("Naava", "Engr", "3"),
                                             ("Miri", "Engr", "5"),
                                             ("Brian", "Engr", "7"),
                                             ("Miri", "Engr", "9"),
                                             ], schema = ["name", "div", "ID"])

# COMMAND ----------

# MAGIC %md
# MAGIC We will first look at the schema of the DataFrame to see what the data types are. We will notice that the ID column is string, not int:

# COMMAND ----------

data_frame_4.schema
StructType([StructField('name', StringType(), True), StructField('div', StringType(), True), StructField('ID', StringType(), True)])

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we cast the ID column from string to int:

# COMMAND ----------

data_frame_4.select(col("*"),col("ID").cast('int').alias("cleaned_ID")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fixing column names
# MAGIC It’s very common to have columns come from data sources with odd naming conventions. When passing data between teams, one team often has norms that won’t match another team’s coding practices.
# MAGIC
# MAGIC Here we are using the alias method on the column object to rename the name column:

# COMMAND ----------

data_frame_4.select(col("ID"),col("name").alias("user_name")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complex data types
# MAGIC We are now going to set up DataFrames for these examples. Keep in mind that Spark always has a strongly enforced schema on all columns, so every row has the same schema, no matter whether the dataframe is structured or semi-structured when working with DataFrames:

# COMMAND ----------

schema_5 = StructType([
        StructField('user', StructType([
             StructField('name', StringType(), True),
             StructField('age', IntegerType(), True),
             StructField('id', IntegerType(), True)
             ])),
         StructField('codes', ArrayType(StringType()), True),
         StructField('location_id', IntegerType(), True),
         ])
data_5 =  [(("Bruce Lee", 21, 1), [5,6,7,8],9)]
data_frame_5 = spark.createDataFrame(data=data_5,schema=schema_5)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will select the name element of the user array:

# COMMAND ----------

data_frame_5.select("user.name").show()

# COMMAND ----------

# MAGIC %md
# MAGIC In the codes column, we have an array of values. We can access them manually using the bracket notation:

# COMMAND ----------

data_frame_5.select(col("codes")[0]).show()

# COMMAND ----------

data_frame_6 = spark.createDataFrame([([[9, 7], [56, 12], [23,43]],), ([ [400, 500]],)],["random_stuff"])
data_frame_6.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Another very useful function is the explode function. It will take a column with an array and transform the whole DataFrame by extracting the array into several unique rows of the DataFrame:

# COMMAND ----------

data_frame_5.select(col("location_id"), explode("codes").alias("new")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Here is another example of using the explode function, this time on a hash structure:

# COMMAND ----------

data_frame_7 = spark.createDataFrame([ ({"apple": "red"}, 1),  ({"orange": "orange"}, 2)], ["fruit", "id"])
data_frame_7.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that now we have key and value columns:

# COMMAND ----------

data_frame_7.select(col("id"), explode("fruit")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Lastly, we will create a complex semi-structured DataFrame. We are using the create_map and array functions to create this structure:

# COMMAND ----------

test = data_frame_4.select(col("id"), create_map("id", struct(["name","div"])).alias("complex_map"), array(["name","div"]).alias("complex_array"))
test.show(truncate=False)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, MapType, ArrayType

test.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## diagrams
# MAGIC The diagrams library is very useful for creating solution diagrams. A solution diagram is often a broad picture of the architecture and key components. It can be organized in a way that explains key interactions.
# MAGIC
# MAGIC Here, we are creating a small example document using the diagrams package:

# COMMAND ----------

# MAGIC %sh apt-get update; apt-get -f -y install graphviz

# COMMAND ----------

# MAGIC %pip install diagrams

# COMMAND ----------

from diagrams import Cluster, Diagram
from diagrams.aws.analytics import Quicksight, EMR
with Diagram("Data Platform", show=False):
    with Cluster("Dev"):
        dashboards = Quicksight("Tableau")
        spark_clusters = [EMR("Notebook_cluster"), EMR("Jobs_cluster")]
        dashboards >> spark_clusters

# COMMAND ----------

from diagrams import Cluster, Diagram
from diagrams import Diagram, Cluster
from diagrams.custom import Custom
with Diagram("Dataplatform", show=False, filename="dataplatform_custom"):
    with Cluster("Prod - VPC"):
        compute = [Custom("Tiny", "./db_cluster.png") , Custom("Med", "./db_cluster.png")]
        dashboards = Custom("Tiny", "./tabl.png")
        dashboards << compute
