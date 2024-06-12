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
data_frame_2 = data_frame_2.withColumn(
    "cleaned_div", 
    when(col("div") == "", None).otherwise(col("div"))
)

display(data_frame_2)

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
