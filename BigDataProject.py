# Databricks notebook source
# MAGIC %md
# MAGIC **1-Azure Storage was linked to Databricks using PySpark to facilitate direct data access and analysis**

# COMMAND ----------

# DBTITLE 1,Connect to Azure Storage
storage_account_name = "bigdataprojectalireza"
storage_account_key = "sHFTpkN7tYoUBmZTgkQrhobI0ok+JQsWDR2+K/75UMYpC+ScFbZ9AEv8u+oiAV1BLuU5YFTI52wM+ASt/DJK1g=="
container_name = "datasets"  
mount_point = "/mnt/datasets"

dbutils.fs.mount(
  source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
  mount_point = mount_point,
  extra_configs = {
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
  }
)

# COMMAND ----------

# MAGIC %md
# MAGIC **2-Preprocessing and visualization**

# COMMAND ----------

display(dbutils.fs.ls("/mnt/datasets"))

# COMMAND ----------

# DBTITLE 1,Load the Data
crime_df = spark.read.csv("/mnt/datasets/crime-rate-by-country-2023.csv", header=True, inferSchema=True)
crime_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC **3-Data Cleaning:
# MAGIC 1-Handle Missing Values 2-Fix Data Types 3-Remove Duplicates 4-Standardize Column Names 5-Outlier Detection**

# COMMAND ----------

# DBTITLE 1,Data Cleaning

from pyspark.sql.functions import col, isnan, when, count

crime_rate_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in crime_rate_df.columns]).show()

edu_income_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in edu_income_df.columns]).show()

crime_rate_df = crime_rate_df.dropna(subset=['crimeIndex', 'pop2023'])
edu_income_df = edu_income_df.dropna(subset=['education_index'])  


crime_rate_df = crime_rate_df.withColumnRenamed('Crime Index', 'crimeIndex')
edu_income_df = edu_income_df.withColumnRenamed('Education Index', 'education_index')


crime_rate_df = crime_rate_df.dropDuplicates()
edu_income_df = edu_income_df.dropDuplicates()


crime_rate_df = crime_rate_df.withColumn('crimeIndex', col('crimeIndex').cast('double'))
crime_rate_df = crime_rate_df.withColumn('pop2023', col('pop2023').cast('double'))
edu_income_df = edu_income_df.withColumn('education_index', col('education_index').cast('double'))

crime_rate_df = crime_rate_df.filter((col('crimeIndex') >= 0) & (col('crimeIndex') <= 100))


crime_rate_df.show(5)
edu_income_df.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC **4-Merge the crime and education datasets for deeper correlation analysis**

# COMMAND ----------

# DBTITLE 1,Merge Crime and Education Datasets

merged_df = pd.merge(crime_rate_pd, edu_income_pd, on='country', how='inner')


print(merged_df.info())
print(merged_df.head())

print(merged_df.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC **_5-First basic visualisation after merging datasets_**

# COMMAND ----------

# DBTITLE 1,Top 10 Populated Countries with Highest Crime (Bar Chart)

top_10_pop = merged_df.sort_values(by='pop2023', ascending=False).head(10)


plt.figure(figsize=(12, 8))
sns.barplot(
    x='crimeIndex', 
    y='country', 
    data=top_10_pop, 
    palette='Reds_r'
)
plt.title('Top 10 Populated Countries with Highest Crime Index (2023)', fontsize=18)
plt.xlabel('Crime Index', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.grid(axis='x', linestyle='--', linewidth=0.5)


for index, value in enumerate(top_10_pop['crimeIndex']):
    plt.text(value - 5, index, f'{value:.2f}', color='white', fontsize=12)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **6-Exploratory Data Analysis for example : Europe and Asia**

# COMMAND ----------

# DBTITLE 1,Filter and Visualize Europe vs. Asia

region_mapping = {
    'Germany': 'Europe', 'France': 'Europe', 'Italy': 'Europe',
    'Japan': 'Asia', 'India': 'Asia', 'China': 'Asia',
    'Brazil': 'Americas', 'Argentina': 'Americas',
    'South Africa': 'Africa', 'Nigeria': 'Africa'
}


merged_df['region'] = merged_df['country'].map(region_mapping)

europe_df = merged_df[merged_df['region'] == 'Europe']
asia_df = merged_df[merged_df['region'] == 'Asia']

print("Europe Data:\n", europe_df.head())
print("Asia Data:\n", asia_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC **_Scatter Analysis_**

# COMMAND ----------

# DBTITLE 1,Scatter Plot (Crime vs. Education by Region)
plt.figure(figsize=(14, 8))
sns.scatterplot(
    x='education_index', 
    y='crimeIndex', 
    hue='region', 
    data=merged_df[(merged_df['region'] == 'Europe') | (merged_df['region'] == 'Asia')], 
    s=100, 
    alpha=0.8, 
    palette='Dark2'
)
plt.title('Crime Index vs. Education Index (Europe vs. Asia)', fontsize=18)
plt.xlabel('Education Index', fontsize=14)
plt.ylabel('Crime Index', fontsize=14)
plt.legend(title='Region')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **7-Comparative Analysis – Income and Education**

# COMMAND ----------

# DBTITLE 1,Average Crime Index by Education Level (Bar Plot)
avg_crime_by_edu = merged_df.groupby('education_level')['crimeIndex'].mean().reset_index()

plt.figure(figsize=(12, 7))
sns.barplot(
    x='education_level', 
    y='crimeIndex', 
    data=avg_crime_by_edu, 
    palette='coolwarm'
)
plt.title('Average Crime Index by Education Level', fontsize=18)
plt.xlabel('Education Level', fontsize=14)
plt.ylabel('Average Crime Index', fontsize=14)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.show()

# COMMAND ----------

# DBTITLE 1,Crime Index by Income Group (Box Plot)
plt.figure(figsize=(12, 8))
sns.boxplot(
    x='income', 
    y='crimeIndex', 
    data=merged_df, 
    palette='muted'
)
plt.title('Crime Index by Income Group', fontsize=18)
plt.xlabel('Income Group', fontsize=14)
plt.ylabel('Crime Index', fontsize=14)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **8-Correlation Analysis**

# COMMAND ----------

# DBTITLE 1,Correlation Heatmap (Full Dataset)
plt.figure(figsize=(10, 7))
corr = merged_df[['crimeIndex', 'education_index', 'pop2023']].corr()
sns.heatmap(
    corr, 
    annot=True, 
    cmap='magma', 
    linewidths=0.5
)
plt.title('Correlation Heatmap (Crime, Education, Population)', fontsize=18)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **9-Population and Crime Trends**

# COMMAND ----------

# DBTITLE 1,Population vs. Crime Index (Scatter with Trend Line)
plt.figure(figsize=(12, 8))
sns.regplot(
    x='pop2023', 
    y='crimeIndex', 
    data=merged_df, 
    scatter_kws={'s': 70, 'alpha': 0.7}, 
    line_kws={'color': 'orange'}
)
plt.title('Crime Index vs. Population (2023)', fontsize=18)
plt.xlabel('Population (Log Scale)', fontsize=14)
plt.ylabel('Crime Index', fontsize=14)
plt.xscale('log')  # Log scale for better visualization
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
