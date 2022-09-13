import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from scipy.spatial import distance

import streamlit as st
import streamlit.components.v1 as stc

from pyspark.sql import functions as F

from ast import literal_eval



spark = SparkSession.builder.getOrCreate()

schema = schema = StructType([
    StructField("acousticness", FloatType(), True),
    StructField("artists", StringType(), True),
    StructField("danceability", FloatType(), True),
    StructField("duration_ms", IntegerType(), True),
    StructField("energy", FloatType(), True),
    StructField("explicit", IntegerType(), True),
    StructField("id", StringType(), True),
    StructField("instrumentalness", FloatType(), True),
    StructField("key", IntegerType(), True),
    StructField("liveness", FloatType(), True),
    StructField("loudness", FloatType(), True),
    StructField("mode", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("popularity", IntegerType(), True),
    StructField("release_date", StringType(), True),
    StructField("speechiness", FloatType(), True),
    StructField("tempo", FloatType(), True),
    StructField("valence", FloatType(), True),
    StructField("year", IntegerType(), True)
    ])
df = spark.read.csv("data.csv",header=True, inferSchema=True, schema=schema)
df = df.na.drop()
df = df.dropDuplicates(["artists","name"])

assemble=VectorAssembler(inputCols=[
 'acousticness',
 'danceability',
 'energy',
 'duration_ms',
 'instrumentalness',
 'valence',
 'tempo',
 'liveness',
 'loudness',
 'popularity',
 'speechiness',
 'year',
 'key'], outputCol='features')
assembled_data=assemble.transform(df)


scale=StandardScaler(inputCol='features',outputCol='standardized')
data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)

KMeans_algo=KMeans(featuresCol='standardized', k=7)
KMeans_fit=KMeans_algo.fit(data_scale_output)

output=KMeans_fit.transform(data_scale_output)

# @F.udf(StringType())
def get_distance(y,num):
    ''' get recommendations'''
    y_line = output.filter(output.name==y)
    y_loc = y_line.toPandas()['standardized'].values[0]
    same_cluster = output.filter(output.prediction==y_line.prediction)
    distance_udf = F.udf(lambda x: float(distance.euclidean(x, y_loc)), FloatType())
    same_cluster = same_cluster.withColumn('distances', distance_udf(F.col('standardized')))
    same_cluster = same_cluster.sort(same_cluster.distances.asc())
    same_cluster = same_cluster.toPandas()
    same_cluster = same_cluster[same_cluster.name != y]
    same_cluster['artists'] = same_cluster.artists.apply(lambda x: x.strip('][').split(', '))
    same_cluster['artists'] = same_cluster['artists'].apply(lambda x: ', '.join([i.strip("'") for i in x]))
    same_cluster = same_cluster.sort_values(by=['distances'])
    return same_cluster[['artists','name','year']].head(num)

# recommendation = get_distance('Singende Bataillone 1. Teil',5)
# print(recommendation)

def main():
    st.title("Spotify Songs Recommender")
    st.write("This is a song recommender based on the Spotify dataset.")
    st.write("The dataset contains 170,000 songs with 18 features.")
    st.write("The features are: acousticness, artists, danceability, duration_ms, energy, explicit, id, instrumentalness, key, liveness, loudness, mode, name, popularity, release_date, speechiness, tempo, valence, year")
    st.write("The recommender uses K-Means clustering to find similar songs.")
    st.write("The user can select a song and the recommender will find the 10 most similar songs.")

    menu = ["Home", "Recommendation","About"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.write("This is the home page.")
    elif choice == "Recommendation":
        st.subheader("Recommendation")
        st.write("This is the recommendation page.")
        song_list = df.toPandas()['name'].values
        selected_song = st.selectbox( "Type or select a song from the dropdown", song_list )
        num_of_songs = st.slider("Number of songs to recommend", 1, 10, 5)
        if st.button('Show Recommendation'):
            if selected_song is not None:
                recommended_song_names = get_distance(selected_song,num_of_songs)
                st.dataframe(recommended_song_names)

    elif choice == "About":
        st.subheader("About")
        st.write("This is the about page.")
        st.text("Built with Streamlit and Pyspark")

if __name__ == '__main__':
    main()
    