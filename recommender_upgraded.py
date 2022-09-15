import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from scipy.spatial import distance
import streamlit as st
from pyspark.sql import functions as F
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS


data = pd.read_csv('data.csv')
data['artists'] = data['artists'].apply(lambda x: x.strip('][').split(', '))
data['artists'] = data['artists'].apply(lambda x: ', '.join([i.strip("'\"") for i in x]))
li = ['remix','Remix','feat.','Feat.']
for w in li:
    data['name'] = data['name'].apply(lambda x: x.split(w)[0])

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


df = spark.createDataFrame(data,schema=schema)
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
def get_distance(y,num,artist):
    ''' get recommendations'''
    y_line = output.filter((output.name==y) & (output.artists==artist))
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
    return same_cluster[['artists','name','year','popularity']].head(num)

# recommendation = get_distance('Singende Bataillone 1. Teil',5)
# print(recommendation)

def cloud(input_val):
    ''' word cloud'''
    st.set_option('deprecation.showPyplotGlobalUse', False)

    comment_words = ''
    stopwords = set(STOPWORDS)
    if type(input_val) is str:
        col = 'artists'
    else:
        col = 'year'
    for val in df.toPandas()[df.toPandas()[col] == input_val]['name']:
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    st.pyplot()

def main():
    st.title("Spotify Songs Recommender")
    st.write("This is a song recommender based on the Spotify dataset.")
    st.write("The dataset contains 170,000 songs with 18 features.")
    st.write("The features are: acousticness, artists, danceability, duration_ms, energy, explicit, id, instrumentalness, key, liveness, loudness, mode, name, popularity, release_date, speechiness, tempo, valence, year")
    st.write("The recommender uses K-Means clustering to find similar songs.")
    st.write("The user can select a song and the recommender will find the 10 most similar songs.")

    image = Image.open('open-graph-default.png')
    st.image(image, caption='Spotify')

    st.write("Select the year and create a word cloud of the songs in that year.")

    year_list = sorted(df.toPandas()['year'].unique(),reverse=True)
    selected_year = st.selectbox( "Type or select a year from the dropdown", year_list)
    
    cloud(selected_year)

    artitst_list = sorted(df.toPandas()['artists'].unique())
    selected_artist = st.selectbox( "Type or select an artist from the dropdown", artitst_list)
    cloud(selected_artist)

    menu = ["Home", "Choose Your Song By Singer","Choose Your Song By Song Name","Interaction with the dataset"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.write("This is the home page.")
    elif choice == "Choose Your Song By Singer":
        st.subheader("Recommendation")
        st.write("Please enter the singer name and choose song name.")
        artitst_list = df.toPandas()['artists'].unique()
        selected_singer = st.selectbox( "Type or select an artist from the dropdown", artitst_list)
        song_list = df.filter(df.artists.contains(selected_singer)).toPandas()['name'].values
        selected_song = st.selectbox( "Type or select a song from the dropdown", song_list)
        num_of_songs = st.slider("Number of songs to recommend", 1, 30, 5)
        if st.button("Recommend"):
            recommendation = get_distance(selected_song,num_of_songs,selected_singer)
            st.write(recommendation)
    elif choice == "Choose Your Song By Song Name":
        st.subheader("Recommendation")
        st.write("This is the recommendation page.")
        song_list = df.toPandas()['name'].unique()
        selected_song = st.selectbox( "Type or select a song from the dropdown", song_list)
        artitst_list = df.filter(df.name==selected_song).toPandas()['artists'].values
        selected_artist = st.selectbox( "Type or select an artist from the dropdown", artitst_list)
        num_of_songs = st.slider("Number of songs to recommend", 1, 30, 5)
        if st.button('Show Recommendation'):
            if selected_song is not None and selected_artist is not None:
                recommended_song_names = get_distance(selected_song,num_of_songs,selected_artist)
                st.dataframe(recommended_song_names)

    elif choice == "Interaction with the dataset":
        st.subheader("Here are some interactions with the dataset")
        # popularity of an artist over the years
        st.write("Popularity of an artist over the years")
        artitst_list = df.toPandas()['artists'].unique()
        selected_singer = st.selectbox( "Type or select an artist from the dropdown", artitst_list)
        # df.filter(df.location.contains('google.com'))
        artist_data = df.filter(df.artists.contains(selected_singer)).toPandas().groupby("year").mean()
        fig = go.Figure([go.Scatter(x=artist_data.index, y=artist_data["popularity"])],layout_title_text="Popularity of "+selected_singer+" over the years")
        st.plotly_chart(fig, use_container_width=True)

        st.write("Top artists ranking for a year")
        # Artist ranking for a year
        year_list = sorted(df.toPandas()['year'].unique(),reverse=True)
        selected_year = st.selectbox( "Type or select a year from the dropdown", year_list,key = 'Artist ranking for a year')

        num_of_artists = st.slider("Number of artists to show", 1, 30, 15)

        artist_ranking_year = df.filter(df.year == str(selected_year)).toPandas()
        artist_ranking_year = artist_ranking_year.groupby("artists").mean().sort_values(["popularity"],ascending=False).head(num_of_artists)
        fig = go.Figure(
            data=[go.Bar(x=artist_ranking_year.index,y=artist_ranking_year["popularity"])],
            layout_title_text="Artist ranking for the year - "+str(selected_year))
        st.plotly_chart(fig, use_container_width=True)

        # Popular artist ranking along with the number of hit songs released over the years

        num_of_artists = st.slider("Number of artists to show", 1, 50, 25,key = 'Popular artist ranking along with the number of hit songs released over the years')
        st.write("Artist ranking along with the number of hit songs released over the years")
        artist_ranking_hit = df.toPandas().groupby("artists").mean().sort_values(["popularity"],ascending=False).head(num_of_artists)
        artist_ranking_hit["artists"] = artist_ranking_hit.index.values
        artist_ranking_hit["count"] = df.toPandas().groupby("artists").count()["popularity"].head(num_of_artists).values

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Bar(x=artist_ranking_hit["artists"],
                        y=artist_ranking_hit["popularity"],
                        text = artist_ranking_hit["count"],
                        hovertemplate = '<b>Artist: </b>%{x}<br><b>Popularity: </b>%{y:.2f}<br><b> # Songs: </b>%{text}',
                        showlegend = False
                        ),
            secondary_y=False,
        )
        fig.add_trace(go.Scatter(
            x = artist_ranking_hit["artists"],
            y = artist_ranking_hit["count"],
            text = artist_ranking_hit["popularity"],
            hovertemplate = '<b>Artist: </b>%{x}<br><b>Popularity: </b>%{text:.2f}<br><b> # Songs: </b>%{y}',
            showlegend = False),
            secondary_y=True,)

        # Set x-axis title
        fig.update_xaxes(title_text="Artist ranking along with the number of songs released over the years")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Popularity</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b># Hit songs", secondary_y=True)

        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
    