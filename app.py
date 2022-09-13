import streamlit as st
import streamlit.components.v1 as stc

def main():
    st.title("Spotify Song Recommender")
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
        stc.html(
            """
            <iframe src="http://localhost:8501" width="100%" height="1000px"></iframe>
            """,
            height=1000,
        )
    elif choice == "About":
        st.subheader("About")
        st.write("This is the about page.")
        st.text("Built with Streamlit and Pyspark")

if __name__ == '__main__':
    main()
    