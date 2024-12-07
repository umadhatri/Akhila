import streamlit as st
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Spotify API credentials
SPOTIFY_CLIENT_ID = "3101c074ca6247d49de867dd0180d867"
SPOTIFY_CLIENT_SECRET = "1e6570f7890c4df788f00ade23fc3d75"

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# Function to get tracks by an artist
def get_tracks_by_artist(artist_name):
    results = sp.search(q=artist_name, type="track", limit=50)
    tracks = []
    for item in results['tracks']['items']:
        track = {
            "name": item["name"],
            "id": item["id"],
            "artist": item["artists"][0]["name"]
        }
        tracks.append(track)
    return tracks

# Function to get audio features of tracks
def get_audio_features(track_ids):
    features = sp.audio_features(track_ids)
    return pd.DataFrame(features)

# Recommendation function
def recommend_songs(seed_track_id, n_recommendations=5):
    seed_features = sp.audio_features([seed_track_id])[0]
    seed_df = pd.DataFrame([seed_features])

    results = sp.recommendations(seed_tracks=[seed_track_id], limit=50)
    track_ids = [track["id"] for track in results["tracks"]]
    related_features = get_audio_features(track_ids)

    feature_columns = ['danceability', 'energy', 'tempo', 'valence']
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(related_features[feature_columns])
    related_features[feature_columns] = normalized

    seed_vector = seed_df[feature_columns].values
    similarities = cosine_similarity(seed_vector, normalized)

    related_features["similarity"] = similarities[0]
    related_features["track_name"] = [track["name"] for track in results["tracks"]]

    recommendations = related_features.sort_values("similarity", ascending=False).head(n_recommendations)
    return recommendations[["track_name", "similarity"]]

# Streamlit app layout
st.title("Spotify Song Recommender")
st.write("This app recommends songs based on a seed track or artist.")

# Input options
option = st.radio("Choose an option to get started:", ["Search by Artist", "Search by Track ID"])

if option == "Search by Artist":
    artist_name = st.text_input("Enter Artist Name:")
    if artist_name:
        tracks = get_tracks_by_artist(artist_name)
        if tracks:
            st.write(f"Top tracks by {artist_name}:")
            for idx, track in enumerate(tracks[:5]):
                st.write(f"{idx + 1}. {track['name']} (ID: {track['id']})")
        else:
            st.write("No tracks found for this artist. Please try another name.")

elif option == "Search by Track ID":
    track_id = st.text_input("Enter Track ID:")
    if track_id:
        try:
            recommendations = recommend_songs(track_id)
            st.write("Recommended Songs:")
            st.table(recommendations)
        except Exception as e:
            st.write("An error occurred. Please check the Track ID and try again.")

# Footer
st.write("Powered by Spotify API and Streamlit")
