"""
Spotify Recommendation Prediction
"""

from src.recommendation import recommend_song

def main():

    print("=" * 50)
    print("Spotify Recommendation System")
    print("=" * 50)

    song = input("Enter Song Name: ")

    recommend_song(song)

if __name__ == "__main__":
    main()
