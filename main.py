import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("songs_normalize.csv")

#Preprocessing steps
def preprocessing(df):
    #Remove duplicates 
    df = df.drop_duplicates(subset=['artist', 'song', 'year'], keep='first')

    #Remove songs with no genre 
    df = df[df['genre'] != 'set()']

    #Check for extremes 
    attributes = [
        "duration_ms", "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]

    #Find extremes for each attribute 
    extremes = []

    for attr in attributes:
        max_row = df.loc[df[attr].idxmax()]
        min_row = df.loc[df[attr].idxmin()]
        extremes.append({
            "attribute": attr,
            "type": "max",
            "value": max_row[attr],
            "artist": max_row["artist"],
            "song": max_row["song"]
        })
        extremes.append({
            "attribute": attr,
            "type": "min",
            "value": min_row[attr],
            "artist": min_row["artist"],
            "song": min_row["song"]
        })
    print("Extremes:")
    print(pd.DataFrame(extremes))

    #One-hot encode genres 
    df['genre'] = df['genre'].str.split(', ')
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(df['genre']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df, genres_encoded], axis=1)
    df = df.drop('genre', axis=1)

    #Convert explicit to int 
    df['explicit'] = df['explicit'].astype(int)

    #Convert duration to seconds 
    df['duration_sec'] = df['duration_ms'] / 1000
    df = df.drop('duration_ms', axis=1)

    #Drop artist and song 
    df = df.drop(['artist', 'song'], axis=1)

    #Correlation matrix of numeric features
    numeric_cols = ['duration_sec', 'popularity', 'danceability', 'energy', 'loudness', 
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                    'valence', 'tempo', 'explicit', 'key', 'mode', 'year']

    corr_matrix = df[numeric_cols].corr(method='pearson')
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

    #Normalize numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    #Save preprocessed data
    df.to_csv("songs_preprocessed.csv", index=False)

    return df

if __name__ == "__main__":
    preprocessing(df)
