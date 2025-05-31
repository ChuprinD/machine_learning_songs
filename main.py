import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import statsmodels.api as sm

#Insights from data exploration
def exploration(df):
    print("DATA EXPLORATION")
    print("Data sample:")
    print(df[0:5])

    print("\nFeatures:")
    for c in df.columns:
        print("\t"+c)

    print("\nNumber of samples: "+str(df.shape[0]))

    #Find extremes for each attribute  
    attributes = [
        "duration_ms", "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]

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

    print(df["instrumentalness"].value_counts())

    #Distribution of features
    features = [
    "duration_ms", "year", "popularity", 
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", 
    "liveness","valence", "tempo"
    ]

    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()  

    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"Distribution of {feature}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    plt.tight_layout()
    plt.savefig("distribution_of_features.png")
    plt.show()

    #Correlation matrix of numeric features
    numeric_cols = ['duration_ms', 'popularity', 'danceability', 'energy', 'loudness', 
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                    'valence', 'tempo', 'explicit', 'key', 'mode', 'year']

    corr_matrix = df[numeric_cols].corr(method='pearson')
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Numeric Features')
    plt.savefig("correlation_matrix.png")
    plt.show()

    input("Data exploration complete.\nPress Enter to continue...")

#Preprocessing steps
def preprocessing_for_multiple_linear_regression(df, model):
    #Remove duplicates 
    df = df.drop_duplicates(subset=['artist', 'song', 'year'], keep='first')

    #Remove songs with no genre 
    df = df[df['genre'] != 'set()']

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

    #Drop instrumentalness
    df = df.drop('instrumentalness', axis=1)

    #Normalize numeric features
    numeric_cols = []
    if model == "MLR":
        numeric_cols = ['duration_sec', 'popularity', 'danceability', 'energy', 'loudness', 
                        'speechiness', 'acousticness', 'liveness', 'valence', 'tempo', 
                        'explicit', 'key', 'mode', 'year']
    elif model == "KNN":
        numeric_cols = ['duration_sec', 'danceability', 'energy', 'loudness', 
                        'speechiness', 'acousticness', 'liveness', 'valence', 'tempo', 
                        'explicit', 'key', 'mode', 'year']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    #Save preprocessed data
    df.to_csv("songs_preprocessed.csv", index=False)

    print("Data after preprocessing:")

    print("Data sample:")
    print(df[0:5])

    print("\nFeatures:")
    for c in df.columns:
        print("\t"+c)

    print("\nNumber of samples: "+str(df.shape[0]))

    input("Preprocessing complete.\nPress Enter to continue...")

    return df

#Multiple linear regression
def perform_multiple_linear_regression(df):
    X = df.drop(columns=["popularity"])
    Y = df["popularity"]

    # Add constant column for intercept
    X_const = sm.add_constant(X)

    # Divide data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X_const, Y, test_size=0.2, random_state=42)

    # Train model
    model = sm.OLS(Y_train, X_train).fit()

    # Predict
    Y_pred = model.predict(X_test)

    # R^2 on test set
    r2_test = sm.OLS(Y_test, sm.add_constant(X_test.drop(columns=['const']))).fit().rsquared

    print("R^2 on test set:", r2_test)
    print(model.summary())

    # Plot residuals
    residuals = Y_test - Y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(Y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted Values (Multiple Linear Regression)")
    plt.xlabel("Predicted Popularity")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("residuals_plot.png")

if __name__ == "__main__":
    df = pd.read_csv("songs_normalize.csv")
    exploration(df)
    df_for_MLR = preprocessing_for_multiple_linear_regression(df)
    perform_multiple_linear_regression(df_for_MLR)
    
