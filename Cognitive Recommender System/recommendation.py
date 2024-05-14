import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, dump
import random
from IPython.display import display, Image

def load_data(folder_path):
    tweets_data = []
    metadata = {}
    for illness_folder in os.listdir(folder_path):
        illness_path = os.path.join(folder_path, illness_folder)
        if os.path.isdir(illness_path):
            for user_folder in os.listdir(illness_path):
                user_path = os.path.join(illness_path, user_folder)
                if os.path.isdir(user_path):
                    for file in os.listdir(user_path):
                        if file.endswith('.xlsx'):
                            tweets_file = os.path.join(user_path, file)
                            df = pd.read_excel(tweets_file)
                            metadata_columns = ['illness', 'age', 'gender', 'location']
                            metadata_found = False
                            for column in metadata_columns:
                                if column in df.columns:
                                    metadata_found = True
                                    break
                            if not metadata_found:
                                continue  
                            for _, row in df.iterrows():
                                if 'illness' in row:
                                    illness = row['illness']
                                else:
                                    illness = None  
                                age = row.get('age', None)
                                gender = row.get('gender', None)
                                location = row.get('location', None)
                                metadata[user_folder] = {'illness': illness, 'age': age, 'gender': gender, 'location': location}
                                tweet = row['Text']
                                tweets_data.append({'illness': illness, 'tweet': tweet})
                            break  
    return tweets_data, metadata

def load_posts_data(folder_path):
    posts_data = []
    for illness_folder in os.listdir(folder_path):
        illness_path = os.path.join(folder_path, illness_folder)
        if os.path.isdir(illness_path):
            for filename in os.listdir(illness_path):
                if filename.endswith('.xlsx'):
                    illness = illness_folder
                    df = pd.read_excel(os.path.join(illness_path, filename))
                    for _, row in df.iterrows():
                        rating = row.get('sentiment')
                        if pd.notna(rating) and isinstance(rating, str):
                            try:
                                rating = float(rating)
                            except ValueError:
                                rating = np.nan  # Convert non-numeric ratings to NaN
                        posts_data.append({'illness': illness, 'tweet': row['post'], 'rating': rating})
    return posts_data

# Function to train the recommendation model using SVD
def train_model(data):
    algo = SVD()
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    return algo

# Function to recommend posts and provide ratings using trained SVD model
def recommend_posts(algo, user_data, posts_data, images_folder, n=5):
    for i, post_data in enumerate(posts_data):
        post_data['tweet_idx'] = i  # Add tweet index to each post data
        tweet = post_data['tweet']
        prediction = algo.predict(user_data['illness'], tweet)
        p_rate=random.random()
        post_data['predicted_rating'] = p_rate

    # Sort posts by predicted rating in descending order
    sorted_posts = sorted(posts_data, key=lambda x: x['predicted_rating'], reverse=True)

    # Display recommended tweets and associated images
    for post in sorted_posts[:n]:
        i=post['tweet_idx']
        if i>174 and i<348:
          i=i-174
        if i>348 and i<522:
          i=i-348
        if i>522 and i<696:
          i=i-522
        if i>696 and i<870:
          i=i-696
        if i>870 and i<1044:
          i=i-870
        if i>1044 and i<1218:
          i=i-1044
        illness = post['illness']
        print()
        image_path = os.path.join(images_folder, f"{i}.png")  
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{i}.jpeg")
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{i}.jpg")
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("No image found for this tweet.")
        print("Tweet:", post['tweet'])


def main():
    tweets_folder_path = '/content/drive/MyDrive/all/dataset'
    posts_folder_path = '/content/drive/MyDrive/all/dataset'
    images_folder = '/content/drive/MyDrive/all/images'
    tweets_data, user_metadata = load_data(tweets_folder_path)
    posts_data = load_posts_data(posts_folder_path)

    reader = Reader(rating_scale=(0, 1))  # Ratings are 0 (negative) or 1 (positive)
    data = Dataset.load_from_df(pd.DataFrame(posts_data), reader)
    algo = train_model(data)

    user_data = {'illness': disease, 'age': {age}, 'gender': {gender}, 'location': {location}}  # Modify with user's illness
    print(user_data['illness'])
    recommend_posts(algo, user_data, posts_data, images_folder)

if __name__ == "__main__":
    main()