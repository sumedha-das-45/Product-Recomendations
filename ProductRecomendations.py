import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("ratings.csv")

# Pivot data to create a user-product matrix
user_product_matrix = df.pivot_table(index='User_Id', columns='Product', values='Rating').fillna(0)

# Normalize data (optional but recommended)
scaler = StandardScaler()
user_product_scaled = scaler.fit_transform(user_product_matrix)

# Compute cosine similarity between users
similarity_matrix = cosine_similarity(user_product_scaled)

# Convert similarity matrix into a DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=user_product_matrix.index, columns=user_product_matrix.index)

# Function to recommend products for a user
def recommend_products(user_id, top_n=2):
    # Get similarity scores for the user
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]

    # Get products rated by similar users
    weighted_scores = {}
    for other_user, sim_score in similar_users.items():
        other_ratings = user_product_matrix.loc[other_user]
        for product, rating in other_ratings.items():
            if user_product_matrix.loc[user_id, product] == 0:
                if product not in weighted_scores:
                    weighted_scores[product] = 0
                weighted_scores[product] += rating * sim_score

    # Sort recommended products by score
    recommended = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

    return [product for product, score in recommended[:top_n]]

# Example: Recommend products for user 1
recommended = recommend_products(1, top_n=2)
print(f"Recommended products for User 1: {recommended}")
