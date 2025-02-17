import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from PIL import Image

# Custom CSS for background and text styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f9;
            color: #333333;
        }
        .css-18e3th9 {
            background-color: #2E3B4E;
        }
        .sidebar .sidebar-content {
            background-color: #2E3B4E;
            color: white;
        }
        .css-1v0mbd3 {
            background-color: #2E3B4E;
            color: white;
        }
        .css-2trqyj {
            background-color: #2E3B4E;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Cache loading models and data to avoid reloading every time
@st.cache_data
def load_data():
    return pd.read_csv("/home/ubuntu/audible-book-recommendations/data/books_with_clusters.csv")

@st.cache_resource
def load_cosine_sim_model():
    with open("/home/ubuntu/audible-book-recommendations/models/cosine_similarity_matrix.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_kmeans_model():
    with open("/home/ubuntu/audible-book-recommendations/models/kmeans_clustering_model.pkl", "rb") as file:
        return pickle.load(file)
# Load dataset and models
merged_df = load_data()
cosine_sim_model = load_cosine_sim_model()
kmeans_model = load_kmeans_model()

# Content-based recommendations for books within the selected genre
def recommend_books_by_content_in_genre(df, cosine_sim, genre, num_recs=5):
    genre_df = df[df['Genre'] == genre]
    if genre_df.empty:
        return []

    genre_idx = genre_df.index.to_list()
    cosine_sim_genre = cosine_sim[genre_idx][:, genre_idx]

    similar_books = list(enumerate(cosine_sim_genre[0]))
    sorted_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:num_recs+1]
    recommendations = [
        (genre_df.iloc[i[0]]['Book Name'], genre_df.iloc[i[0]]['Rating'], genre_df.iloc[i[0]]['Number of Reviews'], genre_df.iloc[i[0]]['cleaned_description'], genre_df.iloc[i[0]]['Price'])
        for i in sorted_books
    ]
    return recommendations

# Clustering-based recommendations for books within the selected genre
def recommend_books_by_cluster_in_genre(df, genre, kmeans_model, num_recs=5):
    genre_df = df[df['Genre'] == genre]
    if genre_df.empty:
        return []

    book_cluster = genre_df['cluster'].mode()[0]
    cluster_books = genre_df[genre_df['cluster'] == book_cluster][['Book Name', 'Rating', 'Number of Reviews', 'cleaned_description', 'Price']].head(num_recs).values.tolist()
    return cluster_books

# Truncate long descriptions
def truncate_description(description, max_length=200):
    return description[:max_length] + '...' if len(description) > max_length else description

def intro_page():
    st.title("📚 Welcome to the Book Recommendation System!")
    
    # Add an image banner to make the page more engaging
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMLDATZl5JIOFBoENdkV2E7sN5a-wZepqPR034qJGmvf-ZrLtT__2rnaaCailIvbJORd0&usqp=CAU", use_container_width=True)
    
    # Section to describe the purpose of the app
    st.markdown("""
        Welcome to our **Book Recommendation System**, where you can explore and discover new books tailored to your interests. Whether you're a fan of fiction, non-fiction, or self-improvement, our system helps you find the perfect book for every mood and need.

        **What can you do here?**
        - 📊 **Explore the Dataset** with **Exploratory Data Analysis (EDA)**: Dive deep into the data and uncover trends like the most popular genres, highly-rated authors, and more.
        - 🔍 **Get Personalized Book Recommendations**: Use advanced **Content-Based** and **Clustering-Based** models to receive tailored book suggestions based on your genre preferences.

        🚀 **How it works:**
        - The system analyzes book features such as ratings, reviews, genres, and descriptions to provide personalized recommendations.
        - **Content-Based** recommendations use book similarity (based on descriptions) to suggest new books within your preferred genres.
        - **Clustering-Based** recommendations group books by similarities in content, providing insights into hidden gems or popular books in your chosen genre.

        🔧 **Tools at Your Disposal**:
        - **Interactive Plots**: Visualize the book trends with interactive charts and graphs.
        - **Customizable Recommendations**: Choose the number of recommendations and the genre that interests you.

        **Use the sidebar to navigate** between different sections and get started on your book discovery journey! ✨
    """, unsafe_allow_html=True)

    

# EDA FAQ Page with Amazon-like FAQ Collapsible Section and Visual Enhancements
def eda_faq_page():
    st.title("📊 EDA - Frequently Asked Questions (FAQs)")

    faq_questions = {
        "What are the most popular genres in the dataset?": "📚 Most Popular Genres",
        "Which authors have the highest-rated books?": "👨‍💻 Highest-Rated Authors",
        "What is the average rating distribution across books?": "📈 Average Rating Distribution",
        "How do ratings vary between books with different review counts?": "⭐ Ratings vs. Review Counts",
        "Which books are frequently clustered together based on descriptions?": "📚 Frequently Clustered Books",
        "How does genre similarity affect book recommendations?": "📊 Genre Similarity and Recommendations",
        "What is the effect of author popularity on book ratings?": "👨‍💻 Effect of Author Popularity on Book Ratings",
        "Which combination of features provides the most accurate recommendations?": "📊 Feature Combinations and Recommendations",
        "Identify books that are highly rated but have low popularity to recommend hidden gems.": "💎 Hidden Gems"
    }

    for question, header in faq_questions.items():
        with st.expander(question):
            if header == "📚 Most Popular Genres":
                genre_count = merged_df['Genre'].value_counts().head(10)
                fig = px.bar(genre_count, x=genre_count.index, y=genre_count.values, labels={'y': 'Book Count', 'index': 'Genres'},
                             title='Top 10 Most Popular Genres', color=genre_count.index, color_continuous_scale='Blues')
                st.plotly_chart(fig)

            elif header == "👨‍💻 Highest-Rated Authors":
                top_authors = merged_df.groupby('Author').agg({'Rating': 'mean', 'Book Name': 'count'}).reset_index()
                top_authors = top_authors[top_authors['Book Name'] > 1].sort_values('Rating', ascending=False).head(10)
                fig = px.bar(top_authors, x='Author', y='Rating', color='Rating', title="Top 10 Authors by Average Rating",
                             labels={'Rating': 'Average Rating'}, color_continuous_scale='Viridis')
                st.plotly_chart(fig)

            elif header == "📈 Average Rating Distribution":
                fig, ax = plt.subplots()
                sns.histplot(merged_df['Rating'], kde=True, ax=ax)
                ax.set_title('Distribution of Ratings')
                st.pyplot(fig)

            elif header == "⭐ Ratings vs. Review Counts":
                fig, ax = plt.subplots()
                sns.scatterplot(data=merged_df, x='Number of Reviews', y='Rating', ax=ax)
                ax.set_title('Ratings vs. Review Counts')
                st.pyplot(fig)

            elif header == "📚 Frequently Clustered Books":
                cluster_count = merged_df['cluster'].value_counts().head(5)
                st.write("Top 5 Clusters by Book Count:")
                st.dataframe(cluster_count)

            elif header == "📊 Genre Similarity and Recommendations":
                st.markdown("""
                Genre similarity plays a crucial role in content-based recommendations by grouping books with similar themes, topics, and styles. Books from the same genre tend to have higher cosine similarity scores, leading to stronger recommendations within the genre.
                """)

            elif header == "👨‍💻 Effect of Author Popularity on Book Ratings":
                author_popularity = merged_df.groupby('Author').agg({'Rating': 'mean', 'Number of Reviews': 'sum'}).reset_index()
                fig = px.scatter(author_popularity, x='Number of Reviews', y='Rating', hover_data=['Author'],
                                 title='Author Popularity vs. Ratings')
                st.plotly_chart(fig)

            elif header == "📊 Feature Combinations and Recommendations":
                st.markdown("""
                Feature combinations such as Genre, Ratings, Review Counts, and Author Popularity can help fine-tune recommendations. By combining these features, we can better understand user preferences and optimize recommendations.
                """)

            elif header == "💎 Hidden Gems":
                hidden_gems = merged_df[(merged_df['Rating'] >= 4.5) & (merged_df['Number of Reviews'] < 100)]
                st.dataframe(hidden_gems[['Book Name', 'Rating', 'Number of Reviews']])

# Recommendation System Page with Cards Layout and Interactive Widgets
def rec_system_page():
    st.title("🔍 Book Recommendation System")

    st.sidebar.header("Choose Recommendation Method")
    rec_method = st.sidebar.radio("Choose Method", ["Content-Based", "Clustering-Based"])

    genre = st.selectbox("Choose a Genre", merged_df['Genre'].unique())
    num_recs = st.slider("Number of Recommendations", 1, 10, 5)

    if rec_method == "Content-Based":
        st.header(f"📚 Content-Based Recommendations for {genre}")
        recs = recommend_books_by_content_in_genre(merged_df, cosine_sim_model, genre, num_recs)
        if recs:
            for rec in recs:
                st.markdown(f"**Book Name:** {rec[0]}\n\n"
                            f"**Rating:** {rec[1]}\n\n"
                            f"**Number of Reviews:** {rec[2]}\n\n"
                            f"**Price:** {rec[4]}\n\n"
                            f"**Description:** {truncate_description(rec[3])}\n\n"
                            "---")
        else:
            st.warning(f"No recommendations found for {genre}.")
    
    elif rec_method == "Clustering-Based":
        st.header(f"📊 Clustering-Based Recommendations for {genre}")
        recs = recommend_books_by_cluster_in_genre(merged_df, genre, kmeans_model, num_recs)
        if recs:
            for rec in recs:
                st.markdown(f"**Book Name:** {rec[0]}\n\n"
                            f"**Rating:** {rec[1]}\n\n"
                            f"**Number of Reviews:** {rec[2]}\n\n"
                            f"**Price:** {rec[4]}\n\n"
                            f"**Description:** {truncate_description(rec[3])}\n\n"
                            "---")
        else:
            st.warning(f"No recommendations found for {genre}.")

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Introduction", "EDA FAQ", "Recommendation System"])

# Display selected page
if menu == "Introduction":
    intro_page()
elif menu == "EDA FAQ":
    eda_faq_page()
elif menu == "Recommendation System":
    rec_system_page()
