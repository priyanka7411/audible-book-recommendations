import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# --- Sidebar Navigation ---
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🤖 Recommendation System"])

# --- Load Dataset ---
df = pd.read_csv("cleaned_book_data_with_clusters.csv")

# =================== HOME PAGE ===================
if page == "🏠 Home":
    st.title("📘 Audible Insights – Smart Audiobook Recommender")

    st.markdown("""
    Welcome to **Audible Insights**! 🎧  
    Discover your next favorite audiobook with personalized recommendations tailored to your taste.
    """)

    st.markdown("---")

    st.markdown("""
    ### 🔍 What You Can Do:
    - **Get Smart Recommendations** based on your favorite books or genres  
    - **Explore Visual Insights** into audiobook trends, ratings, and more  
    - **Download Your Recommendations** to keep or share  
    """)

    st.markdown("""
    ### 🛠️ How It Works:
    1. **Choose a Recommendation Type**  
    2. **Apply Filters**  
    3. **Explore Recommendations & Visual Insights**
    """)

    st.image("https://www.papillion.org/ImageRepository/Document?documentID=12421", use_container_width=True)

# =================== EDA PAGE ==================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    tab1, tab2, tab3 = st.tabs(["🔍 Overview", "📈 Visualizations", "❓ Q&A Insights"])

    # ======================== OVERVIEW TAB ========================
    with tab1:
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📊 Basic Statistics")
        st.write(df.describe())

        st.subheader("❗ Missing Values")
        st.write(df.isnull().sum())

    # ===================== VISUALIZATIONS TAB =====================
    with tab2:
        st.subheader("🎯 Rating Distribution Across Genres")
        fig1 = px.box(df, x="Genre", y="Rating", color="Genre", title="Rating Distribution by Genre")
        fig1.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig1, use_container_width=True, key="viz_rating_genre")

        st.subheader("📚 Most Common Genres")
        genre_counts = df['Genre'].value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Count']
        fig2 = px.bar(genre_counts.head(10), x='Genre', y='Count', color='Genre')
        st.plotly_chart(fig2, use_container_width=True, key="viz_genre_counts")

        st.subheader("✍️ Top Authors by Book Count")
        top_authors = df['Author'].value_counts().reset_index().head(10)
        top_authors.columns = ['Author', 'Count']
        fig3 = px.bar(top_authors, x='Author', y='Count', color='Author')
        st.plotly_chart(fig3, use_container_width=True, key="viz_top_authors")

        st.subheader("🔄 Ratings vs Number of Reviews")
        fig4 = px.scatter(df, x='Number of Reviews', y='Rating', color='Genre',
                          size='Rating', hover_data=['Book Name', 'Author'])
        fig4.update_layout(xaxis_type="log")
        st.plotly_chart(fig4, use_container_width=True, key="viz_reviews_vs_ratings")

        st.subheader("🎧 Listening Time Distribution")
        fig5 = px.histogram(df, x='Listening Time (minutes)', nbins=30, color='Genre')
        st.plotly_chart(fig5, use_container_width=True, key="viz_listening_time")

        st.subheader("💰 Price vs Rating")
        fig6 = px.scatter(df, x='Price', y='Rating', color='Genre', size='Rating',
                          hover_data=['Book Name', 'Author'])
        fig6.update_layout(xaxis_type="log")
        st.plotly_chart(fig6, use_container_width=True, key="viz_price_vs_rating")

        st.subheader("📈 Rating vs Rank")
        fig7 = px.scatter(df, x='Rank', y='Rating', color='Genre',
                          hover_data=['Book Name', 'Author'])
        st.plotly_chart(fig7, use_container_width=True, key="viz_rating_vs_rank")

        st.subheader("🕒 Average Listening Time by Genre")
        avg_time = df.groupby('Genre')['Listening Time (minutes)'].mean().sort_values(ascending=False).reset_index()
        fig8 = px.bar(avg_time.head(10), x='Genre', y='Listening Time (minutes)', color='Genre')
        st.plotly_chart(fig8, use_container_width=True, key="viz_avg_listening_time")

        st.subheader("🔥 Correlation Heatmap")
        corr_df = df[['Rating', 'Number of Reviews', 'Price', 'Listening Time (minutes)', 'Rank']]
        corr_matrix = corr_df.corr()
        fig9 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig9, use_container_width=True, key="viz_correlation_heatmap")

    # ====================== Q&A INSIGHTS TAB ======================
    with tab3:
        st.markdown("## ❓ Frequently Asked Questions (FAQ)")
        st.markdown("Explore insights from the Audible dataset categorized into Easy, Medium, and Scenario-based questions.")
        st.markdown("---")

        # 🟢 EASY LEVEL
        st.markdown("### 🟢 Easy Level Questions")

        st.markdown("#### 📚 1. What are the most popular genres?")
        fig_e1 = px.bar(genre_counts.head(10), x='Genre', y='Count', color='Genre')
        st.plotly_chart(fig_e1, use_container_width=True, key="faq_genre_counts")

        st.markdown("#### ✍️ 2. Which authors have the highest-rated books?")
        top_rated_books = df[['Author', 'Book Name', 'Rating']].sort_values(by='Rating', ascending=False).drop_duplicates('Author').head(10)
        fig_e2 = px.bar(top_rated_books, x='Author', y='Rating', color='Author', hover_data=['Book Name'])
        st.plotly_chart(fig_e2, use_container_width=True, key="faq_top_rated_authors")

        st.markdown("#### 📈 3. What is the average rating distribution across books?")
        fig_e3 = px.histogram(df, x='Rating', nbins=20, color='Genre')
        st.plotly_chart(fig_e3, use_container_width=True, key="faq_rating_distribution")

        if 'Year' in df.columns:
            st.markdown("#### 📅 4. Are there trends in publication years for popular books?")
            yearly_counts = df[df['Rating'] > 4.0].groupby('Year').size().reset_index(name='Count')
            fig_e4 = px.line(yearly_counts, x='Year', y='Count', title="Popular Books Over Years")
            st.plotly_chart(fig_e4, use_container_width=True, key="faq_yearly_trends")

        st.markdown("#### 💬 5. How do ratings vary between books with different review counts?")
        fig_e5 = px.scatter(df, x='Number of Reviews', y='Rating', color='Genre', size='Rating',
                            hover_data=['Book Name', 'Author'])
        fig_e5.update_layout(xaxis_type="log")
        st.plotly_chart(fig_e5, use_container_width=True, key="faq_reviews_vs_ratings")

        st.markdown("---")

        # 🟡 MEDIUM LEVEL
        st.markdown("### 🟡 Medium Level Questions")

        st.markdown("#### 🔗 6. Which books are frequently clustered together based on descriptions?")
        cluster_sample = df[['Book Name', 'cluster']].groupby('cluster').apply(lambda x: x.head(3)).reset_index(drop=True)
        st.dataframe(cluster_sample)

        st.markdown("#### 🔁 7. How does genre similarity affect book recommendations?")
        st.markdown("Books within the same genre often appear in similar clusters and are recommended together due to shared themes, tone, and listening experience.")

        st.markdown("#### 📊 8. What is the effect of author popularity on book ratings?")
        author_stats = df.groupby('Author').agg({'Rating': 'mean', 'Book Name': 'count'}).reset_index()
        author_stats.columns = ['Author', 'Avg Rating', 'Book Count']
        fig_m1 = px.scatter(author_stats, x='Book Count', y='Avg Rating', size='Avg Rating', color='Avg Rating')
        st.plotly_chart(fig_m1, use_container_width=True, key="faq_author_popularity")

        st.markdown("#### 🧠 9. Which combination of features provides the most accurate recommendations?")
        st.markdown("Combining `Genre`, `Listening Time`, `Rating`, and `cleaned_description` (via clustering or embeddings) produces more accurate and personalized recommendations. Hybrid models using both collaborative and content-based filtering perform best.")

        st.markdown("---")

        # 🔴 SCENARIO-BASED
        st.markdown("### 🔴 Scenario-Based Questions")

        st.markdown("#### 👽 10. A new user likes science fiction books. Which top 5 books should be recommended?")
        sci_fi_books = df[df['Genre'].str.contains('Science Fiction', case=False, na=False)].sort_values(by='Rating', ascending=False).head(5)
        st.dataframe(sci_fi_books[['Book Name', 'Author', 'Rating', 'Genre']])

        st.markdown("#### 🔍 11. For a user who has previously rated thrillers highly, recommend similar books.")
        thriller_books = df[df['Genre'].str.contains('Thriller', case=False, na=False)]
        similar_thrillers = thriller_books.sort_values(by='Rating', ascending=False).head(5)
        st.dataframe(similar_thrillers[['Book Name', 'Author', 'Rating', 'Genre']])

        st.markdown("#### 💎 12. Identify books that are highly rated but have low popularity to recommend hidden gems.")
        hidden_gems = df[(df['Rating'] >= 4.5) & (df['Number of Reviews'] < 100)].sort_values(by='Rating', ascending=False).head(10)
        st.dataframe(hidden_gems[['Book Name', 'Author', 'Rating', 'Number of Reviews']])

        st.markdown("💡 *These insights help users discover personalized content and optimize the recommendation system!*")


# =================== RECOMMENDATION SYSTEM ===================
elif page == "🤖 Recommendation System":
    st.title("🔮 Personalized Book Recommendations")

    # Load models
    try:
        svd_model = joblib.load("svd_recommender_model.pkl")
    except Exception as e:
        st.error(f"Collaborative model couldn't be loaded: {e}")
        st.stop()

    try:
        cosine_sim = joblib.load("cosine_sim.pkl")
    except Exception as e:
        st.warning("Content-based model not found.")
        cosine_sim = None

    try:
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    except Exception as e:
        st.warning("TF-IDF vectorizer not found.")
        tfidf_vectorizer = None

    try:
        kmeans_model = joblib.load("kmeans_model.pkl")
    except Exception as e:
        st.warning("Clustering model not found.")
        kmeans_model = None

    # Recommendation Type
    approach = st.radio("🧠 Choose Recommendation Type:", ["Content-Based", "Clustering-Based", "Hybrid"])

    # User Filters
    st.markdown("### 🎯 Choose Your Preferences")
    selected_genres = st.multiselect("📚 Preferred Genres", sorted(df['Genre'].unique()))
    selected_authors = st.multiselect("✍️ Preferred Authors", sorted(df['Author'].unique()))

    min_rating = st.slider("⭐ Minimum Rating", min_value=0.0, max_value=5.0, step=0.1, value=4.0)
    min_reviews = st.slider("💬 Minimum Number of Reviews", min_value=0, max_value=5000, step=100, value=100)

    filtered_df = df.copy()
    if selected_genres:
        filtered_df = filtered_df[filtered_df['Genre'].isin(selected_genres)]
    if selected_authors:
        filtered_df = filtered_df[filtered_df['Author'].isin(selected_authors)]
    filtered_df = filtered_df[
        (filtered_df['Rating'] >= min_rating) & 
        (filtered_df['Number of Reviews'] >= min_reviews)
    ]

    if filtered_df.empty:
        st.warning("No books found. Try adjusting filters.")
        st.stop()

    book_list = sorted(filtered_df["Book Name"].unique())
    selected_book = st.selectbox("📖 Choose a book you like", book_list)
    top_n = 5

    if st.button("✨ Show Recommendations"):
        recommended = pd.DataFrame()

        if approach == "Content-Based":
            if cosine_sim is None:
                st.error("Model not loaded.")
                st.stop()
            indices = pd.Series(df.index, index=df["Book Name"]).drop_duplicates()
            if selected_book not in indices:
                st.error("Selected book not found in the dataset.")
                st.stop()
            idx = indices[selected_book]
            sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
            book_indices = [i[0] for i in sim_scores]
            recommended = df.iloc[book_indices]

        elif approach == "Clustering-Based":
            cluster_id = df[df["Book Name"] == selected_book]["cluster"].values[0]
            cluster_books = df[(df["cluster"] == cluster_id) & (df["Book Name"] != selected_book)]
            cluster_size = len(cluster_books)
            st.info(f"🔍 Found {cluster_size} similar books in the same cluster.")
            recommended = cluster_books.sample(min(top_n, cluster_size))

        elif approach == "Hybrid":
            dummy_user_id = 9999
            predictions = []
            for book in df[df["Book Name"] != selected_book]["Book Name"].unique():
                try:
                    est_rating = svd_model.predict(dummy_user_id, book).est
                    content_score = 0
                    if cosine_sim is not None:
                        idx = df[df["Book Name"] == selected_book].index[0]
                        sim_idx = df[df["Book Name"] == book].index[0]
                        content_score = cosine_sim[idx][sim_idx]
                    hybrid_score = 0.7 * est_rating + 0.3 * content_score
                    predictions.append((book, hybrid_score))
                except:
                    continue
            pred_df = pd.DataFrame(predictions, columns=["Book Name", "Hybrid Score"])
            recommended = pred_df.merge(df, on="Book Name", how="left").sort_values("Hybrid Score", ascending=False).head(top_n)

        # Display Recommendations
        st.markdown("### 📚 Recommended Books")
        if recommended.empty:
            st.warning("No recommendations found. Try changing filters.")
        else:
            for _, row in recommended.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image("https://img.icons8.com/ios-filled/100/book.png", width=80)
                    with col2:
                        st.markdown(f"**📖 {row['Book Name']}**")
                        st.markdown(f"👤 Author: `{row['Author']}`")
                        st.markdown(f"🎧 Genre: `{row['Genre']}` | ⏱️ Listening Time: `{int(row['Listening Time (minutes)'])} mins`")
                        st.markdown(f"⭐ Rating: `{row['Rating']}` | 💬 Reviews: `{row['Number of Reviews']}`")

            # ✅ Download Button
            csv = recommended.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Recommendations as CSV",
                data=csv,
                file_name="recommended_books.csv",
                mime="text/csv",
            )
