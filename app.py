import streamlit as st
import pandas as pd
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download NLTK data
nltk.download('vader_lexicon')


# Custom CSS for styling
def apply_css():
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-image: linear-gradient(to bottom right, #f5f7fa, #e4e8f0);
            background-attachment: fixed;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(to bottom, #2c3e50, #3498db) !important;
            color: white;
        }

        /* Sidebar headers */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown {
            color: white !important;
        }

        /* Sidebar select boxes */
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSlider label {
            color: white !important;
        }

        /* Sidebar select boxes dropdown */
        [data-testid="stSidebar"] .stSelectbox div[role="button"] p {
            color: white !important;
        }

        [data-testid="stSidebar"] .stSelectbox div[role="button"] {
            background-color: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
        }

        /* Cards */
        .css-1fv8s86, .stMetric {
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 15px;
            border-left: 4px solid #3498db;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background: white;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
            margin-right: 5px;
            border: 1px solid #e1e4e8;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background: #3498db !important;
            color: white !important;
            border-color: #3498db !important;
        }

        .stTabs [aria-selected="true"] [data-testid="stMarkdownContainer"] p {
            color: white !important;
        }

        /* Metrics */
        .stMetric {
            text-align: center;
        }

        .stMetric label {
            font-size: 1rem !important;
            color: #2c3e50 !important;
        }

        .stMetric div {
            font-size: 1.5rem !important;
            font-weight: bold !important;
            color: #3498db !important;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50 !important;
        }

        /* Dataframes */
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Buttons */
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            transition: all 0.3s;
        }

        .stButton button:hover {
            background-color: #2980b9;
            color: white;
        }

        /* Expanders */
        .stExpander {
            background: rgba(255,255,255,0.9);
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        ::-webkit-scrollbar-thumb {
            background: #3498db;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #2980b9;
        }
    </style>
    """, unsafe_allow_html=True)


# Set page config
st.set_page_config(
    page_title="Sri Lanka Tourism Sentiment Analysis",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_css()


# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv('dataset3.csv', encoding='ISO-8859-1')

    # Clean travel date
    def clean_travel_date(date):
        try:
            if isinstance(date, str):
                if '-' in date and len(date.split('-')) == 2:
                    return pd.to_datetime(date + '-01')
                elif '-' in date and len(date.split('-')[0]) == 2:
                    return pd.to_datetime('20' + date + '-01')
                else:
                    return pd.to_datetime(parser.parse(date))
            else:
                return pd.to_datetime(date)
        except (ValueError, TypeError):
            return pd.NaT

    df['travel_date'] = df['Travel_Date'].apply(clean_travel_date)
    df['travel_date'] = df['travel_date'].dt.strftime('%Y-%m-%d')
    df['text'] = df['Text'].str.lower().astype(str)

    # Apply sentiment analysis
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(
        lambda score: 'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral')

    return df


df = load_data()

# Sidebar filters with custom styling - Updated version
with st.sidebar:
    # Remove default page navigation elements
    st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 10px; border-radius: 10px; background: rgba(255,255,255,0.1); margin-bottom: 20px;">
        <h2 style="color: white; text-align: center;">🌴 Filters</h2>
    </div>
    """, unsafe_allow_html=True)

    destination_filter = st.selectbox(
        "Select Destination",
        options=["All"] + sorted(df['ï»¿Location_Name'].unique().tolist())
    )

    location_type_filter = st.selectbox(
        "Select Location Type",
        options=["All"] + sorted(
            str(x) for x in df['Location_Type'].unique().tolist()) if 'Location_Type' in df.columns else ["All"]
    )

    year_filter = st.selectbox(
        "Select Year",
        options=["All"] + sorted(pd.to_datetime(df['travel_date']).dt.year.unique().tolist())
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; font-size: 14px;">
        Sri Lanka Tourism Dashboard<br>
        © 2023 All Rights Reserved
    </div>
    """, unsafe_allow_html=True)

# Apply filters
filtered_df = df.copy()
if destination_filter != "All":
    filtered_df = filtered_df[filtered_df['ï»¿Location_Name'] == destination_filter]
if location_type_filter != "All" and 'Location_Type' in df.columns:
    filtered_df = filtered_df[filtered_df['Location_Type'] == location_type_filter]
if year_filter != "All":
    filtered_df = filtered_df[pd.to_datetime(filtered_df['travel_date']).dt.year == int(year_filter)]

# Main app with enhanced styling
st.markdown("""
<div style="background: linear-gradient(to right, #3498db, #2c3e50); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: white; text-align: center;">🌴 Sri Lanka Tourism Sentiment Analysis Dashboard</h1>
    <p style="color: white; text-align: center;">
    This dashboard helps tourism stakeholders understand visitor sentiment about Sri Lankan destinations.
    Analyze reviews by location, time period, and visitor demographics to identify strengths and areas for improvement.
    </p>
</div>
""", unsafe_allow_html=True)

# Overview metrics with cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Reviews", len(filtered_df))

with col2:
    positive_percent = len(filtered_df[filtered_df['sentiment'] == 'positive']) / len(filtered_df) * 100 if len(
        filtered_df) > 0 else 0
    st.metric("Positive Reviews",
              f"{len(filtered_df[filtered_df['sentiment'] == 'positive']):,} ({positive_percent:.1f}%)")

with col3:
    negative_percent = len(filtered_df[filtered_df['sentiment'] == 'negative']) / len(filtered_df) * 100 if len(
        filtered_df) > 0 else 0
    st.metric("Negative Reviews",
              f"{len(filtered_df[filtered_df['sentiment'] == 'negative']):,} ({negative_percent:.1f}%)")

# Tabs with custom styling
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📍 Destination Analysis", "👥 Visitor Insights", "🔤 Text Analysis"])

with tab1:
    st.header("Overall Sentiment Trends")

    # Sentiment over time
    st.subheader("Sentiment Trend Over Time")
    if len(filtered_df) > 0:
        temporal_df = filtered_df.copy()
        temporal_df['year_month'] = pd.to_datetime(temporal_df['travel_date']).dt.to_period('M')
        temporal_sentiment = temporal_df.groupby('year_month')['sentiment'].value_counts(normalize=True).unstack()

        fig, ax = plt.subplots(figsize=(10, 5))
        temporal_sentiment[['positive', 'negative', 'neutral']].plot(kind='line', marker='o', ax=ax)
        plt.title('Sentiment Trend Over Time')
        plt.xlabel('Time Period')
        plt.ylabel('Proportion of Reviews')
        plt.legend(title='Sentiment')
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected filters")

    # Word clouds
    st.subheader("Most Frequent Words by Sentiment")
    if len(filtered_df) > 0:
        fig = plt.figure(figsize=(15, 5))
        for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
            plt.subplot(1, 3, i + 1)
            text = ' '.join(filtered_df[filtered_df['sentiment'] == sentiment]['text'])
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'{sentiment.capitalize()} Sentiment')
            plt.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected filters")

with tab2:
    st.header("Destination Performance Analysis")

    # Destination sentiment
    st.subheader("Sentiment by Destination")
    if len(filtered_df) > 0:
        dest_sentiment = filtered_df.groupby('ï»¿Location_Name')['sentiment'].value_counts(
            normalize=True).unstack().sort_values('positive', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        dest_sentiment.head(10)[['positive', 'neutral', 'negative']].plot(kind='barh', stacked=True, ax=ax)
        plt.title('Sentiment Distribution by Destination (Top 10)')
        plt.xlabel('Proportion of Reviews')
        plt.ylabel('Destination')
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected filters")

    # Map of positive reviews
    st.subheader("Geographic Distribution of Positive Reviews")
    if len(filtered_df) > 0:
        try:
            geolocator = Nominatim(user_agent="tourism_analysis")
            positive_reviews = filtered_df[filtered_df['sentiment'] == 'positive']
            location_counts = positive_reviews['User_Location'].value_counts().reset_index()
            location_counts.columns = ['Location', 'Positive_Reviews']

            locations = []
            for loc in location_counts['Location'].unique()[:20]:  # Limit to top 20 for performance
                try:
                    location = geolocator.geocode(loc + ', Sri Lanka', timeout=10)
                    if location:
                        locations.append({
                            'Location': loc,
                            'Latitude': location.latitude,
                            'Longitude': location.longitude,
                            'Positive_Reviews':
                                location_counts[location_counts['Location'] == loc]['Positive_Reviews'].values[0]
                        })
                except:
                    continue

            if locations:
                locations_df = pd.DataFrame(locations)
                m = folium.Map(location=[7.8731, 80.7718], zoom_start=7)
                for idx, row in locations_df.iterrows():
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=row['Positive_Reviews'] / 2,
                        popup=f"{row['Location']}: {row['Positive_Reviews']} positive reviews",
                        color='green',
                        fill=True,
                        fill_color='green'
                    ).add_to(m)
                folium_static(m, width=1000, height=500)
            else:
                st.warning("Could not geocode locations for mapping")
        except:
            st.warning("Geocoding service unavailable - map cannot be displayed")
    else:
        st.warning("No data available for the selected filters")

with tab3:
    st.header("Visitor Insights")

    # Visitor locations with most positive reviews
    st.subheader("Visitor Locations with Most Positive Reviews")
    if len(filtered_df) > 0:
        location_sentiment = filtered_df.groupby('User_Location')['sentiment'].agg(
            total_reviews='count',
            positive_reviews=lambda x: (x == 'positive').sum(),
            negative_reviews=lambda x: (x == 'negative').sum()
        ).reset_index()

        location_sentiment['positive_pct'] = (location_sentiment['positive_reviews'] / location_sentiment[
            'total_reviews']) * 100
        location_sentiment = location_sentiment[location_sentiment['total_reviews'] >= 3].sort_values('positive_pct',
                                                                                                      ascending=False)

        if len(location_sentiment) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=location_sentiment.head(10), x='positive_pct', y='User_Location', palette='viridis', ax=ax)
            plt.title('Top Locations with Positive Reviews')
            plt.xlabel('Percentage of Positive Reviews')
            plt.ylabel('User Location')
            st.pyplot(fig)

            st.dataframe(
                location_sentiment.head(10)[['User_Location', 'total_reviews', 'positive_reviews', 'positive_pct']])
        else:
            st.warning("No locations with sufficient reviews for analysis")
    else:
        st.warning("No data available for the selected filters")

    # Demographic analysis
    if 'User_Age' in df.columns or 'User_Gender' in df.columns:
        st.subheader("Demographic Analysis")

        if 'User_Age' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=filtered_df, x='sentiment', y='User_Age', order=['positive', 'neutral', 'negative'], ax=ax)
            plt.title('Age Distribution by Sentiment')
            st.pyplot(fig)

        if 'User_Gender' in df.columns:
            gender_sentiment = pd.crosstab(filtered_df['User_Gender'], filtered_df['sentiment'],
                                           normalize='index') * 100
            fig, ax = plt.subplots(figsize=(10, 5))
            gender_sentiment.plot(kind='bar', stacked=True, ax=ax)
            plt.title('Sentiment Distribution by Gender')
            plt.ylabel('Percentage')
            st.pyplot(fig)

with tab4:
    st.header("Text Analysis")

    # Key words by sentiment
    st.subheader("Key Words Driving Sentiment")
    if len(filtered_df) > 0:
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(filtered_df['text'])
        feature_names = vectorizer.get_feature_names_out()

        # Get average scores by sentiment
        positive_words = X[filtered_df['sentiment'] == 'positive'].mean(axis=0).A1
        negative_words = X[filtered_df['sentiment'] == 'negative'].mean(axis=0).A1

        top_positive = pd.Series(positive_words, index=feature_names).sort_values(ascending=False).head(20)
        top_negative = pd.Series(negative_words, index=feature_names).sort_values(ascending=False).head(20)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Positive Words**")
            fig, ax = plt.subplots(figsize=(8, 6))
            top_positive.plot(kind='barh', color='green', ax=ax)
            plt.title('Most Important Positive Words')
            st.pyplot(fig)

        with col2:
            st.markdown("**Top Negative Words**")
            fig, ax = plt.subplots(figsize=(8, 6))
            top_negative.plot(kind='barh', color='red', ax=ax)
            plt.title('Most Important Negative Words')
            st.pyplot(fig)
    else:
        st.warning("No data available for the selected filters")

    # Sample reviews
    st.subheader("Sample Reviews")
    if len(filtered_df) > 0:
        sentiment_sample = st.selectbox("Select sentiment to view samples", ['positive', 'neutral', 'negative'])
        sample_reviews = filtered_df[filtered_df['sentiment'] == sentiment_sample].sample(min(5, len(filtered_df)))

        for idx, row in sample_reviews.iterrows():
            with st.expander(f"Review from {row.get('User_Location', 'unknown')} "):
                st.write(row['text'])
                st.caption(f"Date: {row['travel_date']} | Sentiment: {row['sentiment']} ({row['sentiment_score']:.2f})")
    else:
        st.warning("No data available for the selected filters")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(44, 62, 80, 0.1); border-radius: 10px;">
    <strong>Sri Lanka Tourism Dashboard</strong><br>
    Data last updated: {}<br>
    For official use only
</div>
""".format(pd.to_datetime('today').strftime('%Y-%m-%d')), unsafe_allow_html=True)