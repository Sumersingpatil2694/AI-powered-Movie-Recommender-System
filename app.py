# ===============================================================
# 🎬 MOVIE RECOMMENDER SYSTEM - VERSION v3.0
# ===============================================================
from dotenv import load_dotenv
import os
load_dotenv()
import streamlit as st
import pandas as pd
import pickle
import requests
import json
from datetime import datetime
from fuzzywuzzy import fuzz
from collections import Counter
import time

# ===============================================================
# 🔐 CONFIGURATION CLASS
# Stores all API keys, URLs, and app settings
# ===============================================================
class Config:
    """
    Configuration class for storing application constants
    - API_KEY: TMDB API key for fetching movie data
    - BASE_URL: TMDB API base endpoint
    - IMAGE_BASE: Base URL for movie poster images
    - PLACEHOLDER: Fallback image when poster not available
    - CACHE_TIME: Duration to cache API responses (in seconds)
    - REQUEST_TIMEOUT: Maximum time to wait for API response
    - MAX_RETRIES: Number of retry attempts for failed API calls
    """
    API_KEY = os.getenv("TMDB_API_KEY")
    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
    PLACEHOLDER = "https://via.placeholder.com/300x450?text=No+Image"
    CACHE_TIME = 3600  
    REQUEST_TIMEOUT = 15 
    MAX_RETRIES = 3 
 
# ===============================================================
# 📦 DATA LOADING FUNCTIONS
# Load pre-processed movie data and similarity matrix
# ===============================================================
@st.cache_resource
def load_data():
    """
    Load the preprocessed movie data and similarity matrix
    Uses Streamlit's cache to avoid reloading on every interaction
    
    Returns:
        tuple: (movies_dataframe, similarity_matrix) or (None, None) on error
    """
    try:
        # Load movie dictionary from pickle file
        movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
        movies = pd.DataFrame(movies_dict)
        
        # Load similarity matrix
        try:
            similarity = pickle.load(open("similarity.pkl", "rb"))
        except FileNotFoundError:
            st.warning("⚠️ Similarity matrix not found. Building from scratch...")
            similarity = build_similarity_matrix(movies)
            
        return movies, similarity
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None

def build_similarity_matrix(movies):
    """
    Build similarity matrix if similarity.pkl is missing
    Uses CountVectorizer and cosine_similarity from sklearn
    
    Args:
        movies: DataFrame containing movie data with 'tags' column
    
    Returns:
        numpy.ndarray: Cosine similarity matrix
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    if 'tags' in movies.columns:
        # Convert text tags to numerical vectors
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies['tags']).toarray()
        
        # Calculate cosine similarity between all movie vectors
        similarity = cosine_similarity(vectors)
        
        # Save for future use
        with open("similarity.pkl", "wb") as f:
            pickle.dump(similarity, f)
        
        return similarity
    else:
        st.error("❌ 'tags' column not found in data")
        return None

# ===============================================================
# 🎨 CUSTOM CSS STYLING 
# Applies beautiful dark theme with red accents
# ===============================================================
def apply_custom_css():
    """
    Apply custom CSS styling to the Streamlit app
    Creates a Netflix-inspired dark theme with:
    - Black/dark gray gradient background
    - Red (#e50914) accent colors
    - Smooth hover animations
    - Professional card layouts
    - Custom buttons and inputs
    """
    st.markdown("""
    <style>
    /* ===== MAIN BACKGROUND ===== */
    /* Gradient background for the entire app */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #0d0d0d 100%);
    }
    
    /* ===== TEXT STYLING ===== */
    /* All headers and text with white color and subtle glow effect */
    h1, h2, h3, h4, label, .stMarkdown {
        color: white !important;
        font-family: 'Helvetica Neue', 'Arial', sans-serif;
        text-shadow: 0 0 10px rgba(229, 9, 20, 0.3);
    }
    
    /* Main title with gradient text effect */
    h1 {
        font-size: 3em !important;
        font-weight: 700 !important;
        margin-bottom: 0.5em !important;
        background: linear-gradient(90deg, #e50914 0%, #f40612 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* ===== MOVIE CARDS ===== */
    /* Card container for each movie with hover effect */
    .movie-card {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.4), 0 0 20px rgba(229, 9, 20, 0.1);
        margin: 10px 0;
        border: 1px solid rgba(229, 9, 20, 0.2);
        cursor: pointer;
    }
    
    /* Hover effect: lift up and enlarge slightly */
    .movie-card:hover {
        transform: translateY(-15px) scale(1.05);
        box-shadow: 0 12px 24px rgba(229, 9, 20, 0.5);
        border-color: #e50914;
    }
    
    /* Movie title styling inside cards */
    .movie-title {
        color: #ffffff;
        font-weight: bold;
        font-size: 18px;
        margin: 15px 0;
        min-height: 45px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    
    /* ===== RATING BADGES ===== */
    /* Small badge showing movie rating with pulse animation */
    .rating-badge {
        background: linear-gradient(135deg, #e50914 0%, #f40612 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 25px;
        font-size: 13px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        box-shadow: 0 4px 8px rgba(229, 9, 20, 0.4);
        animation: pulse 2s infinite;
    }
    
    /* Pulse animation for rating badges */
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 8px rgba(229, 9, 20, 0.4); }
        50% { box-shadow: 0 4px 16px rgba(229, 9, 20, 0.8); }
    }
    
    /* ===== SIDEBAR STYLING ===== */
    /* Dark sidebar with red border */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141414 0%, #0a0a0a 100%) !important;
        border-right: 2px solid rgba(229, 9, 20, 0.3);
    }
    
    /* ===== BUTTONS ===== */
    /* Primary button styling with hover effect */
    .stButton>button {
        background: linear-gradient(135deg, #e50914 0%, #f40612 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
        cursor: pointer;
    }
    
    /* Button hover effect: lift and brighten */
    .stButton>button:hover {
        background: linear-gradient(135deg, #f40612 0%, #f40612 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6);
    }
    
    /* ===== INPUT FIELDS ===== */
    /* Text input and select box styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background: #2d2d2d !important;
        color: white !important;
        border: 2px solid rgba(229, 9, 20, 0.4) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 16px !important;
    }
    
    /* Input focus state: highlight with red border */
    .stTextInput>div>div>input:focus {
        border-color: #e50914 !important;
        box-shadow: 0 0 10px rgba(229, 9, 20, 0.5) !important;
    }
    
    /* ===== STATISTICS CARDS ===== */
    /* Cards for displaying statistics with slide animation on hover */
    .stat-card {
        background: linear-gradient(145deg, #1a1a1a, #252525);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #e50914;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    
    /* Stat card hover effect: slide right */
    .stat-card:hover {
        transform: translateX(10px);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.3);
    }
    
    /* Large numbers in stat cards */
    .stat-card h2 {
        color: #e50914 !important;
        font-size: 2.5em !important;
        margin: 10px 0 !important;
    }
    
    /* Stat card labels */
    .stat-card h3 {
        color: #ffffff !important;
        font-size: 1.2em !important;
        margin-bottom: 10px !important;
    }
    
    /* ===== GENRE TAGS ===== */
    /* Small tags for movie genres */
    .genre-tag {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        color: #e50914;
        padding: 8px 18px;
        border-radius: 25px;
        display: inline-block;
        margin: 5px;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid rgba(229, 9, 20, 0.4);
        transition: all 0.3s ease;
    }
    
    /* Genre tag hover: invert colors */
    .genre-tag:hover {
        background: #e50914;
        color: white;
        transform: scale(1.1);
    }
    
    /* ===== HERO SECTION ===== */
    /* Large banner section on home page */
    .hero-section {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.2) 0%, rgba(0, 0, 0, 0.8) 100%);
        padding: 40px;
        border-radius: 20px;
        margin: 20px 0;
        border: 2px solid rgba(229, 9, 20, 0.3);
        box-shadow: 0 8px 24px rgba(229, 9, 20, 0.2);
    }
    
    /* ===== INFO BOXES ===== */
    /* Information boxes with left border accent */
    .info-box {
        background: rgba(229, 9, 20, 0.1);
        border-left: 4px solid #e50914;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* ===== CLICKABLE MOVIE IMAGE ===== */
    /* Make movie poster images clickable */
    .clickable-movie {
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .clickable-movie:hover {
        transform: scale(1.05);
        filter: brightness(1.2);
    }
    
    /* ===== AUTOCOMPLETE SUGGESTIONS ===== */
    /* Styling for search suggestions dropdown */
    .suggestion-box {
        background: #2d2d2d;
        border: 2px solid rgba(229, 9, 20, 0.4);
        border-radius: 8px;
        padding: 10px;
        margin-top: 5px;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .suggestion-item {
        padding: 10px;
        margin: 5px 0;
        background: #1a1a1a;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .suggestion-item:hover {
        background: #e50914;
        transform: translateX(5px);
    }
    
    /* ===== SCROLLBAR STYLING ===== */
    /* Custom scrollbar for better aesthetics */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #e50914;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #f40612;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    /* Adjustments for mobile devices */
    @media (max-width: 768px) {
        .movie-card {
            padding: 15px;
        }
        .movie-title {
            font-size: 14px;
            min-height: 35px;
        }
        h1 {
            font-size: 2em !important;
        }
    }
/* ================= OTT CAPSULE BUTTONS ================= */

.ott-capsule-wrapper {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 10px;
}

.ott-capsule {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 999px; /* capsule shape */
    background: linear-gradient(135deg, #1f1f1f, #2b2b2b);
    border: 1px solid rgba(255,255,255,0.15);
    color: white;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
}

.ott-capsule:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 8px 18px rgba(229,9,20,0.5);
    border-color: #e50914;
}

.ott-capsule img {
    width: 22px;
    height: 22px;
    border-radius: 50%;
}

    </style>
    """, unsafe_allow_html=True)

# ===============================================================
# 🌐 API FUNCTIONS WITH RETRY LOGIC
# Functions to fetch data from TMDB API with error handling
# ===============================================================
def make_api_request(url, params, retries=Config.MAX_RETRIES):
    """
    Make HTTP request to API with automatic retry on failure
    
    Args:
        url: API endpoint URL
        params: Query parameters dictionary
        retries: Number of retry attempts (default from Config)
    
    Returns:
        dict: JSON response from API or None on failure
        
    Features:
        - Automatic retry with exponential backoff
        - Rate limit handling (HTTP 429)
        - Connection error handling
        - Timeout handling
        - User-agent header to avoid blocking
    """
    for attempt in range(retries):
        try:
            # Add user agent to avoid being blocked by servers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Make GET request with timeout
            response = requests.get(
                url, 
                params=params, 
                timeout=Config.REQUEST_TIMEOUT,
                headers=headers
            )
            
            # Success: return JSON data
            if response.status_code == 200:
                return response.json()
            
            # Rate limit exceeded: wait and retry with exponential backoff
            elif response.status_code == 429:
                wait_time = 2 ** attempt  # 1s, 2s, 4s, ...
                time.sleep(wait_time)
                continue
            
            # Other error: return None
            else:
                return None
                
        except requests.exceptions.ConnectionError:
            # Network connection error: retry if attempts remaining
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return None
                
        except requests.exceptions.Timeout:
            # Request timeout: retry if attempts remaining
            if attempt < retries - 1:
                time.sleep(1)
                continue
            else:
                return None
                
        except Exception:
            # Unexpected error: return None
            return None
    
    return None

@st.cache_data(ttl=Config.CACHE_TIME)
def fetch_movie_details(movie_id):
    """
    Fetch complete details for a specific movie by ID
    Includes videos, cast, similar movies, and reviews
    
    Args:
        movie_id: TMDB movie ID (integer)
    
    Returns:
        dict: Complete movie data including poster, overview, cast, etc.
        
    Cached for 1 hour to reduce API calls
    """
    url = f"{Config.BASE_URL}/movie/{movie_id}"
    params = {
        'api_key': Config.API_KEY,
        'append_to_response': 'videos,credits,similar,reviews'
    }
    return make_api_request(url, params)
@st.cache_data(ttl=Config.CACHE_TIME)
def fetch_watch_providers(movie_id, country="IN"):
    """
    Fetch OTT platforms where the movie is available
    Uses TMDB Watch Providers API
    """
    url = f"{Config.BASE_URL}/movie/{movie_id}/watch/providers"
    params = {"api_key": Config.API_KEY}

    data = make_api_request(url, params)

    if data and "results" in data and country in data["results"]:
        return data["results"][country]

    return None

@st.cache_data(ttl=Config.CACHE_TIME)
def fetch_poster(movie_id):
    """
    Fetch poster image URL for a specific movie
    Returns placeholder image if poster not available
    
    Args:
        movie_id: TMDB movie ID
    
    Returns:
        str: Full URL to poster image or placeholder
    """
    data = fetch_movie_details(movie_id)
    if data and data.get("poster_path"):
        return Config.IMAGE_BASE + data["poster_path"]
    return Config.PLACEHOLDER

@st.cache_data(ttl=Config.CACHE_TIME)
def get_trending(time_window='week', page=1):
    """
    Fetch trending movies from TMDB
    
    Args:
        time_window: 'day' or 'week' (default: 'week')
        page: Page number for pagination (default: 1)
    
    Returns:
        list: List of up to 12 trending movie dictionaries
    """
    url = f"{Config.BASE_URL}/trending/movie/{time_window}"
    params = {'api_key': Config.API_KEY, 'page': page}
    
    data = make_api_request(url, params)
    if data:
        return data.get("results", [])[:12]
    return []

@st.cache_data(ttl=Config.CACHE_TIME)
def search_movies_api(query):
    """
    Search for movies using TMDB search API
    
    Args:
        query: Movie title or keywords to search
    
    Returns:
        list: List of matching movies from TMDB
    """
    url = f"{Config.BASE_URL}/search/movie"
    params = {'api_key': Config.API_KEY, 'query': query}
    
    data = make_api_request(url, params)
    if data:
        return data.get("results", [])
    return []

# ===============================================================
# 🔍 SEARCH FUNCTIONS
# Fuzzy search and autocomplete functionality
# ===============================================================
def get_movie_suggestions(query, movies, limit=10):
    """
    Get movie suggestions for autocomplete based on partial query
    
    Args:
        query: Partial movie title entered by user
        movies: DataFrame containing all movies
        limit: Maximum number of suggestions to return
    
    Returns:
        list: List of matching movie titles
        
    Features:
        - Case-insensitive search
        - Partial matching (contains query)
        - Limited to top N results
    """
    if not query or len(query) < 2:
        return []
    
    query = query.lower().strip()
    
    # Filter movies that contain the query string
    matches = movies[movies['title'].str.lower().str.contains(query, na=False)]
    
    # Return top N matches
    return matches['title'].head(limit).tolist()

def fuzzy_search_movie(query, movies, threshold=60):
    """
    Find best matching movie using fuzzy string matching
    Handles typos and variations in movie titles
    
    Args:
        query: Movie title to search for
        movies: DataFrame containing all movies
        threshold: Minimum similarity score (0-100) to accept match
    
    Returns:
        tuple: (movie_index, matched_title) or (None, None) if no match
        
    Search Strategy:
        1. Try exact match first (fastest)
        2. Try partial match (contains query)
        3. Use fuzzy matching with similarity scores
    """
    query = query.lower().strip()
    
    # Step 1: Try exact match
    exact_match = movies[movies['title'].str.lower() == query]
    if not exact_match.empty:
        return exact_match.index[0], exact_match.iloc[0]['title']
    
    # Step 2: Try partial match (movie title contains query)
    partial_match = movies[movies['title'].str.lower().str.contains(query, na=False)]
    if not partial_match.empty:
        return partial_match.index[0], partial_match.iloc[0]['title']
    
    # Step 3: Fuzzy matching with similarity scores
    movies_copy = movies.copy()
    movies_copy['similarity_score'] = movies_copy['title'].apply(
        lambda x: fuzz.ratio(query.lower(), x.lower())
    )
    
    # Get movie with highest similarity score
    best_match = movies_copy.loc[movies_copy['similarity_score'].idxmax()]
    
    # Only return if score meets threshold
    if best_match['similarity_score'] >= threshold:
        return best_match.name, best_match['title']
    
    return None, None

# ===============================================================
# 🎯 RECOMMENDATION ENGINE
# Core algorithm for finding similar movies
# ===============================================================
def recommend(movie_name, movies, similarity, n=15):
    """
    Generate movie recommendations based on similarity
    
    Args:
        movie_name: Title of the movie to base recommendations on
        movies: DataFrame containing all movies
        similarity: Cosine similarity matrix
        n: Number of recommendations to return (default: 15)
    
    Returns:
        tuple: (recommendations_list, posters_list, error_message)
            - recommendations_list: List of dicts with title, id, similarity score
            - posters_list: List of poster URLs
            - error_message: None if successful, error string if failed
    
    Algorithm:
        1. Find the movie using fuzzy search
        2. Get similarity scores for all other movies
        3. Sort by similarity (descending)
        4. Return top N most similar movies
        5. IMPORTANT: Include the searched movie as first result
    """
    
    # Find movie index using fuzzy search
    movie_idx, matched_title = fuzzy_search_movie(movie_name, movies)
    
    if movie_idx is None:
        return [], [], f"❌ Movie '{movie_name}' not found. Try a different title."
    
    # Get the searched movie details
    searched_movie = movies.iloc[movie_idx]
    
    # Get similarity scores for all movies
    # List of tuples: [(index, similarity_score), ...]
    distances = list(enumerate(similarity[movie_idx]))
    
    # Sort by similarity score (descending)
    distances = sorted(distances, reverse=True, key=lambda x: x[1])
    
    # Build recommendations list
    recommendations = []
    posters = []
    
    # FIRST: Add the searched movie itself (100% match)
    recommendations.append({
        'title': searched_movie['title'],
        'movie_id': searched_movie['movie_id'],
        'similarity': 100.0,
        'is_searched': True  # Flag to highlight this movie
    })
    posters.append(fetch_poster(searched_movie['movie_id']))
    
    # THEN: Add similar movies (skip index 0 which is the same movie)
    for idx, score in distances[1:n]:
        movie = movies.iloc[idx]
        recommendations.append({
            'title': movie['title'],
            'movie_id': movie['movie_id'],
            'similarity': round(score * 100, 2),
            'is_searched': False
        })
        posters.append(fetch_poster(movie['movie_id']))
    
    return recommendations, posters, None

# ===============================================================
# 💾 WATCHLIST MANAGEMENT 
# ===============================================================

def load_watchlist(user_id='default'):
    """
    Load user's watchlist from JSON file
    """
    filepath = f"watchlist_{user_id}.json"
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load watchlist: {e}")
    return []


def save_watchlist(watchlist, user_id='default'):
    """
    Save user's watchlist to JSON file
    FIX: Converts numpy types to native Python types
    """
    filepath = f"watchlist_{user_id}.json"

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif hasattr(obj, "item"):   # numpy.int64, numpy.float64
            return obj.item()
        return obj

    try:
        watchlist = convert(watchlist)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(watchlist, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Could not save watchlist: {e}")
        return False


def add_to_watchlist(movie_title, movie_id):
    """
    Add a movie to user's watchlist
    FIX: Explicit type casting
    """
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    # 🔥 IMPORTANT FIX
    movie_id = int(movie_id)

    if not any(item['title'] == movie_title for item in st.session_state.watchlist):
        st.session_state.watchlist.append({
            'title': str(movie_title),
            'movie_id': movie_id,
            'added_date': datetime.now().isoformat()
        })
        save_watchlist(st.session_state.watchlist)
        return True
    return False


def remove_from_watchlist(index):
    """
    Remove a movie from watchlist by index
    """
    if 'watchlist' in st.session_state and index < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(index)
        save_watchlist(st.session_state.watchlist)
        return True
    return False

# ===============================================================
# 📊 STATISTICS FUNCTIONS
# Display dataset statistics and analytics
# ===============================================================
def show_statistics(movies):
    """
    Display overview statistics about the movie dataset
    Shows: Total movies, genres, year range, average rating
    
    Args:
        movies: DataFrame containing all movie data
    """
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Stat 1: Total Movies
    with col1:
        total_movies = len(movies)
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎬 Total Movies</h3>
            <h2>{total_movies:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Stat 2: Unique Genres
    with col2:
        try:
            all_genres = set()
            if 'genres' in movies.columns:
                for genres_str in movies['genres'].dropna():
                    if isinstance(genres_str, str):
                        try:
                            genres_list = eval(genres_str) if genres_str.startswith('[') else []
                            for g in genres_list:
                                if isinstance(g, dict):
                                    all_genres.add(g.get('name', ''))
                                else:
                                    all_genres.add(str(g))
                        except:
                            pass
            unique_genres = len([g for g in all_genres if g])
        except:
            unique_genres = "N/A"
        
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎭 Genres</h3>
            <h2>{unique_genres if unique_genres != 0 else 'N/A'}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Stat 3: Year Range
    with col3:
        try:
            if 'release_date' in movies.columns:
                years = pd.to_datetime(movies['release_date'], errors='coerce').dt.year.dropna()
                if len(years) > 0:
                    year_range = f"{int(years.min())}-{int(years.max())}"
                else:
                    year_range = "N/A"
            else:
                year_range = "N/A"
        except:
            year_range = "N/A"
        
        st.markdown(f"""
        <div class='stat-card'>
            <h3>📅 Year Range</h3>
            <h2>{year_range}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Stat 4: Average Rating
    with col4:
        try:
            if 'vote_average' in movies.columns:
                avg_rating = round(movies['vote_average'].mean(), 1)
            else:
                avg_rating = "N/A"
        except:
            avg_rating = "N/A"
        
        st.markdown(f"""
        <div class='stat-card'>
            <h3>⭐ Avg Rating</h3>
            <h2>{avg_rating if avg_rating != "N/A" else "N/A"}</h2>
        </div>
        """, unsafe_allow_html=True)

# ===============================================================
# 🎬 MOVIE DETAILS PAGE
# Display comprehensive information about a specific movie
# ===============================================================
def display_movie_details(movie_id):
    """
    Display detailed information about a movie including:
    - Poster and basic info
    - Rating, year, runtime
    - Genres
    - Overview/plot
    - Trailer video
    - Top cast members
    - Similar movies
    
    Args:
        movie_id: TMDB movie ID to display
        
    Features:
        - Fetches real-time data from TMDB API
        - Shows trailer if available
        - Displays cast with photos
        - Suggests similar movies (clickable)
        - Add to watchlist button
    """
    
    with st.spinner("🔄 Loading movie details..."):
        data = fetch_movie_details(movie_id)
    
    # Handle API error
    if not data or data.get('success') == False:
        st.error("❌ Could not fetch movie details. API connection issue.")
        st.info("💡 This might be due to:")
        st.write("- Invalid API key")
        st.write("- Network connection issues")
        st.write("- Rate limiting")
        return
    
    # Layout: Poster on left, info on right
    col1, col2 = st.columns([1, 2])
    
    # LEFT COLUMN: Poster image
    with col1:
        if data.get("poster_path"):
            st.image(Config.IMAGE_BASE + data["poster_path"],width=300)
        else:
            st.image(Config.PLACEHOLDER,width=300)
    
    # RIGHT COLUMN: Movie information
    with col2:
        # Title
        st.title(f"🎬 {data.get('title', 'Unknown')}")
        
        # Rating, Year, Runtime badges
        rating = data.get('vote_average', 0)
        year = data.get('release_date', 'Unknown')[:4] if data.get('release_date') else 'Unknown'
        runtime = data.get('runtime', 0)
        
        st.markdown(f"""
        <div class='rating-badge'>⭐ {rating}/10</div>
        <div class='rating-badge'>📅 {year}</div>
        <div class='rating-badge'>🕒 {runtime} min</div>
        """, unsafe_allow_html=True)
        
        # ================= FIXED OTT CAPSULE PLATFORMS =================
        providers = fetch_watch_providers(movie_id)

        if providers and "flatrate" in providers:
            st.markdown("### 📺 Available On")

            # Build the HTML properly with proper escaping
            capsule_html = '<div class="ott-capsule-wrapper">'

            for p in providers["flatrate"]:
            # ✅ 1. Safe access with .get() method
             logo_path = p.get("logo_path", "")
            if logo_path:
                logo = Config.IMAGE_BASE.replace("/w500", "/w92") + logo_path
            else:
                # ✅ 2. Placeholder for missing logos
                logo = "https://via.placeholder.com/22"
            
            name = p.get("provider_name", "Unknown")
            link = providers.get("link", f"https://www.themoviedb.org/movie/{movie_id}/watch")

            # ✅ 3. Triple single quotes for better escaping
            capsule_html += f'''
        <a href="{link}" target="_blank" style="text-decoration:none;">
            <div class="ott-capsule">
            <img src="{logo}" alt="{name}">
            <span>{name}</span>
            </div>
        </a>
        '''

            capsule_html += '</div>'

            # ✅ 4. Render the HTML
            st.markdown(capsule_html, unsafe_allow_html=True)

        else:
            st.info("📭 Currently not available on OTT platforms in India")

        # Genres as tags
        genres = data.get('genres', [])
        if genres:
            st.markdown("**Genres:**")
            genre_html = "".join([
                f"<span class='genre-tag'>{g['name']}</span>" 
                for g in genres
            ])
            st.markdown(genre_html, unsafe_allow_html=True)
        
        # Overview/Plot
        st.markdown("**📖 Overview:**")
        st.write(data.get('overview', 'No overview available.'))
        
        # Add to Watchlist button
        if st.button("⭐ Add to Watchlist", key=f"watchlist_{movie_id}"):
            if add_to_watchlist(data.get('title'), movie_id):
                st.success("✅ Added to watchlist!")
            else:
                st.info("ℹ️ Already in watchlist")
    
    st.markdown("---")
    
    # TRAILER SECTION
    videos = data.get('videos', {}).get('results', [])
    trailer = next((v for v in videos if v.get('type') == 'Trailer'), None)
    
    if trailer:
        st.subheader("🎥 Trailer")
        st.video(f"https://www.youtube.com/watch?v={trailer['key']}")
    
    # CAST SECTION
    credits = data.get('credits', {})
    cast = credits.get('cast', [])[:10]
    
    if cast:
        st.subheader("🎭 Top Cast")
        cast_cols = st.columns(5)
        for idx, actor in enumerate(cast[:5]):
            with cast_cols[idx]:
                if actor.get('profile_path'):
                    st.image(Config.IMAGE_BASE + actor['profile_path'],width=300)
                else:
                    st.image(Config.PLACEHOLDER,width=300)
                st.caption(f"**{actor['name']}**")
                st.caption(f"*{actor.get('character', '')}*")
    
    # SIMILAR MOVIES SECTION
    similar = data.get('similar', {}).get('results', [])[:6]
    if similar:
        st.subheader("🔗 Similar Movies")
        sim_cols = st.columns(6)
        for idx, movie in enumerate(similar):
            with sim_cols[idx]:
                if movie.get('poster_path'):
                    st.image(Config.IMAGE_BASE + movie['poster_path'],width=300)
                else:
                    st.image(Config.PLACEHOLDER,width=300)
                st.caption(movie['title'])
                # Clickable button to view this similar movie
                if st.button("View", key=f"similar_{movie['id']}_{idx}"):
                    st.session_state.selected_movie_id = movie['id']
                    st.rerun()

# ===============================================================
# 🏠 HOME PAGE
# Main landing page with trending movies and features
# ===============================================================
def show_home_page(movies):
    """
    Display the home page with:
    - Hero section with app title
    - Feature highlights
    - Quick statistics
    - Trending movies preview (FIXED: now clickable)
    - How it works section
    - Call to action
    
    Args:
        movies: DataFrame containing all movies
        
    FIXES APPLIED:
    - Trending movie posters now clickable to view details
    - Home search bar now functional
    """
    
    # HERO SECTION
    st.markdown("""
    <div class='hero-section'>
        <h1 style='text-align: center; font-size: 4em;'>🎬 Movie Recommender Pro</h1>
        <p style='text-align: center; font-size: 1.5em; color: #ffffff;'>
            Discover Your Next Favorite Movie with AI-Powered Recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # FEATURE HIGHLIGHTS
    st.markdown("### ✨ Why Choose Movie Recommender Pro?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <h3>🤖 AI-Powered</h3>
            <p>Advanced machine learning algorithms analyze movie similarities using cosine similarity and content-based filtering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
            <h3>🎯 Personalized</h3>
            <p>Get tailored recommendations based on your favorite movies. Our smart fuzzy search understands what you're looking for</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-box'>
            <h3>📊 Real-Time Data</h3>
            <p>Access up-to-date information from TMDB with trending movies, ratings, cast details, and trailers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # QUICK STATS
    st.markdown("### 📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎬 Movies</h3>
            <h2>{len(movies):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        watchlist_count = len(st.session_state.get('watchlist', []))
        st.markdown(f"""
        <div class='stat-card'>
            <h3>⭐ Watchlist</h3>
            <h2>{watchlist_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        search_count = len(st.session_state.get('search_history', []))
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🔍 Searches</h3>
            <h2>{search_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🔥 Trending</h3>
            <h2>Updated</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # TRENDING MOVIES PREVIEW (FIXED: Now clickable)
    st.markdown("### 🔥 Trending This Week")
    
    with st.spinner("🔄 Loading trending movies..."):
        trending = get_trending('week')
    
    if trending:
        cols = st.columns(6)
        for idx, movie in enumerate(trending[:6]):
            with cols[idx]:
                # Display poster
                if movie.get('poster_path'):
                    st.image(Config.IMAGE_BASE + movie['poster_path'],width=300)
                else:
                    st.image(Config.PLACEHOLDER,width=300)
                
                # Display title and rating
                st.caption(f"**{movie['title']}**")
                st.caption(f"⭐ {movie.get('vote_average', 'N/A')}/10")
                
                # FIXED: Details button now works!
                if st.button("View Details", key=f"home_trending_{movie['id']}"):
                    st.session_state.selected_movie_id = movie['id']
                    st.session_state.page = "🎬 Movie Details"
                    st.rerun()
    else:
        st.warning("⚠️ Could not load trending movies. Check your API connection.")
    
    # HOW IT WORKS
    st.markdown("### 🎯 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Search
        Type any movie name. Our smart search understands typos and variations!
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Discover
        Get AI-powered recommendations based on genre, cast, director, and plot similarities
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Explore
        View detailed information, trailers, cast, and similar movies
        """)
    
    with col4:
        st.markdown("""
        #### 4️⃣ Save
        Add favorites to your watchlist for easy access later
        """)
    
    # CALL TO ACTION WITH FUNCTIONAL SEARCH
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>🚀 Ready to Discover Amazing Movies?</h2>", unsafe_allow_html=True)
    
    # Home page search bar now functional!
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        home_search = st.text_input(
            "🔍 Quick Search",
            placeholder="Enter a movie name...",
            key="home_search_input",
            label_visibility="collapsed"
        )
        
        if st.button("🎬 Search Now!", use_container_width=True, key="home_search_btn"):
            if home_search:
                # Store search query and navigate to search page
                st.session_state.search_query = home_search
                st.session_state.page = "🔍 Search & Recommendations"
                st.rerun()
            else:
                st.warning("Please enter a movie name")

# ===============================================================
# 🔍 SEARCH PAGE WITH AUTOCOMPLETE
#      autocomplete suggestions
# ===============================================================
def show_search_page(movies, similarity):
    """
    Display search page with:
    - Search input with autocomplete (FIXED)
    - Number of results slider
    - Movie recommendations
    - Search history
    
    Args:
        movies: DataFrame containing all movies
        similarity: Cosine similarity matrix
        
    FIXES APPLIED:
    - Added autocomplete suggestions while typing
    - Search query from home page is pre-filled
    """
    
    st.title("🔍 Search & Get Recommendations")
    
    # Get search query from session state if coming from home page
    default_query = st.session_state.get('search_query', '')
    
    # Search input with autocomplete
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "🎬 Enter a movie name:",
            placeholder="e.g., Avatar, The Dark Knight, Inception...",
            value=default_query,
            key="main_search_input"
        )
        
        # FIXED: Show autocomplete suggestions
        if search_query and len(search_query) >= 2:
            suggestions = get_movie_suggestions(search_query, movies, limit=5)
            if suggestions:
                st.markdown("**💡 Suggestions:**")
                suggestion_cols = st.columns(len(suggestions))
                for idx, suggestion in enumerate(suggestions):
                    with suggestion_cols[idx]:
                        if st.button(suggestion, key=f"suggestion_{idx}"):
                            st.session_state.search_query = suggestion
                            st.rerun()
    
    # Clear search query from session state after using it
    if 'search_query' in st.session_state:
        del st.session_state.search_query
    
    with col2:
        num_recommendations = st.slider(
            "# Results",
            min_value=6,
            max_value=24,
            value=15
        )
    
    # Search button
    if st.button("🎯 Get Recommendations", use_container_width=True) or (search_query and len(search_query) > 2):
        if not search_query:
            st.warning("⚠️ Please enter a movie name")
        else:
            with st.spinner("🔍 Finding similar movies..."):
                recommendations, posters, error = recommend(
                    search_query, movies, similarity, num_recommendations
                )
                
                if error:
                    st.error(error)
                    
                    # Suggest API search as fallback
                    st.info("💡 Trying online search...")
                    api_results = search_movies_api(search_query)
                    
                    if api_results:
                        st.success(f"✅ Found {len(api_results)} results from TMDB")
                        cols = st.columns(4)
                        for idx, movie in enumerate(api_results[:8]):
                            with cols[idx % 4]:
                                if movie.get('poster_path'):
                                    st.image(Config.IMAGE_BASE + movie['poster_path'],width=300)
                                else:
                                    st.image(Config.PLACEHOLDER,width=300)
                                st.write(f"**{movie['title']}**")
                                st.caption(f"⭐ {movie.get('vote_average', 'N/A')}/10")
                else:
                    # Add to search history
                    if 'search_history' not in st.session_state:
                        st.session_state.search_history = []
                    st.session_state.search_history.append({
                        'query': search_query,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    st.success(f"✅ Found {len(recommendations)} similar movies!")
                    
                    # Display recommendations
                    cols = st.columns(4)
                    for idx, (rec, poster) in enumerate(zip(recommendations, posters)):
                        with cols[idx % 4]:
                            # Highlight the searched movie
                            if rec.get('is_searched', False):
                                st.markdown(f"""
                                <div class='movie-card' style='border: 3px solid #e50914;'>
                                    <div class='movie-title'>🎯 {rec['title']}</div>
                                    <div class='rating-badge'>YOUR SEARCH</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='movie-card'>
                                    <div class='movie-title'>{rec['title']}</div>
                                    <div class='rating-badge'>{rec['similarity']}% Match</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.image(poster,width=300)
                            
                            # Action buttons
                            btn_col1, btn_col2 = st.columns(2)
                            
                            with btn_col1:
                                if st.button("⭐", key=f"save_{rec['movie_id']}_{idx}", help="Add to watchlist"):
                                    if add_to_watchlist(rec['title'], rec['movie_id']):
                                        st.success("✅ Added!")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.info("Already added")
                            
                            with btn_col2:
                                if st.button("ℹ️", key=f"info_{rec['movie_id']}_{idx}", help="View details"):
                                    st.session_state.selected_movie_id = rec['movie_id']
                                    st.session_state.page = "🎬 Movie Details"
                                    st.rerun()
    
    # Show search history
    if st.session_state.get('search_history'):
        with st.expander("📜 Recent Searches"):
            for search in st.session_state.search_history[-10:][::-1]:
                st.text(f"🔍 {search['query']}")

# ===============================================================
# ⭐ WATCHLIST PAGE
#      Now shows movie posters
# ===============================================================
def show_watchlist_page(movies):
    """
    Display user's watchlist with:
    - Movie posters (FIXED: now showing)
    - Movie titles
    - Remove button
    - View details button
    - Export options
    
    Args:
        movies: DataFrame containing all movies
        
    FIXES APPLIED:
    - Watchlist now displays movie posters correctly
    - Better layout with images
    """
    
    st.title("⭐ Your Personal Watchlist")
    
    watchlist = st.session_state.get('watchlist', [])
    
    if not watchlist:
        st.info("📝 Your watchlist is empty. Start adding movies!")
        st.markdown("### How to add movies:")
        st.write("1. Search for movies in the **Search & Recommendations** page")
        st.write("2. Click the ⭐ button to add to watchlist")
        st.write("3. Browse **Trending Now** movies and add your favorites")
    else:
        st.success(f"🎬 You have {len(watchlist)} movies in your watchlist")
        
        # FIXED: Display watchlist with posters in grid layout
        cols = st.columns(4)
        for idx, item in enumerate(watchlist):
            with cols[idx % 4]:
                movie_title = item['title'] if isinstance(item, dict) else item
                movie_id = item.get('movie_id') if isinstance(item, dict) else None
                
                # Get movie poster
                if movie_id:
                    poster_url = fetch_poster(movie_id)
                else:
                    # Try to find movie_id from movies dataframe
                    movie_row = movies[movies['title'] == movie_title]
                    if not movie_row.empty:
                        movie_id = movie_row.iloc[0]['movie_id']
                        poster_url = fetch_poster(movie_id)
                    else:
                        poster_url = Config.PLACEHOLDER
                
                # Display poster
                st.image(poster_url,width=300)
                
                # Display title
                st.markdown(f"**{movie_title}**")
                
                # Added date
                if isinstance(item, dict) and 'added_date' in item:
                    st.caption(f"Added: {item['added_date'][:10]}")
                
                # Action buttons
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    if st.button("🗑️", key=f"remove_{idx}", help="Remove"):
                        remove_from_watchlist(idx)
                        st.success("Removed!")
                        time.sleep(0.3)
                        st.rerun()
                
                with btn_col2:
                    if movie_id:
                        if st.button("ℹ️", key=f"details_{idx}", help="Details"):
                            st.session_state.selected_movie_id = movie_id
                            st.session_state.page = "🎬 Movie Details"
                            st.rerun()
        
        # Export watchlist
        st.markdown("---")
        st.subheader("💾 Export Your Watchlist")
        
        col1, col2 = st.columns(2)
        
        with col1:
            watchlist_text = "\n".join([
                f"{i+1}. {item['title'] if isinstance(item, dict) else item}" 
                for i, item in enumerate(watchlist)
            ])
            st.download_button(
                "📥 Download as TXT",
                watchlist_text,
                file_name="my_watchlist.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            watchlist_json = json.dumps(watchlist, indent=2)
            st.download_button(
                "📥 Download as JSON",
                watchlist_json,
                file_name="my_watchlist.json",
                mime="application/json",
                use_container_width=True
            )

# ===============================================================
# ⚙️ SETTINGS PAGE
#     About section
# ===============================================================
def show_settings_page():
    """
    Display settings page with:
    - API configuration
    - Data management (clear history, clear watchlist)
    - Cache management
    - About section (FIXED: improved content)
    
    FIXES APPLIED:
    - Enhanced About section with better formatting
    - Added version info and features list
    - Improved layout
    """
    
    st.title("⚙️ Settings & About")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    st.info(f"Current API Key: {Config.API_KEY[:10]}...")
    st.caption("To change API key, set TMDB_API_KEY environment variable")
    
    st.markdown("---")
    
    # Data Management
    st.subheader("💾 Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Search History", use_container_width=True):
            st.session_state.search_history = []
            st.success("✅ Search history cleared")
            time.sleep(0.5)
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Watchlist", use_container_width=True):
            st.session_state.watchlist = []
            save_watchlist([])
            st.success("✅ Watchlist cleared")
            time.sleep(0.5)
            st.rerun()
    
    st.markdown("---")
    
    # Cache Management
    st.subheader("📊 Cache Management")
    if st.button("🔄 Clear All Caches", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ All caches cleared")
        time.sleep(0.5)
        st.rerun()
    
    st.markdown("---")
    
    # FIXED: Enhanced About section
    st.subheader("ℹ️ About Movie Recommender Pro")
    
    st.markdown("""
    <div class='info-box'>
        <h2 style='text-align: center;'>🎬 Movie Recommender Pro v3.0 - Original Netflix red and black theme</h2>
        <p style='text-align: center; font-size: 1.1em;'>
            AI-Powered Movie Recommendation System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Core Features:**
        - ✅ AI-Powered Recommendations
        - ✅ Smart Fuzzy Search
        - ✅ Autocomplete Suggestions
        - ✅ Real-time TMDB Integration
        - ✅ Persistent Watchlist
        - ✅ Trending Movies
        """)
    
    with col2:
        st.markdown("""
        **🛠️ Technical Stack:**
        - ✅ Python 3.8+
        - ✅ Streamlit Framework
        - ✅ TMDB API
        - ✅ Scikit-learn (ML)
        - ✅ FuzzyWuzzy (Search)
        - ✅ Pandas (Data Processing)
        """)
    
    # Algorithm explanation
    st.markdown("### 🤖 How It Works")
    st.markdown("""
    <div class='info-box'>
        <p><strong>1. Content-Based Filtering:</strong> Analyzes movie metadata including genres, cast, crew, keywords, and plot</p>
        <p><strong>2. Vectorization:</strong> Converts text data into numerical vectors using CountVectorizer</p>
        <p><strong>3. Cosine Similarity:</strong> Calculates similarity scores between all movies</p>
        <p><strong>4. Recommendation:</strong> Returns top N most similar movies based on your selection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Credits
    st.markdown("### 👨‍💻 Developer")
    st.info("Built with ❤️ using Python and Streamlit")
    
    # Data source
    st.markdown("### 📊 Data Source")
    st.info("Movie data provided by The Movie Database (TMDB) API")
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB")
    
    # Version history
    with st.expander("📝 Version History"):
        st.markdown("""
        **v3.0 (Current)**
        - ✅ Fixed home page movie details clickable
        - ✅ Added search autocomplete suggestions
        - ✅ Fixed watchlist posters display
        - ✅ Improved About section
        - ✅ Better navigation naming
        - ✅ Functional home search bar
        - ✅ Professional code comments
        
        **v2.5**
        - Netflix-inspired UI design
        - Enhanced error handling
        - API retry logic
        - Better fuzzy search
        
        **v2.0**
        - Added trending movies
        - Persistent watchlist
        - Statistics dashboard
        
        **v1.0**
        - Basic recommendation engine
        - Movie search functionality
        """)

# ===============================================================
# 🚀 MAIN APPLICATION
# Entry point and navigation logic
# ===============================================================
def main():
    """
    Main application function
    - Configures Streamlit page settings
    - Loads data
    - Handles navigation
    - Initializes session state
    - Routes to appropriate page based on selection
    """
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="🎬 Movie Recommender Pro",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS styling
    apply_custom_css()
    
    # Load movie data and similarity matrix
    movies, similarity = load_data()
    
    # Check if data loaded successfully
    if movies is None:
        st.error("❌ Failed to load movie data. Please check your data files.")
        st.info("Make sure movie_dict.pkl exists in the current directory")
        st.stop()
        return
    
    # Initialize session state variables
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = load_watchlist()
    
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    if "selected_movie_id" not in st.session_state:
        st.session_state.selected_movie_id = None
    
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # FIXED: Improved navigation labels
    st.sidebar.markdown(
    "<h3 style='font-size:27px; margin-bottom:6px;'>🎬 Navigation</h3>",
    unsafe_allow_html=True
)

    page = st.sidebar.radio(
        "Browse",
        [
            "🏠 Home",
            "🔍 Search & Recommendations",
            "🔥 Trending Now",
            "🎬 Movie Details",
            "⭐ My Watchlist",
            "📊 Statistics",
            "⚙️ Settings & About"
        ],
        index=[
            "🏠 Home",
            "🔍 Search & Recommendations",
            "🔥 Trending Now",
            "🎬 Movie Details",
            "⭐ My Watchlist",
            "📊 Statistics",
            "⚙️ Settings & About"
        ].index(st.session_state.page) if st.session_state.page in [
            "🏠 Home",
            "🔍 Search & Recommendations",
            "🔥 Trending Now",
            "🎬 Movie Details",
            "⭐ My Watchlist",
            "📊 Statistics",
            "⚙️ Settings & About"
        ] else 0
    )
    
    # Update session state page
    st.session_state.page = page
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page(movies)
    
    elif page == "🔍 Search & Recommendations":
        show_search_page(movies, similarity)
    
    elif page == "🔥 Trending Now":
        st.title("🔥 Trending Movies")
        
        # Time window selector
        time_window = st.radio(
            "Select time period:",
            ["Today", "This Week"],
            horizontal=True
        )
        
        period = 'day' if time_window == "Today" else 'week'
        
        with st.spinner("🔄 Loading trending movies..."):
            trending = get_trending(period)
        
        if trending:
            st.success(f"✅ Found {len(trending)} trending movies")
            
            cols = st.columns(4)
            for idx, movie in enumerate(trending):
                with cols[idx % 4]:
                    if movie.get('poster_path'):
                        st.image(Config.IMAGE_BASE + movie['poster_path'],width=300)
                    else:
                        st.image(Config.PLACEHOLDER,width=300)
                    
                    st.markdown(f"**{movie['title']}**")
                    st.caption(f"⭐ {movie.get('vote_average', 'N/A')}/10")
                    st.caption(f"📅 {movie.get('release_date', 'Unknown')[:4]}")
                    
                    # View Details button
                    if st.button("View Details", key=f"trending_{movie['id']}_{idx}"):
                        st.session_state.selected_movie_id = movie['id']
                        st.session_state.page = "🎬 Movie Details"
                        st.rerun()
        else:
            st.error("❌ Could not fetch trending movies")
            st.info("Please check your internet connection and API key")
    
    elif page == "🎬 Movie Details":
        st.title("🎬 Movie Details")
        
        # Check if movie is selected
        if st.session_state.selected_movie_id:
            display_movie_details(st.session_state.selected_movie_id)
            
            # Back button
            if st.button("⬅️ Back"):
                st.session_state.selected_movie_id = None
                st.rerun()
        else:
            st.info("ℹ️ No movie selected. Please select a movie from Search or Trending pages.")
            
            # Quick movie selector
            st.subheader("Or select a movie manually:")
            selected_title = st.selectbox(
                "Choose a movie:",
                movies['title'].values
            )
            
            if st.button("Show Details"):
                movie_id = movies[movies['title'] == selected_title]['movie_id'].values[0]
                st.session_state.selected_movie_id = movie_id
                st.rerun()
    
    elif page == "⭐ My Watchlist":
        show_watchlist_page(movies)
    
    elif page == "📊 Statistics":
        st.title("📊 Dataset Statistics & Analytics")
        show_statistics(movies)
    
    elif page == "⚙️ Settings & About":
        show_settings_page()

# ===============================================================
# 🎯 ENTRY POINT
# Run the application
# ===============================================================
if __name__ == "__main__":
    main()
