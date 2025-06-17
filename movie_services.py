# movie_service.py
import requests
import re
from config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL

class MovieService:
    def __init__(self):
        self.api_key = TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.image_base_url = TMDB_IMAGE_BASE_URL
        self.session = requests.Session()
    
    def extract_year(self, title):
        """Extract year from movie title like 'Movie Title (1995)'"""
        match = re.search(r'$$(\d{4})$$', title)
        return match.group(1) if match else None
    
    def clean_title(self, title):
        """Remove year from title for better search results"""
        return re.sub(r'\s*$$\d{4}$$\s*', '', title).strip()
    
    def search_movie(self, title):
        """Search for a movie and return the first result"""
        try:
            clean_title = self.clean_title(title)
            year = self.extract_year(title)
            
            url = f"{self.base_url}/search/movie"
            params = {
                'api_key': self.api_key,
                'query': clean_title,
                'year': year if year else None
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['results']:
                return data['results'][0]
            return None
            
        except Exception as e:
            print(f"Error searching for movie '{title}': {e}")
            return None
    
    def get_movie_poster(self, title):
        """Get movie poster URL"""
        movie_data = self.search_movie(title)
        if movie_data and movie_data.get('poster_path'):
            return f"{self.image_base_url}{movie_data['poster_path']}"
        return None
    
    def get_movie_details(self, title):
        """Get comprehensive movie details"""
        movie_data = self.search_movie(title)
        if movie_data:
            return {
                'title': movie_data.get('title', title),
                'overview': movie_data.get('overview', 'No overview available'),
                'release_date': movie_data.get('release_date', 'Unknown'),
                'vote_average': movie_data.get('vote_average', 0),
                'vote_count': movie_data.get('vote_count', 0),
                'poster_url': f"{self.image_base_url}{movie_data['poster_path']}" if movie_data.get('poster_path') else None,
                'backdrop_url': f"https://image.tmdb.org/t/p/w1280{movie_data['backdrop_path']}" if movie_data.get('backdrop_path') else None
            }
        return None

# Create a global instance
movie_service = MovieService()