// Global variables
let currentUser = '';
let viewMode = 'popular'; // 'popular', 'search', 'recommendations'
let userRatings = {};
let userReviews = {};

// DOM ready event listener
document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI
    setupEventListeners();
    
    // Load popular movies by default
    getPopular();
});

// Set up event listeners
function setupEventListeners() {
    // Register/login button
    document.getElementById('register-btn').addEventListener('click', registerUser);
    
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            switchTab(this.getAttribute('data-tab'));
        });
    });
    
    // Search button
    document.getElementById('search-btn').addEventListener('click', searchMovies);
    
    // Search input (Enter key)
    document.getElementById('search-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchMovies();
        }
    });
}

// Switch between tabs
function switchTab(tab) {
    // Update UI
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById().classList.add('active');
    
    // Show/hide search box
    document.getElementById('search-box').style.display = tab === 'search' ? 'flex' : 'none';
    
    // Update view mode
    viewMode = tab;
    
    // Load content based on selected tab
    if (tab === 'popular') {
        getPopular();
    } else if (tab === 'search') {
        // Just show the search box, don't perform search yet
        document.getElementById('movie-list').innerHTML = '<p>Enter your search terms above.</p>';
    } else if (tab === 'recommendations') {
        getRecommendations();
    }
}

// Register user
function registerUser() {
    const username = document.getElementById('username').value.trim();
    
    if (!username) {
        alert('Please enter a username');
        return;
    }
    
    showLoading();
    
    fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            alert(data.error);
            return;
        }
        
        currentUser = username;
        alert();
        
        // Update UI to show user is logged in
        document.getElementById('username').disabled = true;
        document.getElementById('register-btn').textContent = 'Logged In';
        document.getElementById('register-btn').disabled = true;
        
        // Update user stats
        updateUserStats();
        
        // Reload current view
        if (viewMode === 'recommendations') {
            getRecommendations();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoading();
        alert('Error registering user. Please try again.');
    });
}

// Get popular movies
function getPopular() {
    showLoading();
    
    fetch('/popular')
    .then(response => response.json())
    .then(data => {
        displayMovies(data.movies);
        hideLoading();
        updateUserStats();
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoading();
        document.getElementById('movie-list').innerHTML = '<p>Error loading movies. Please try again.</p>';
    });
}

// Get recommendations
function getRecommendations() {
    if (!currentUser) {
        alert('Please register first to get personalized recommendations');
        switchTab('popular');
        return;
    }
    
    showLoading();
    
    fetch()
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.error) {
            document.getElementById('movie-list').innerHTML = 
                `<p>Error: ${data.error}</p>
                <p>Try rating some movies first to get better recommendations!</p>`;
            return;
        }
        
        if (data.recommendations.length === 0) {
            document.getElementById('movie-list').innerHTML = 
                '<p>We don\'t have enough data to make recommendations yet.</p>' +
                '<p>Please rate some movies first!</p>';
            return;
        }
        
        displayMovies(data.recommendations);
        updateUserStats();
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoading();
        document.getElementById('movie-list').innerHTML = 
            '<p>Error getting recommendations. Please try again.</p>';
    });
}

// Search for movies
function searchMovies() {
    const query = document.getElementById('search-input').value.trim();
    
    if (!query || query.length < 2) {
        alert('Please enter at least 2 characters to search');
        return;
    }
    
    showLoading();
    
    fetch(`/search?query=${encodeURIComponent(query)}`)
    .then(response => response.json())
    .then(data => {
        viewMode = 'search';
        displayMovies(data.movies);
        hideLoading();
        updateUserStats();
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoading();
        document.getElementById('movie-list').innerHTML = 
            '<p>Error searching movies. Please try again.</p>';
    });
}

// Rate a movie
function rateMovie(movieId, rating) {
    if (!currentUser) {
        alert('Please register first');
        return;
    }
    
    fetch('/rate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            username: currentUser, 
            movie_id: movieId, 
            rating: parseFloat(rating)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // Store rating
        userRatings[movieId] = parseFloat(rating);
        
        // Show success message
        alert(`Rated movie with ${rating} stars!`);
        
        // Update recommendations if we're in recommendation view
        if (viewMode === 'recommendations') {
            getRecommendations();
        } else {
            updateUserStats();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error rating movie. Please try again.');
    });
}

// Submit detailed feedback
function submitFeedback(movieId, rating, review) {
    if (!currentUser) {
        alert('Please register first');
        return;
    }
    
    if (!review.trim()) {
        alert('Please write a review');
        return;
    }
    
    fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            username: currentUser, 
            movie_id: movieId, 
            rating: parseFloat(rating), 
            review 
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // Store rating and review
        userRatings[movieId] = parseFloat(rating);
        userReviews[movieId] = review;
        
        alert('Thank you for your feedback!');
        
        // Update recommendations
        if (viewMode === 'recommendations') {
            getRecommendations();
        } else {
            updateUserStats();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error submitting feedback. Please try again.');
    });
}

// Display movies
function displayMovies(movies) {
    const movieList = document.getElementById('movie-list');
    movieList.innerHTML = '';
    
    if (!movies || movies.length === 0) {
        movieList.innerHTML = '<p>No movies found. Try a different search or rate some movies first!</p>';
        return;
    }
    
    // Add a header based on view mode
    const header = document.createElement('h2');
    header.className = 'section-header';
    
    if (viewMode === 'popular') {
        header.textContent = 'Popular Movies';
    } else if (viewMode === 'search') {
        header.textContent = 'Search Results';
    } else if (viewMode === 'recommendations') {
        header.textContent = 'Recommended For You';
    }
    
    movieList.appendChild(header);
    
    // Create movie grid
    const grid = document.createElement('div');
    grid.className = 'movie-grid';
    movieList.appendChild(grid);
    
    movies.forEach(movie => {
        const card = document.createElement('div');
        card.className = 'movie-card';
        
        const title = document.createElement('h3');
        title.textContent = movie.title_clean || movie.title;
        
        const year = document.createElement('span');
        year.className = 'year';
        year.textContent = movie.year ? `(${movie.year})` : '';
        
        const genres = document.createElement('p');
        genres.className = 'genres';
        genres.textContent = movie.genres;
        
        // Average rating display if available
        let avgRating = null;
        if (movie.avg_rating) {
            avgRating = document.createElement('div');
            avgRating.className = 'avg-rating';
            avgRating.innerHTML = `<span>★</span> ${parseFloat(movie.avg_rating).toFixed(1)} (${movie.num_ratings} ratings)`;
        }
        
        // User rating and feedback section
        const ratingSection = document.createElement('div');
        ratingSection.className = 'rating-section';
        
        const ratingLabel = document.createElement('p');
        ratingLabel.textContent = 'Your Rating:';
        
        const ratingSelect = document.createElement('select');
        ratingSelect.id = `rating-${movie.movieId}`;
        
        for (let i = 0.5; i <= 5; i += 0.5) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `${i} ★`;
            if (i === 5) option.selected = true;
            // If user has already rated this movie, select that rating
            if (userRatings[movie.movieId] === i) {
                option.selected = true;
            }
            ratingSelect.appendChild(option);
        }
        
        const rateButton = document.createElement('button');
        rateButton.className = 'rate-btn';
        rateButton.textContent = userRatings[movie.movieId] ? 'Update Rating' : 'Rate';
        rateButton.onclick = function() {
            rateMovie(movie.movieId, document.getElementById(`rating-${movie.movieId}`).value);
        };
        
        // Feedback section (expanded)
        const feedbackSection = document.createElement('div');
        feedbackSection.className = 'feedback-section';
        
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Leave Feedback';
        details.appendChild(summary);
        
        const textarea = document.createElement('textarea');
        textarea.id = `review-${movie.movieId}`;
        textarea.placeholder = 'Write your review here...';
        if (userReviews[movie.movieId]) {
            textarea.value = userReviews[movie.movieId];
        }
        details.appendChild(textarea);
        
        const submitButton = document.createElement('button');
        submitButton.textContent = 'Submit Feedback';
        submitButton.onclick = function() {
            submitFeedback(
                movie.movieId, 
                document.getElementById(`rating-${movie.movieId}`).value,
                document.getElementById(`review-${movie.movieId}`).value
            );
        };
        details.appendChild(submitButton);
        
        feedbackSection.appendChild(details);
        
        // Assemble the card
        card.appendChild(title);
        if (year) card.appendChild(year);
        card.appendChild(genres);
        if (avgRating) card.appendChild(avgRating);
        
        ratingSection.appendChild(ratingLabel);
        ratingSection.appendChild(ratingSelect);
        ratingSection.appendChild(rateButton);
        
        card.appendChild(ratingSection);
        card.appendChild(feedbackSection);
        
        // If in recommendation mode, add explanation
        if (viewMode === 'recommendations') {
            const explanation = document.createElement('p');
            explanation.className = 'recommendation-reason';
            explanation.textContent = 'Recommended based on your preferences';
            card.appendChild(explanation);
        }
        
        grid.appendChild(card);
    });
}

// Show loading indicator
function showLoading() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('movie-list').style.display = 'none';
}

// Hide loading indicator
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('movie-list').style.display = 'block';
}

// Update user stats
function updateUserStats() {
    if (!currentUser) {
        document.getElementById('user-stats').style.display = 'none';
        return;
    }
    
    const userStats = document.getElementById('user-stats');
    const ratedCount = document.getElementById('rated-count');
    const feedbackCount = document.getElementById('feedback-count');
    
    // Get counts
    const ratingsCount = Object.keys(userRatings).length;
    const reviewsCount = Object.keys(userReviews).length;
    
    // Update UI
    ratedCount.textContent = ratingsCount;
    feedbackCount.textContent = reviewsCount;
    
    // Show stats
    userStats.style.display = ratingsCount > 0 ? 'block' : 'none';
}
