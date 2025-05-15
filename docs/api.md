# API Documentation

This document describes the API endpoints of the movie recommendation system.

## Endpoints

### User Registration

- **URL**: `/register`
- **Method**: `POST`
- **Request Body**: 
  ```json
  {
    "username": "string"
  }
  ```
- **Response**: 
  ```json
  {
    "user_id": "number",
    "success": "boolean"
  }
  ```

### Get Popular Movies

- **URL**: `/popular`
- **Method**: `GET`
- **Response**: 
  ```json
  {
    "movies": [
      {
        "movieId": "number",
        "title": "string",
        "genres": "string",
        "avg_rating": "number",
        "num_ratings": "number"
      }
    ]
  }
  ```

### Search Movies

- **URL**: `/search`
- **Method**: `GET`
- **Query Parameters**: `query` (search term)
- **Response**: 
  ```json
  {
    "movies": [
      {
        "movieId": "number",
        "title": "string",
        "genres": "string"
      }
    ]
  }
  ```

### Get Recommendations

- **URL**: `/recommend`
- **Method**: `GET`
- **Query Parameters**: `username` (registered username)
- **Response**: 
  ```json
  {
    "recommendations": [
      {
        "movieId": "number",
        "title": "string",
        "genres": "string"
      }
    ]
  }
  ```

### Rate Movie

- **URL**: `/rate`
- **Method**: `POST`
- **Request Body**: 
  ```json
  {
    "username": "string",
    "movie_id": "number",
    "rating": "number" // 0.5 to 5
  }
  ```
- **Response**: 
  ```json
  {
    "success": "boolean"
  }
  ```

### Submit Feedback

- **URL**: `/feedback`
- **Method**: `POST`
- **Request Body**: 
  ```json
  {
    "username": "string",
    "movie_id": "number",
    "rating": "number", // 0.5 to 5
    "review": "string"
  }
  ```
- **Response**: 
  ```json
  {
    "success": "boolean"
  }
  ```

### Update Models (Admin Only)

- **URL**: `/update-models`
- **Method**: `POST`
- **Request Body**: 
  ```json
  {
    "password": "string" // admin password
  }
  ```
- **Response**: 
  ```json
  {
    "success": "boolean",
    "message": "string"
  }
  ```
