# Spotify-like Music Streaming and Recommendation Service

### This repository details the implementation of a music streaming and recommendation service similar to Spotify, utilizing a variety of technologies and datasets for a complete and dynamic user experience.

## Project Overview:

#### The project leverages:
1. Free Music Archive (FMA) for a diverse music dataset.
2. MongoDB for scalable data storage.
3. Apache Spark for efficient large-scale data processing.
4. Apache Kafka for real-time music recommendation.

## Repository Structure:
```py
└── analysis_for_PCA.py # Script for finding the optimal number of PCA components for normalization.
└── feature_extraction.py # Script for extracting audio features like MFCCs, etc and loading extracted features into MongoDB.
├── preprocessing.py # Script for cleaning up tracks metadata for the website.
├── model.py # Script for training music recommendation model with Spark using MinhashLSH and Approximate Nearest Neighbours.
├── app.py # Flask/Django app for the actual music streaming service/
└── producer.py # Script for streaming the dataset using Kafka.
```


## Setup Instructions
### 1. Data Acquisition
  [Download and extract the Free Music Archive (FMA) dataset from here: https://github.com/mdeff/fma](https://github.com/mdeff/fma)

### 2. Feature Extraction and Storage
  #### Process and store music features by running `feature_extraction.py` to extract necessary audio features.

### 3. Model Training and Recommendation System
   Develop and train the recommendation model:
    - Use `model.py` to apply the machine learning algorithms via Apache Spark. </li> 
    - Adjust parameters and algorithms as needed for optimal recommendations. </li> 

### 4. Web Application and Real-Time Recommendations
  Deploy the web application and set up real-time recommendation:
    - Utilize `app.py` to launch a user-friendly music streaming interface. </li> 
    - Run `producer.py` to handle live music streaming based on user activity. </li> 

## Technologies and Challenges:
### Used Technologies:
  - MongoDB: For efficient management of large datasets. </li> 
  - Apache Spark: Utilized for scalable data processing and machine learning. </li> 
  - Apache Kafka: Employs real-time data streaming for dynamic music recommendations. </li> 
  - Python: Primary language for backend and data processing scripts. </li> 
  - Flask/Django: Frameworks for web application development. </li> 

### Implementation Challenges:
  - Data Handling and Processing: Managing large volumes of audio data efficiently. </li> 
  - Real-Time Data Streaming: Implementing a robust system with Apache Kafka for live recommendations. </li> 
  - User Interface Development: Creating an engaging and responsive web interface. </li> 

## Team:
- Manal Aamir: [GitHub](https://github.com/manal-aamir) </li> 
- Mohammad Malik: [GitHub](https://github.com/mohammad-malik) </li> 
- Aqsa Fayaz: [GitHub](https://github.com/Aqsa-Fayyaz) </li> 

