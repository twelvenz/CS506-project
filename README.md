# CS506-project

# Proposal:
This project aims to build a system that predicts the most likely artist(s) to perform at next year's Super Bowl halftime show. Using historical halftime performer data combined with music industry popularity metrics, the project will produce a list of likely candidates for a given year.

# Project timeline:
Week 1-2: Data collection and cleaning

- Download and explore Kaggle datasets (Top Spotify Artists, Past Super Bowl Performers)
- Identify and collect any additional needed data
- Clean and merge datasets


Week 3-4: Exploratory data analysis and feature engineering

- Analyze trends in past performers
- Create relevant features (e.g., genre popularity trends, artist career stage)
- Visualize patterns in the data


Week 5-6: Model development and training

- Test multiple modeling approaches (ex. XGBoost, Time Series Analysis Model)
- Evaluate model performance


Week 7: Model refinement and visualization

- Create interactive visualizations (ex. probabilistic time-series graph)
- Finalize predictions for the upcoming Super Bowl


Week 8: Final report and presentation preparation

# Project Goals:
Develop a model that ranks potential Super Bowl halftime performers for a given year using historical and contemporary music-industry data.
- Build a labeled dataset of past Super Bowl halftime performers with associated artist-level features
- Train a model that ranks the true performer within the top 5 predictions for at least X% of withheld test years
- Identify and analyze the most influential features contributing to performer selection
- (Extra) Create a responsive time-series data interface to predict the top 5 most likely Super Bowl performers with up-to-date Spotify information 

# Data Collection Plan:
The project will rely on two primary datasets:
1. Historical Super Bowl Halftime Performer Data (i.e. "TV, Halftime Shows, and the Big Game"): A dataset containing past Super Bowl halftime performers and event metadata (e.g., year, artist, special guests). This data will be used to label which artists were selected in each year.
2. Spotify Most-Listened Artist Data (i.e. "Spotify Global Music Dataset (2009-2025)"): A publicly available Kaggle dataset containing Spotify streaming statistics such as total streams, popularity scores, and artist-level metadata. This dataset will show artist popularity and mainstream relevance.

Data Collection Methods
- Downloading and versioning datasets directly from Kaggle
- Basic preprocessing and normalization to align artist names across datasets
- Temporal filtering to ensure that only Spotify data available prior to each Super Bowl is used for modeling
- (Extra) Using the Spotify Web API to collect up-to-date data

