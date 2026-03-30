# CS506-project

# Proposal:
This project aims to build a system that predicts the ranks of the most likely artist(s) to perform at next year's Super Bowl halftime show. Our goal is to identify artists who best fit the measurable profile of past halftime headliners using music industry metrics. Using historical halftime performer data combined with Billboard chart history and Spotify streaming statistics, the project will produce a ranked shortlist of candidates for a given year. Our focus will be primarily on Billboard data and Spotify data as the core sources for driving these rankings.

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
- Build a labeled dataset of past Super Bowl halftime performers (post-2019 because thats when Roc Nation took over) with associated artist-level features, treating all top-charting artists each year as candidates
- Train a model that ranks the true performer within the top 5 predictions for at least X% of withheld test years
- Identify and analyze the most influential features contributing to performer selection
- (Extra) Create a responsive time-series data interface to predict the top 5 most likely Super Bowl performers with up-to-date Spotify information 

# Data Collection Plan:
The project will rely on two primary datasets:
1. Historical Super Bowl Halftime Performer Data (i.e. "TV, Halftime Shows, and the Big Game"): A dataset containing past Super Bowl halftime performers and event metadata (e.g., year, artist, special guests). This data will be used to label which artists were selected in each year. We limit this to post-2019 Super Bowls to exclude the marching band era and non-Roc Nation era to focus on mainstream headliners.
2. Spotify Most-Listened Artist Data (i.e. "Spotify Global Music Dataset (2009-2025)"): A publicly available Kaggle dataset containing Spotify streaming statistics such as total streams, popularity scores, and artist-level metadata. This dataset will show artist popularity and mainstream relevance.

To address the limited number of Super Bowl events, we expand our dataset by treating every top-charting artist in a given year as a candidate, labeling whether they performed at the halftime show or not. This gives us thousands of artist-year observations, giving us a much more workable training set. Billboard chart data, which extends back decades, serves as our primary historical signal, while Spotify data supplements the more recent years.
Our focus for this project is primarily on Billboard data and Spotify data to predict the next Super Bowl halftime performer. We spent a significant amount of time collecting, filtering, and verifying the Billboard data to ensure it accurately reflects what is represented on the Billboard website. 
Our sources came from free sites, but we still cross-checked what we could against Billboard directly. We then converted the data into CSV format to facilitate data exploration. Through this process, we found that Super Bowl halftime headliners typically have top-charting songs and albums, which will be a crucial factor when we build our ranking system.

Data Collection Methods
- Downloading and versioning datasets directly from Kaggle
- Basic preprocessing and normalization to align artist names across datasets
- Temporal filtering to ensure that only Spotify data available prior to each Super Bowl is used for modeling
- Using the Spotify Web API to collect up-to-date data

# Modeling Approach:
We are still deciding on modeling approach. We have considered XGBoost, but we need more time to decide.

# Data Visualization Plan:
We have planned time-series plots to examine artist popularity trends over time and summary charts to compare characteristics of past performers.

# Test Plan:
Data from earlier Super Bowl years will be used for training, while a subset of more recent years will be withheld for testing. The model will be scored on how well it ranks the correct halftime performer near the top.
