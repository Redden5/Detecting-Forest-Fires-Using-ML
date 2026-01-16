# Detecting-Forest-Fires-Using-ML
Project completed in conjunction the Undergraduate Creative & Discovery Research Award from Marshall University, under the guidence of Dr. Husnu Narman.

Early Detection of Forest Fires Using Machine Learning

Objectives:
- Analyze historical wildfire and environmental datasets.
- Engineer meaningful features related to fire risk.
- Train, evaluate, and compare different machine learning models for different aspects of wildfire prediction.
- Identify limitations and future research directions.

Data Sources
- Historical wildfire perimeters and occurences.
- Weather and climate variables (temperature, humidity, and precipitation).
- Geospatial information (location, elevation, region).
- Large raw datasets are intentionally excluded from this repository and must be downloaded seperately.

Methodolgy
- Data Cleaning and Processing
- Feature Engineering
    - Environmental risk indicators
    - Temporal trends
    - Spatial features
- Modeling
    - Supervised machine learning models
    - Train/Evaluate/Test
- Evaluation
    - Accuracy
    - Percision / Recall
    - F1-score
    -ROC-AUC
-Technology Used
    - Python
    - Numpy / Pandas
    - Scikit-learn
    - Matplotlib
    - Geospatial tools (GeoJSON)

Results 
- The models demonstrate that wildfire risk can be reasonably predicted under certain environmental conditions, with ensemble methods generally outperforming simpler baselines.

Key findings include:
- Strong correlation between fire occurrence and temperature, humidity, and vegetation dryness.
- Improved performance with engineered temporal and spatial features
- Tradeoffs between recall (detecting fires) and false positives.
