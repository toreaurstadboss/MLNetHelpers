
# ML.NET Helpers

This solution is planned to contain a set of helpers for ML.NET projects 
For now, it contains a sample project that demonstrates how to use ML.NET for sentiment analysis.

## Sentiment Analysis 

This project demonstrates how to use ML.NET for sentiment analysis, which is a common task in natural language processing (NLP). The project includes the following features:
https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis?WT.mc_id=twitter

The sentiment analysis will be trained on a Yelp! review dataset for food reviews from cafes and restaurants.
It will classify the sentiment of the reviews as either positive or negative.
Using Machine Learning, the model will be trained to predict the sentiment of a review based on its text.

Since the sentiment is either Positive or Negative, this is called a binary classification.

Output for project SentimentAnalysis : 

		
```bash
==== Create and train the model ======
==== End of training =====

==== Evaluating Model accuracy with Test data =====

Model quality metrics evaluation:
---------------------------------
Accuracy: 82.89%
Accuracy: 90.11%
Accuracy: 83.67%
========== End of model evaluation =======

=============== Prediction Test of model with a single sample and test dataset ===============

Sentiment: This was a very bad streak. The waiter was however very nice and helpful. | Prediction: Negative | Probability: 0.370712 Score -0.52916354
=============== End of Predictions ===============


 ====== Prediction Test of loaded model with multiple samples =======
Sentiment: This was a horrible meal | Prediction: Negative | Probability: 0.04314197 Score -3.0991588
Sentiment: I love this spaghetti | Prediction: Positive | Probability: 0.99720186 Score 5.87599
===== End of predictions =====
