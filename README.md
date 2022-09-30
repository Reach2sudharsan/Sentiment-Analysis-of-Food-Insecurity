# **Sentiment Analysis of Food Insecurity With Transfer and Deep Learning**
### **Author:** Sudharsan Gopalakrishnan

</p>
<p align="left">
<a href="https://www.tweepy.org/">
    <img src="https://img.shields.io/badge/powered%20by-tweepy-blue" alt="Test Status">
</a>
<a href="https://github.com/keras-team/keras">
    <img src="https://img.shields.io/badge/powered%20by-keras-brightgreen" alt="Test Status">
</a>
<a href="https://matplotlib.org">
    <img src="https://img.shields.io/badge/powered%20by-matplotlib-brightgreen" alt="Package Version">
</a>
<a href="https://www.nltk.org">
    <img src="https://img.shields.io/badge/powered%20by-nltk-brightgreen" alt="Package Version">
</a>
<a href="https://pypi.org/project/wordcloud/">
    <img src="https://img.shields.io/badge/powered%20by-wordcloud-brightgreen" alt="Package Version">
</a>
</p>

## Research Project
This research revolves around tackling the issue of food insecurity by analyzing related tweets through Sentiment Analysis using Transfer and Deep Learning.

***Using:*** 
- *Transfer Learning: IMDb movie reviews (training data) --> Food Insecurity tweets (testing data)*
- *Deep Learning: BiLSTM (Bidirectional Long Short Term Memory) Model*

## Methods

### Data Collection
I mined 1558 tweets from Twitter using the Python <a href="https://www.tweepy.org/">Twitter API</a> called tweepy, which I classifed the sentiment of (positive or negative). I used Deep Learning for this research, so I would need a lot of training data for the model that I used. I chose to use 50000 IMDb movie reviews as my training data.

### BiLSTM
I used a BiLSTM (Bidirectional Long Short Term Memory) model in order to classify the sentiment. The model's architecture is displayed in the below diagrams. Through research, unlike the standard LSTM approach, a BiLSTM model involves its inputs flowing in both directions and is thus capable of using information from both of its sides.
###
<img src="figures/model_pics/BiLSTM_model.png" width=500, height=400>

###
Using transfer learning, I trained this model with the IMDb data and tested it with the Food Insecurity Twitter data.

## Link to Research Paper
https://jsr.org/preprints/scholarlaunch/preprint/view/171


