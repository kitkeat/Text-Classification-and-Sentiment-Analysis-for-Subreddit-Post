# Web APIs & NLP

---

## Goal of the project

---

The objective is to create a model that is able to classify a post to either an Marvel or an DC post. To train the model, we will be using post from the subreddit as the train data. The intention is it as a generalized trained model which will be use outside of reddit.

After classifying the post, the model will also run a sentiment analysis.

The outcome of the analysis serves as a additional supporting document to recommend or to reject the brand.

From the sentiment analysis, I would identify the top words in the positive and negative post.

## Data Dictionary

---

## Data Acquisition:

---
The data was collected using the Pushshift's API. For each subreddit, a total of 10,000 post was collected. The actual total data collected was 9000+ as the API did not give exactly 100 post per request.

## Data Cleaning & EDA:

---

**Natural Langugage Processing**

- Lower cased -> RegexTokenization -> StopWord -> Stemming -> Rejoined -> Demoji -> TFIDVectorizer

- The target has a balance distribution of 0.5/0.5

## Modeling:

---

**Baseline Model: Random Forest**

Decision Tree was used as the baseline model as it is non-parametric and easy to understand.

**Additional Model: Multinomial Naive Bayes & Logistic**

Multinomial Naive Bayes and Logistic Regression were used to compare it's result. It was shown to have a significant improvemenets in the metric.

Confusion matrix was used to describe the performance of the classification model.

## Conclusion

---

## Recommendation

---






