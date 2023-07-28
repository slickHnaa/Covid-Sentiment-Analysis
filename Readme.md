# 💉💉💉 Vaccination Sentiment Analysis 💉💉💉

This project aims at developing a machine learning model to assess the sentiment (positive, neutral, or negative) of Twitter posts related to covid vaccinations. 

## Summary
| Code      | Name        | Published Article |  Deployed App |
|-----------|-------------|:-------------:|------:|
| LP5 | Fine Tuning Pretrained From HuggingFace |  [ARTICLE](https://medium.com/@hnayiteyadjin/sentimental-analysis-of-covid-vaccines-by-utilizing-pretrained-models-from-huggingface-a7e1e73152b) | [APP](slickdata/finetuned-Sentiment-classfication-ROBERTA-model-App) |


## Content of Repo
- **App** contains app.py file of best fine tuned pretrained model
- **EDA** contains notebook of EDA of dataset
- **Models** contains notebooks of three fine tuned pretrained models


## Setup
### Objective

- The main goal is to build a machine learning model that can accurately predict the sentiment of covid vaccination-related tweets.


### Dataset

- The dataset consists of tweets collected and classified through [![Dataset](https://img.shields.io/badge/Dataset-Crowdbreaks.org-blue)](https://www.crowdbreaks.org/).

- Tweets have been labeled as positive (👍), neutral (🤐), or negative(👎).

- Usernames and web addresses have been removed for privacy reasons.
**Variable definition:**

| Feature    | Meaning                                                                               |
|------------|---------------------------------------------------------------------------------------|
| tweet_id   | Unique identifier of the tweet                                                        |
| safe_tweet | Text contained in the tweet. Sensitive information (usernames, URLs) removed          |
| label      | Sentiment of the tweet (-1 for negative, 0 for neutral, 1 for positive)               |
| agreement  | Percentage of agreement among the three reviewers for the given label                 |

## Modelling 
### Fine-tuning of Pre-trained Models 
- Utilized pre-trained models from the Hugging Face library for Natural Language Processing (NLP) tasks.

- Models used: 
  - ROBERTA: "slickdata/finetuned-Sentiment-classfication-ROBERTA-model"
  - BERT: "slickdata/finetuned-Sentiment-classfication-BERT-model"
  - DISTILBERT: "slickdata/finetuned-Sentiment-classfication-DistilBert-model"

- Models have been fine-tuned on sentiment classification, suitable for analyzing sentiment in vaccination-related tweets.

- Leveraged pre-trained models' contextual understanding of language and ability to extract meaningful representations from text.

- During the modeling process, the selected model was loaded based on the user's choice using the corresponding identifier from the Hugging Face library.

- Performed parameter tuning and fine-tuning to optimize the performance of the selected models.

- Adjusted hyperparameters (learning rate, batch size, epochs) to achieve the best results.

- Goal: Accurately classify tweets into positive, neutral, or negative sentiment categories related to COVID-19 vaccinations.

### GPU Acceleration with Google Colab
- Utilized GPUs (Graphics Processing Units) for enhanced performance and faster model training.

- GPUs are efficient in handling parallel computations, accelerating deep learning tasks.

- Leveraged Google Colab with free GPU resources for efficient model training and evaluation.

- Google Colab provided a cost-effective solution for leveraging GPU acceleration.

- Note: Google Colab sessions have time limits and may disconnect after inactivity. Save progress and re-establish connection when needed.

## Evaluation
The evaluation metric for this project is the Root Mean Squared Error (RMSE), a commonly used metric for regression tasks. 

Here are the key points about the RMSE evaluation metric:

- RMSE is calculated as the square root of the mean of the squared differences between predicted and actual values.
- It provides a single value representing the overall model performance, with lower values indicating better accuracy.
- The RMSE metric allows for easy interpretation of prediction errors in the same unit as the target variable.

## Deployment
To deploy the model, follow these steps outlined here 

```bash
https://github.com/slickHnaa/Covid-Sentiment-Analysis.git
```
To use the deployed app visit:

```bash
slickdata/finetuned-Sentiment-classfication-ROBERTA-model-App
```


## Author
`Henry Nii Ayitey-Adjin` 

`Data Analyst`

`Azubi Africa`
