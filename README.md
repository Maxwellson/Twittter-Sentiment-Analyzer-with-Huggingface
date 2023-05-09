# Covid-19 Twitter Sentiment Analysis

## Data collection
The data comes from tweets collected and classified through Crowdbreaks.org [Muller, Martin M., and Marcel Salathe. "Crowdbreaks: Tracking Health Trends Using Public Social Media Data and Crowdsourcing." Frontiers in public health 7 (2019).]. Tweets have been classified as pro-vaccine (1), neutral (0) or anti-vaccine (-1). The tweets have had usernames and web addresses removed.

**Download files**:

**Train.csv** - Labelled tweets on which to train your model

**Test.csv** - Tweets that you must classify using your trained model

**SampleSubmission.csv** - is an example of what your submission file should look like. The order of the rows does not matter, but the names of the ID must be correct. Values in the 'label' column should range between -1 and 1.

**NLP_Primer_twitter_challenge.ipynb** - is a starter notebook to help you make your first submission on this challenge.

**Variable definition:**

**tweet_id:** Unique identifier of the tweet

**safe_tweet:** Text contained in the tweet. Some sensitive information has been removed like usernames and urls

**label:** Sentiment of the tweet (-1 for negative, 0 for neutral, 1 for positive)

**agreement:** The tweets were labeled by three people. Agreement indicates the percentage of the three reviewers that agreed on the given label. You may use this column in your training, but agreement data will not be shared for the test set.

## Data cleaning and preparation
The dataset was downloaded from Zindi website and inspected for data quality. After checks, some missing values were found. However, they were small and dropping them will not affect the dataset hece they were dropped. 
After, the data was manually splitted to have a training subset (a dataset the model will learn on), and an evaluation subset (a dataset the model with use to compute metric scores) to help avoid some training problems like overfitting.

## Development phase
Project activities were conducted on google colab and hugging face. 

**HuggingFace**
Huggingface  was used for setting up website routing and integrating the back end machine learning models with the dashboard.
Furthermore, huggingFace transformers were used for easy text summarization. Transformer models have proven to be exceptionally efficient over a wide range of ML tasks, including Natural Language Processing (NLP), Computer Vision, and Speech.


**Hugging face models fine-tuned**
1. Bert-Based - For sentiment analysis
2. RoBERTa - For sentiment analysis
3. Distillbert - For sentiment triggers extraction.

All models were trained with the same parameters to enable a good comparison for the best model.
Training arguments used were 
training_args = TrainingArguments(
    "twitter_sentiment_analysis_model",           # Directory to save the output files
    evaluation_strategy="epoch",                  # Evaluate the model at specified intervals
    save_strategy="epoch",                         # Save the model at specified intervals
    logging_steps=100,                               # Save the model every 500 steps
    logging_strategy="epoch",
    load_best_model_at_end=True,                  # Load the best model at the end of training
    num_train_epochs=12,                           # Number of epochs for training the model
    per_device_train_batch_size=16,                # Batch size for training
    )

**Evaluation metric**
All models were evaluated using the root mean squared error (RMSE).The roberta model had the lowest RMSE score of 0.59 followed by distillbert, 0.62 and finally bert-based with o.65. Per the results, the roberta performed better with the dataset as it yielded the lowest average difference between the predicted values by the model and the lowest values. Although on a general note, this RMSE score is still high, these project is purely for learning processes hence was used for the remaining project.

## Creating Interactive interface
To make the fine-tuned model more accessible to all, two models with the lowest score were embedded in a gradio and streamlit. The distillbert model was embedded in streamlit app with a customised interface and the roberta model in a gradio app.

# Dockerfile & Running Project Locally

The github repo is provided with a dockerfile and requirements.txt file to recreate the app deployed in the project. The dockerfile creates a virtual environment with required python version and packages for web app deployment. The required Python version must be  3.9. All the dependencies required for the code in the repo can be installed using requirements.txt. 

# Author
Linda Adzigbli

# Links
Medium-https://medium.com/@cnorkplim/covid-19-twitter-sentiment-analysis-bde898d97604
Hugging face-https://huggingface.co/lindaclara22