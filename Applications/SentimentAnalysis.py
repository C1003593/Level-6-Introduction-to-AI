from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import langid
import re
from afinn import Afinn  
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Api key to search Youtube API
api_key = "AIzaSyAtD0eBdYNMR1crlbNjCc_exLWfDCafxQY"
youtube = build("youtube", "v3", developerKey=api_key)

#Checks for the video entered
Videourl = input("Please enter a video url: ")
VideourlSplit = Videourl.split("=")
video_id = VideourlSplit[1]

#2 sample videos
#https://www.youtube.com/watch?v=reUZRyXxUs4
#https://www.youtube.com/watch?v=QOCZYRXL0AQ

#This function gets all the Youtube comments
def get_all_english_video_comments(youtube, **kwargs):
    comments = []
    timestamps = []  
    while True:
        results = youtube.commentThreads().list(**kwargs).execute()
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            timestamp = item['snippet']['topLevelComment']['snippet']['publishedAt']
            lang, _ = langid.classify(comment)
            if lang == 'en':
                comments.append(comment)
                timestamps.append(timestamp)
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
        else:
            break
    return comments, timestamps

#The comments are preprocessed, things such as punctuation are removed
def preprocess_comments(comments):
    preprocessed_comments = []
    for comment in comments:
        comment = re.sub(r'[^A-Za-z0-9 ]+', '', comment).lower()
        preprocessed_comments.append(comment)
    return preprocessed_comments

#This gets the metadata about the video
def get_video_details(youtube, video_id):
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()
    return response['items'][0]

video_details = get_video_details(youtube, video_id)


video_title = video_details['snippet']['title']
video_description = video_details['snippet']['description']
channel_title = video_details['snippet']['channelTitle']
published_at = video_details['snippet']['publishedAt']

statistics = video_details.get('statistics', {})
#This ensures that if one of the stats are not available the program will say
views = statistics.get('viewCount', 'Not available')
likes = statistics.get('likeCount', 'Not available')
dislikes = statistics.get('dislikeCount', 'Not available')
comment_count = statistics.get('commentCount', 'Not available')

comments, timestamps = get_all_english_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')


preprocessed_comments = preprocess_comments(comments)

#This starts the vader analyser
vader_analyzer = SentimentIntensityAnalyzer()
vader_sentiments = [vader_analyzer.polarity_scores(comment) for comment in preprocessed_comments]

#This starts the afinn analyser
afinn = Afinn()
afinn_sentiments = [afinn.score(comment) for comment in preprocessed_comments]

#This starts the textblob analyser
textblob_sentiments = [TextBlob(comment).sentiment.polarity for comment in preprocessed_comments]


data = {
    'Timestamps': timestamps,
    'Comments': comments,
    'VADER Compound': [s['compound'] for s in vader_sentiments],
    'AFINN Score': afinn_sentiments,
    'TextBlob Polarity': textblob_sentiments
}

#This loads the data into a pandas dataframe
df = pd.DataFrame(data)

#This gets the 20 most used words
tfidf_vectorizer = TfidfVectorizer(max_features=20) 

#Term Frequency - Inverse Document Frequency, this decides how important a word is, the score is between 0 and 1
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_comments)


feature_names = tfidf_vectorizer.get_feature_names_out()


mean_tfidf_scores = np.mean(tfidf_matrix, axis=0).A1

#This loads the top 20 words into a dataframe along with their mean_tfidf scores
tfidf_df = pd.DataFrame({'Word': feature_names, 'Mean TF-IDF Score': mean_tfidf_scores})


tfidf_df = tfidf_df.sort_values(by='Mean TF-IDF Score', ascending=False)

#This prints the top 20 words determined by the highest tfidf scores
print("Top 20 Words by Mean TF-IDF Score:")
print(tfidf_df[['Word', 'Mean TF-IDF Score']][:20])

#This decides if something is positive, neutral or negative based on the score
def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

#This makes all 3 analysers analyse the sentiments of the comments
df['VADER Sentiment'] = df['VADER Compound'].apply(categorize_sentiment)
df['AFINN Sentiment'] = df['AFINN Score'].apply(categorize_sentiment)
df['TextBlob Sentiment'] = df['TextBlob Polarity'].apply(categorize_sentiment)


vader_sentiment_counts = df['VADER Sentiment'].value_counts()


afinn_sentiment_counts = df['AFINN Sentiment'].value_counts()


textblob_sentiment_counts = df['TextBlob Sentiment'].value_counts()


#This plots the negative, neutral and positive sentiment distribution for the comments
plt.figure(figsize=(8, 5))
labels = ['Positive', 'Neutral', 'Negative']
vader_sizes = [vader_sentiment_counts.get(sentiment, 0) for sentiment in labels]
afinn_sizes = [afinn_sentiment_counts.get(sentiment, 0) for sentiment in labels]
textblob_sizes = [textblob_sentiment_counts.get(sentiment, 0) for sentiment in labels]
plt.suptitle(f'Positive and negative comment distribution for video: {video_title}')

explode = (0.1, 0, 0)  

#Vader sentiment distribution
plt.subplot(1, 3, 1)
plt.pie(vader_sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('VADER Sentiment Distribution')

#AFINN sentiment distribution
plt.subplot(1, 3, 2)
plt.pie(afinn_sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('AFINN Sentiment Distribution')

#TextBlob sentiment distribution
plt.subplot(1, 3, 3)
plt.pie(textblob_sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('TextBlob Sentiment Distribution')

plt.tight_layout()
plt.show()

print("VADER Sentiment Summary:")
print(vader_sentiment_counts)


print("\nAFINN Sentiment Summary:")
print(afinn_sentiment_counts)


print("\nTextBlob Sentiment Summary:")
print(textblob_sentiment_counts)


df = df.iloc[::-1]

#This plots the sentiment polarity over time (How positive/negative/neutral the video comments have been)
plt.figure(figsize=(12, 6))
plt.plot(df['Timestamps'], df['VADER Compound'], label='VADER', color='lightblue')
plt.plot(df['Timestamps'], afinn_sentiments, label='AFINN', color='lightcoral')
plt.plot(df['Timestamps'], textblob_sentiments, label='TextBlob', color='lightgreen')


num_ticks = 8  
x_ticks = np.linspace(0, len(df['Timestamps']) - 1, num_ticks, dtype=int)
plt.xticks(x_ticks, [df['Timestamps'].iloc[i] for i in x_ticks], rotation=45)

plt.title(f'Sentiment Analysis Over Time for video: {video_title}')
plt.xlabel('Timestamp')
plt.ylabel('Sentiment Polarity Score')
plt.legend()

plt.tight_layout()
plt.show()

#This shows how many of each type of comment there are
plt.figure(figsize=(10, 6))
sentiments = ['Positive', 'Neutral', 'Negative']
vader_counts = [vader_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
afinn_counts = [afinn_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
textblob_counts = [textblob_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]


width = 0.2
x = range(len(sentiments))


plt.bar(x, vader_counts, width, label='VADER', align='center')
plt.bar([i + width for i in x], afinn_counts, width, label='AFINN', align='center')
plt.bar([i + width * 2 for i in x], textblob_counts, width, label='TextBlob', align='center')

#This sets the labels
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Comments')
plt.xticks([i + width for i in x], sentiments)
plt.title(f'Sentiment Analysis Comparison for Video {video_title}')
plt.legend()

#This plots the Top 20 words by TF-IDF score
plt.figure(figsize=(12, 6))
plt.barh(tfidf_df['Word'][:20], tfidf_df['Mean TF-IDF Score'][:20], color='lightblue')
plt.title('Top 20 Words by Mean TF-IDF Score')
plt.xlabel('Mean TF-IDF Score')
plt.ylabel('Word')
plt.gca().invert_yaxis()
plt.show()

#This Filters the published_at number to not include confusing figures
published_at = published_at.replace("T", " " )
published_at = published_at.replace("Z", " " )
print("\n")
#This displays the metadata about the video
print("Video Title:", video_title)
print("Channel Title:", channel_title)
print("Published At:", published_at)
print("\n")
print("Description:", video_description)
print("\n")
print("Views:", views)
print("Likes:", likes)
#Youtube disabled dislikes but this would work if ever reenabled
print("Dislikes:", dislikes)
print("Comment count:", comment_count)
print("\n")

#https://viralyft.com/blog/youtube-like-to-view-ratio#:~:text=Marketing%20specialists%20often%20suggest%20aiming,likes%20for%20every%201%2C000%20views. Source on 4% figure
print("An average video has the like to view ratio of 4:100 or 4%")
viewsint = int(views)
likesint = int(likes)
ViewtoLikeRatio = (likesint/viewsint)*100
ViewtoLikeRatio = round(ViewtoLikeRatio, 1)
ViewtoLikeRatio = str(ViewtoLikeRatio)
#This shows how many likes to views the video has
print("This video has a like to view ratio of " + ViewtoLikeRatio + ":100 or " + ViewtoLikeRatio + "%")
print("\n")

#https://blog.promolta.com/how-to-measure-your-viewer-engagement#:~:text=Golden's%20ratio%20for%20comments%20to,two%20hundred%20viewers%20will%20comment. Source on 0.5% figure on comments
print("An average video has the comment to view ratio of 5:1000 or 0.5%")
commentsint = int(comment_count)
CommenttoLikeRatio = (commentsint/viewsint)*100
CommenttoLikeRatio = round(CommenttoLikeRatio, 2)
CommenttoLikeRatioFigure = CommenttoLikeRatio*10
CommenttoLikeRatioFigure = str(CommenttoLikeRatioFigure)
CommenttoLikeRatio = str(CommenttoLikeRatio)
#This shows how many comments to views the video has
print("This video has a comment to view ratio of " + CommenttoLikeRatioFigure + ":1000 or " + CommenttoLikeRatio + "%")




