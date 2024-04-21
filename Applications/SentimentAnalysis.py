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
from matplotlib.dates import DateFormatter

api_key = "AIzaSyAtD0eBdYNMR1crlbNjCc_exLWfDCafxQY"
youtube = build("youtube", "v3", developerKey=api_key)

Videourl = input("Please enter a video url: ")
VideourlSplit = Videourl.split("=")
video_id = VideourlSplit[1]

#https://www.youtube.com/watch?v=reUZRyXxUs4
#https://www.youtube.com/watch?v=QOCZYRXL0AQ

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

#Comments are preprocessed
def preprocess_comments(comments):
    preprocessed_comments = []
    for comment in comments:
        comment = re.sub(r'[^A-Za-z0-9 ]+', '', comment).lower()
        preprocessed_comments.append(comment)
    return preprocessed_comments

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
views = statistics.get('viewCount', 'Not available')
likes = statistics.get('likeCount', 'Not available')
dislikes = statistics.get('dislikeCount', 'Not available')
comment_count = statistics.get('commentCount', 'Not available')

comments, timestamps = get_all_english_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')


preprocessed_comments = preprocess_comments(comments)


vader_analyzer = SentimentIntensityAnalyzer()
vader_sentiments = [vader_analyzer.polarity_scores(comment) for comment in preprocessed_comments]


afinn = Afinn()
afinn_sentiments = [afinn.score(comment) for comment in preprocessed_comments]


textblob_sentiments = [TextBlob(comment).sentiment.polarity for comment in preprocessed_comments]


data = {
    'Timestamps': timestamps,
    'Comments': comments,
    'VADER Compound': [s['compound'] for s in vader_sentiments],
    'AFINN Score': afinn_sentiments,
    'TextBlob Polarity': textblob_sentiments
}


df = pd.DataFrame(data)


tfidf_vectorizer = TfidfVectorizer(max_features=20) 


tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_comments)


feature_names = tfidf_vectorizer.get_feature_names_out()


mean_tfidf_scores = np.mean(tfidf_matrix, axis=0).A1


tfidf_df = pd.DataFrame({'Word': feature_names, 'Mean TF-IDF Score': mean_tfidf_scores})


tfidf_df = tfidf_df.sort_values(by='Mean TF-IDF Score', ascending=False)


print("Top 20 Words by Mean TF-IDF Score:")
print(tfidf_df[['Word', 'Mean TF-IDF Score']][:20])


def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


df['VADER Sentiment'] = df['VADER Compound'].apply(categorize_sentiment)
df['AFINN Sentiment'] = df['AFINN Score'].apply(categorize_sentiment)
df['TextBlob Sentiment'] = df['TextBlob Polarity'].apply(categorize_sentiment)


vader_sentiment_counts = df['VADER Sentiment'].value_counts()


afinn_sentiment_counts = df['AFINN Sentiment'].value_counts()


textblob_sentiment_counts = df['TextBlob Sentiment'].value_counts()



plt.figure(figsize=(8, 5))
labels = ['Positive', 'Neutral', 'Negative']
vader_sizes = [vader_sentiment_counts.get(sentiment, 0) for sentiment in labels]
afinn_sizes = [afinn_sentiment_counts.get(sentiment, 0) for sentiment in labels]
textblob_sizes = [textblob_sentiment_counts.get(sentiment, 0) for sentiment in labels]
plt.suptitle(f'Positive and negative comment distribution for video: {video_title}')

explode = (0.1, 0, 0)  

# VADER
plt.subplot(1, 3, 1)
plt.pie(vader_sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('VADER Sentiment Distribution')

# AFINN
plt.subplot(1, 3, 2)
plt.pie(afinn_sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('AFINN Sentiment Distribution')

# TextBlob
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


plt.xlabel('Sentiment Category')
plt.ylabel('Number of Comments')
plt.xticks([i + width for i in x], sentiments)
plt.title(f'Sentiment Analysis Comparison for Video {video_title}')
plt.legend()


plt.figure(figsize=(12, 6))
plt.barh(tfidf_df['Word'][:20], tfidf_df['Mean TF-IDF Score'][:20], color='lightblue')
plt.title('Top 20 Words by Mean TF-IDF Score')
plt.xlabel('Mean TF-IDF Score')
plt.ylabel('Word')
plt.gca().invert_yaxis()
plt.show()

published_at = published_at.replace("T", " " )
published_at = published_at.replace("Z", " " )
print("\n")
print("Video Title:", video_title)
print("Channel Title:", channel_title)
print("Published At:", published_at)
print("\n")
print("Description:", video_description)
print("\n")
print("Views:", views)
print("Likes:", likes)
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
print("This video has a comment to view ratio of " + CommenttoLikeRatioFigure + ":1000 or " + CommenttoLikeRatio + "%")




