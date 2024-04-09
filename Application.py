import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import langid
import re
from afinn import Afinn  
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
import pygad

def URLCHECK():
    print ("This option will analyse a video based on the url entered.")

    api_key = "AIzaSyAtD0eBdYNMR1crlbNjCc_exLWfDCafxQY"
    youtube = build("youtube", "v3", developerKey=api_key)

    Videourl = input("Please enter a video url: ")
    VideourlSplit = Videourl.split("=")
    video_id = VideourlSplit[1]


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


    def preprocess_comments(comments):
        preprocessed_comments = []
        for comment in comments:
            comment = re.sub(r'[^A-Za-z0-9 ]+', '', comment).lower()
            preprocessed_comments.append(comment)
        return preprocessed_comments


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


    print("VADER Sentiment Summary:")
    print(vader_sentiment_counts)


    print("\nAFINN Sentiment Summary:")
    print(afinn_sentiment_counts)


    print("\nTextBlob Sentiment Summary:")
    print(textblob_sentiment_counts)


    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamps'], df['VADER Compound'], label='VADER', color='lightblue')
    plt.plot(df['Timestamps'], afinn_sentiments, label='AFINN', color='lightcoral')
    plt.plot(df['Timestamps'], textblob_sentiments, label='TextBlob', color='lightgreen')
    plt.title(f'Sentiment Analysis Over Time for Video {video_id}')
    plt.xlabel('Timestamp')
    plt.ylabel('Sentiment Polarity Score')
    plt.legend()
    plt.xticks(rotation=45)


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
    plt.title(f'Sentiment Analysis Comparison for Video {video_id}')
    plt.legend()


    plt.figure(figsize=(12, 6))
    plt.barh(tfidf_df['Word'][:20], tfidf_df['Mean TF-IDF Score'][:20], color='lightblue')
    plt.title('Top 20 Words by Mean TF-IDF Score')
    plt.xlabel('Mean TF-IDF Score')
    plt.ylabel('Word')
    plt.gca().invert_yaxis()
    plt.show()

    #Make sure to display each method's accuracy.
    #Make sure to display overall positive and negative score
    #Make sure to display the most used positive word and most used negative word

def Pandas(): #https://www.kaggle.com/datasets/cedricaubin/ai-ml-salaries
    print ("This option will analysis the salaries dataset.")
    salary_data = pd.read_csv("salaries.csv")
    print (salary_data.shape)
    salary_data.salary_in_usd.hist(figsize=(10,5), grid=False, color="#FC5864", ec="black", bins=10).set_title('salary')
    plt.xlabel('Salary (USD)')
    plt.ylabel('Frequency')

        
    min_salary = salary_data['salary_in_usd'].min()
    max_salary = salary_data['salary_in_usd'].max()

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    for i, (experience_level, group) in enumerate(salary_data.groupby('experience_level')):
        row = i // 2
        col = i % 2
        axs[row, col].set_xlabel('Salary (USD)')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].grid(False)
        
        if experience_level == "SE":
            group['salary_in_usd'].hist(ax=axs[row, col], color="#AAFC58", ec="black", bins=20, grid=False, range=(min_salary, max_salary))
            axs[row, col].set_title(f'Salary Distribution for senior level position')
        elif experience_level == "EN":
            group['salary_in_usd'].hist(ax=axs[row, col], color="#58A3FC", ec="black", bins=20, grid=False, range=(min_salary, max_salary))
            axs[row, col].set_title(f'Salary Distribution for entry level position')
        elif experience_level == "MI":
            group['salary_in_usd'].hist(ax=axs[row, col], color="#BE58FC", ec="black", bins=20, grid=False, range=(min_salary, max_salary))
            axs[row, col].set_title(f'Salary Distribution for intermediate position')
        elif experience_level == "EX":
            group['salary_in_usd'].hist(ax=axs[row, col], color="#FCCF58", ec="black", bins=20, grid=False, range=(min_salary, max_salary))
            axs[row, col].set_title(f'Salary Distribution for director position')

    if len(salary_data['experience_level'].unique()) < 4:
        axs[-1, -1].axis('off')

    plt.tight_layout()

    
    
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    for i, (remote_ratio, group) in enumerate(salary_data.groupby('remote_ratio')):
        row = i // 2
        col = i % 2
        axs[row, col].set_xlabel('Salary (USD)')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].grid(False)
        
        if remote_ratio == 0:
            group['salary_in_usd'].hist(ax=axs[row, col], color="#3182FF", ec="black", bins=20, grid=False, range=(min_salary, max_salary))
            axs[row, col].set_title(f'Salary Distribution for workers who are 0% remote')
        elif remote_ratio == 50:
            group['salary_in_usd'].hist(ax=axs[row, col], color="#B81A3C", ec="black", bins=20, grid=False, range=(min_salary, max_salary))
            axs[row, col].set_title(f'Salary Distribution for workers who are 50% remote')
        elif remote_ratio == 100:
            group['salary_in_usd'].hist(ax=axs[row, col], color="#0BE59A", ec="black", bins=20, grid=False, range=(min_salary, max_salary))
            axs[row, col].set_title(f'Salary Distribution for workers who are 100% remote')

    if len(salary_data['remote_ratio'].unique()) < 4:
        axs[-1, -1].axis('off')

    plt.tight_layout()
    plt.show()
    
def csv_writer(): 
    print("This option will make a randomised data file.")
    with open('file.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'age', 'gender', 'relationship_status', 'region', 'hair_colour', 'eye_colour', 'height(cm)'])
        y = 0
        z = input("PLease choose the amount of records you'd like to generate: ")
        z = int(z)
        while y < z:
            age = random.randint(16, 116)
            
            randrel = random.randint(1, 3)
            if randrel == 1:
                relationship = "Single"
            elif randrel == 2:
                relationship = "Married"
            elif randrel == 3: 
                relationship = "Divorced"
            
            randgen = random.randint(1, 101)
            if randgen < 50:
                gender = "M"
            elif randgen < 101 and randgen > 51:
                gender = "F"
            elif randgen == 101:
                gender = "/"
                
            randreg = random.randint(1, 6)
            if randreg == 1:
                region = "Europe"
            elif randreg == 2:
                region = "Asia"
            elif randreg == 3:
                region = "Africa"
            elif randreg == 4:
                region = "Oceania"
            elif randreg == 5:
                region = "North America"
            elif randreg == 6:
                region = "South America"
            
            randhair = random.randint(1, 5)
            if randhair == 1:
                hair = "Blonde"
            elif randhair == 2:
                hair = "Black"
            elif randhair == 3:
                hair = "Red"
            elif randhair == 4:
                hair = "Grey"
            elif randhair == 5:
                hair = "Bald"
                
            randeye = random.randint(1, 4)
            if randeye == 1:
                eye = "Blue"
            elif randeye == 2:
                eye = "Green"
            elif randeye == 3:
                eye = "Grey"
            elif randeye == 4:
                eye = "Brown"
            
            if gender == "M":
                height = random.randint(157, 196)
            elif gender == "F":
                height = random.randint(142, 188)
            else:
                height = random.randint(142, 196)
                
            firstrand = random.randint(1, 25)
            lastrand = random.randint(1, 25)
            
            firstrand = random.randint(1, 25)
            if gender == "M" or gender == "/":
                match firstrand: 
                    case 1:
                        fname = "John"
                    case 2:
                        fname = "David"
                    case 3:
                        fname = "Adam"
                    case 4:
                        fname = "Ken"
                    case 5:
                        fname = "Matthew"
                    case 6:
                        fname = "Luke"
                    case 7:
                        fname = "Leo"
                    case 8:
                        fname = "Jeff"
                    case 9:
                        fname = "James"
                    case 10:
                        fname = "Robert"
                    case 11:
                        fname = "Ryan"
                    case 12:
                        fname = "Alex"
                    case 13:
                        fname = "Kyle"
                    case 14:
                        fname = "Ben"
                    case 15:
                        fname = "Daniel"
                    case 16:
                        fname = "Steven"
                    case 17:
                        fname = "Ronald"
                    case 18:
                        fname = "Ethan"
                    case 19:
                        fname = "Lee"
                    case 20:
                        fname = "Stan"
                    case 21:
                        fname = "Eric"
                    case 22:
                        fname = "Aubrey"
                    case 23:
                        fname = "Lewis"
                    case 24:
                        fname = "Kenny"
                    case 25:
                        fname = "Anthony"
                        
            elif gender == "F" or gender == "/":
                match firstrand: 
                    case 1:
                        fname = "Allison"
                    case 2:
                        fname = "Jane"
                    case 3:
                        fname = "Elisa"
                    case 4:
                        fname = "Jackie"
                    case 5:
                        fname = "Alexa"
                    case 6:
                        fname = "Lucille"
                    case 7:
                        fname = "Beth"
                    case 8:
                        fname = "Taylor"
                    case 9:
                        fname = "Hayley"
                    case 10:
                        fname = "Ella"
                    case 11:
                        fname = "Karen"
                    case 12:
                        fname = "Rebecca"
                    case 13:
                        fname = "Betty"
                    case 14:
                        fname = "Reena"
                    case 15:
                        fname = "Fiona"
                    case 16:
                        fname = "Sylvia"
                    case 17:
                        fname = "Jade"
                    case 18:
                        fname = "Laura"
                    case 19:
                        fname = "Julie"
                    case 20:
                        fname = "Jenny"
                    case 21:
                        fname = "Phoebe"
                    case 22:
                        fname = "Jess"
                    case 23:
                        fname = "Hollie"
                    case 24:
                        fname = "Kyra"
                    case 25:
                        fname = "Lisa"
                
            match lastrand: 
                case 1:
                    lname = "Smith"
                case 2:
                    lname = "Baldwin"
                case 3:
                    lname = "Brown"
                case 4:
                    lname = "Gosling"
                case 5:
                    lname = "James"
                case 6:
                    lname = "Daniels"
                case 7:
                    lname = "Lee"
                case 8:
                    lname = "Nguyen"
                case 9:
                    lname = "Mcgill"
                case 10:
                    lname = "White"
                case 11:
                    lname = "Pinkman"
                case 12:
                    lname = "Shipman"
                case 13:
                    lname = "Beneke"
                case 14:
                    lname = "Marsden"
                case 15:
                    lname = "Firth"
                case 16:
                    lname = "Walker"
                case 17:
                    lname = "Hinch"
                case 18:
                    lname = "Anderson"
                case 19:
                    lname = "Stallard"
                case 20:
                    lname = "Arnold"
                case 21:
                    lname = "Caine"
                case 22:
                    lname = "Hufford"
                case 23:
                    lname = "Clark"
                case 24:
                    lname = "Jacques"
                case 25:
                    lname = "Potts"
            
            name = fname + " " + lname
                
                
            writer.writerow([name, age, gender , relationship, region, hair, eye, height])
            y = y + 1

x = 0
while x == 0:
    print("Option 1: Youtube video analysis")
    print("Option 2: Salaries dataset classification")
    print("Option 3: Evolutionary programming")
    print("Option 4: CSV file generator")
    choice = input("Please choose option 1, 2, 3 or 4: ")
    if choice == "1":
        URLCHECK()

        choicecontinue = input("Would you like to choose another option (Y/N): ")
        if choicecontinue == "n":
            x = 1
        elif choicecontinue =="N":
            x = 1
        elif choicecontinue == ("y", "Y"):
            x = 0
    
    elif choice == "2":
        Pandas()
        
        
        choicecontinue = input("Would you like to choose another option (Y/N): ")
        if choicecontinue == "n":
            x = 1
        elif choicecontinue =="N":
            x = 1
        elif choicecontinue == ("y", "Y"):
            x = 0
        
    elif choice == "3":
        print ("You have chosen option 3")

        function_inputs = [4,-2,3.5,5,-11,-4.7]  
        desired_output = 100  

        def fitness_func(solution, solution_idx):
            output = numpy.sum(solution * function_inputs)
            fitness = 1.0 / numpy.abs(output - desired_output)
            return fitness

        num_generations = 50
        num_parents_mating = 10
        sol_per_pop = 50
        num_genes = len(function_inputs)
        init_range_low = -2
        init_range_high = 5
        mutation_percent_genes = 1

        ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating, 
                            fitness_func=fitness_func,
                            sol_per_pop=sol_per_pop, 
                            num_genes=num_genes,
                            init_range_low=init_range_low,
                            init_range_high=init_range_high,
                            mutation_percent_genes=mutation_percent_genes)
        ga_instance.run()
        ga_instance.plot_result()
        
        
        
        choicecontinue = input("Would you like to choose another option (Y/N): ")
        if choicecontinue == "n":
            x = 1
        elif choicecontinue =="N":
            x = 1
        elif choicecontinue == ("y", "Y"):
            x = 0

    elif choice == "4":
        print ("You have chosen option 4")
        csv_writer()
        
        choicecontinue = input("Would you like to choose another option (Y/N): ")
        if choicecontinue == "n":
            x = 1
        elif choicecontinue =="N":
            x = 1
        elif choicecontinue == ("y", "Y"):
            x = 0

    else:
        print ("Please choose a valid option")
        
        
