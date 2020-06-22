#!/usr/bin/env python
import time
import os
from nb import NaiveBayes
from rank_classifier import RankClassifier
from knn import KNN
import random
from document import Document
from tfidf import Index
from kmeans import KMeans
from util import *
from collections import defaultdict, Counter
from flask import Flask, request,render_template,redirect
option_count=5
k_n=5

app = Flask(__name__)
app.config["DEBUG"] = True
@app.route("/recommend")
def recommend():

    # end = False
    # recommendation
    # while not end:
            
    all_docs=lst[0]
    # print(all_docs)
    test_docs=lst[1]
    classifier_list=lst[2]
    # pick random documents from test docs and provide titles to the user.
    global user_docs 
    user_docs = random.sample(test_docs, option_count)
    return redirect('http://localhost:5000/onepage')

@app.route('/onepage')
def onepage():
            user=user_docs
            return render_template('TE.html',user=user)
@app.route('/about')
def about():
            return render_template('about.html')
@app.route('/mailus')
def mailus():
            return render_template('mail.html')
@app.route('/inside',methods = ['POST', 'GET'])
def inside():
            global ls
            ls=[]
            c=request.form
            choice=c['val']
            if choice == 'r':
                return redirect("http://localhost:5000/recommend")
            else:
                user_choice = int(choice)-1
                selected_doc = user_docs[user_choice]
                all_docs=lst[0]
                classifier_list=lst[2]
                # classifiers are sorted according to their f_measure in decreasing order. It helps when all
                # three classifiers differ in their predictions.

                classifier_list = sorted(classifier_list, key=lambda cl: cl.stats['f_measure'], reverse=True)

                prediction_list = list()
                for classifier in classifier_list:
                    prediction_list.append(classifier.classify([selected_doc])[0])

                prediction_count = Counter(prediction_list)
                top_prediction = prediction_count.most_common(1)

                if top_prediction[0][1] > 1:
                    prediction = top_prediction[0][0]
                else:
                    prediction = prediction_list[0]

                # create knn instance using documents of predicted topic. and find k closest documents.
                knn = KNN(all_docs[prediction])
                k_neighbours = knn.find_k_neighbours(selected_doc, k_n)
                ls.append(selected_doc)
                ls.append(k_neighbours)
                return render_template('scnd.html',ls=ls)
'''
def recommendation(all_docs, test_docs, classifier_list):

    print("Recommendation System")
    print("---------------------")

    # ask user for the desired option count and recommendation count. set default value in case invalid inputs.
    try:
        option_count = int(input("\nEnter number of articles to choose from. [number from 5 to 10 suggested]: "))
        if option_count < 1 or option_count > 20:
            print("Invalid Choice.. By default selected 5.")
            option_count = 5
    except:
        print("Invalid Choice.. By default selected 5.")
        option_count = 5

    try:
        k_n = int(input("\nEnter number of recommendation per article. [number from 5 to 10 suggested]: "))
        if k_n < 1 or k_n > 20:
            print("Invalid Choice.. By default selected 5.")
            k_n = 5
    except:
        print("Invalid Choice.. By default selected 5.")
        k_n = 5

    end = False

    # run the loop until user quits.
    while not end:

        # pick random documents from test docs and provide titles to the user.
        user_docs = random.sample(test_docs, option_count)

        while True:
            print("\n---Available Choices For Articles(Titles)---\n")

            for i in range(len(user_docs)):
                print(str(i+1) + ": " + user_docs[i].title)

            print("r: Refresh List")
            print("q: Quit()\n")

            choice = input("Enter Choice: ")

            if choice == 'q':
                end = True
                break
            elif choice == 'r':
                break
            else:
                try:
                    user_choice = int(choice) - 1
                    if user_choice < 0 or user_choice >= len(user_docs):
                        print("Invalid Choice.. Try Again..")
                        continue
                except:
                    print("Invalid Choice.. Try Again..")
                    continue
                selected_doc = user_docs[user_choice]

                # classifiers are sorted according to their f_measure in decreasing order. It helps when all
                # three classifiers differ in their predictions.
                classifier_list = sorted(classifier_list, key=lambda cl: cl.stats['f_measure'], reverse=True)

                prediction_list = list()
                for classifier in classifier_list:
                    prediction_list.append(classifier.classify([selected_doc])[0])

                prediction_count = Counter(prediction_list)
                top_prediction = prediction_count.most_common(1)

                if top_prediction[0][1] > 1:
                    prediction = top_prediction[0][0]
                else:
                    prediction = prediction_list[0]

                # create knn instance using documents of predicted topic. and find k closest documents.
                knn = KNN(all_docs[prediction])
                k_neighbours = knn.find_k_neighbours(selected_doc, k_n)

                while True:
                    print("\nRecommended Articles for : " + selected_doc.title)
                    for i in range(len(k_neighbours)):
                        print(str(i+1) + ": " + k_neighbours[i].title)
                        print(str(i+1) + ": " + k_neighbours[i].text)
                    next_choice = input("\nEnter Next Choice: [Article num to read the article. "
                                            "'o' to read the original article. "
                                            "'b' to go back to article choice list.]  ")

                    if next_choice == 'b':
                        break
                    elif next_choice == 'o':
                        text = selected_doc.text
                        print("\nArticle Text for original title : " + selected_doc.title)
                        print(text)
                    else:
                        try:
                            n_choice = int(next_choice) - 1
                            if n_choice < 0 or n_choice >= k_n:
                                print("Invalid Choice.. Try Again..")
                                continue
                        except:
                            print("Invalid Choice.. Try Again..")
                            continue
                        text = k_neighbours[n_choice].text
                        print("\nArticle Text for recommended title : " + k_neighbours[n_choice].title)
                        print(text)
'''
@app.route("/")
def main():

    start_time = time.time()

    # Read documents, divide according to the topics and separate train and test data-set.

    t_path = "../bbc/"

    all_docs = defaultdict(lambda: list())

    topic_list = list()
    for topic in os.listdir(t_path):
        d_path = t_path + topic + '/'
        topic_list.append(topic)
        temp_docs = list()

        for f in os.listdir(d_path):
            f_path = d_path + f
            temp_docs.append(Document(f_path, topic))

        all_docs[topic] = temp_docs[:] 
    fold_count = 10

    train_docs, test_docs = list(), list()

    for key, value in all_docs.items():
        random.shuffle(value)
        test_len = int(len(value)/fold_count)
        train_docs += value[:-test_len]
        # explanation
    #   lis = [1,2,3,4,5]
    # print(lis[:-4])
    # print(lis[-4:])
        test_docs += value[-test_len:]

    # Create tfidf and tfidfie index of training docs, and store into the docs.
    index = Index(train_docs)

    test_topics = [d.topic for d in test_docs]

    for doc in train_docs:
        doc.vector = doc.tfidfie

    for doc in test_docs:
        doc.vector = doc.tf

    # create classifier instances.
    nb = NaiveBayes()
    rc = RankClassifier()
    kmeans = KMeans(topic_list)
    
    classifier_list = [nb,rc,kmeans]

    for i in range(len(classifier_list)):

        classifier = classifier_list[i]

        classifier.confusion_matrix, c_dict = init_confusion_matrix(topic_list)

        classifier.train(train_docs)
        predictions = classifier.classify(test_docs)

        # Update the confusion matrix and statistics with updated values.
        classifier.confusion_matrix = update_confusion_matrix(test_topics, predictions, classifier.confusion_matrix,
                                                              c_dict)

        classifier.stats = cal_stats(classifier.confusion_matrix)
    
    global lst
    lst=[]
    lst.append(all_docs)
    lst.append(test_docs)
    lst.append(classifier_list)
    return redirect('http://localhost:5000/recommend')
    # recommendation(all_docs, test_docs, classifier_list)

if __name__ == "__main__":
	app.run()
