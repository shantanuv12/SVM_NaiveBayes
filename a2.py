from utils import *
import math
import sys
import string
import re
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
import nltk

wordnet_lemmatizer = WordNetLemmatizer()

allowed_tokens = string.ascii_letters

def bgrams(doc):
    bg = nltk.bigrams(doc)
    l = []
    for i in bg:
        str = ''
        for j in range(len(i)):
            if(j == 0):
                str += i[0]
            else:
                str += ' ' + i[j]
        l.append(str)
    return l

def lem(doc):
    return [wordnet_lemmatizer.lemmatize(w) for w in doc.split()]

def randomPrediction():
    return random.randint(1,5)

def maxfreqprediction(num_reviews_in_class):
    return np.argmax(num_reviews_in_class) + 1

def createDict(filename, stemming = False, addedFeatures = False):
    '''
        Function to process the reviews and create the dictionary
    '''
    dict = {}
    total_words = 0
    count_in_class = [0,0,0,0,0]
    num_reviews_in_class = [0,0,0,0,0]
    num_words_in_class = [0,0,0,0,0,0]
    reader = json_reader(filename)
    for i,line in enumerate(reader):
        if (addedFeatures):
            s1 = lem(re.sub('[^%s]' % allowed_tokens, ' ', line['text']))
            s = bgrams(' '.join(s1))
            if(stemming):
                s = getStemmedDocuments(' '.join(s))
        else:
            s = re.sub('[^%s]' % allowed_tokens, ' ', line['text']).split()
            if(stemming):
                s = getStemmedDocuments(' '.join(s))
        num_reviews_in_class[int(line['stars']) - 1] += 1
        for word in s:
            num_words_in_class[int(line['stars']) - 1] += 1
            word = word.lower()
            if word not in dict:
                dict[word] = [0,0,0,0,0]
                dict[word][int(line['stars']) - 1] = 1
            else:
                dict[word][int(line['stars']) - 1] += 1
    return dict, num_reviews_in_class, num_words_in_class

def classProbab(num_reviews_in_class):
    '''
        Calculating the class probablities
    '''
    probab = [0.0,0.0,0.0,0.0,0.0]
    size = sum(num_reviews_in_class)
    for i in range(5):
        probab[i] = num_reviews_in_class[i] / size
    return probab

def predictRating(review, dict, num_reviews_in_class, num_words_in_class, classProb, stemming = False, addedFeatures = False):
    '''
        Predicting the rating from learned paramters
    '''
    if(addedFeatures):
        s1 = lem(re.sub('[^%s]' % allowed_tokens, ' ', review))
        s = bgrams(' '.join(s1))
        if(stemming):
            s = getStemmedDocuments(' '.join(s))
    else:
        s = re.sub('[^%s]' % allowed_tokens, ' ', review).split()
        if(stemming):
            s = getStemmedDocuments(' '.join(s))
    prob = [0.0,0.0,0.0,0.0,0.0]
    mod_v = len(dict)
    max_sum = float('-inf')
    max_ind = -1
    for i in range(5):
        sum = 0.0
        for word in s:
            word = word.lower()
            n = 0
            if word in dict:
                n = dict[word][i]
            sum += math.log((n + 1) / (mod_v + num_words_in_class[i]))
        sum += math.log(classProb[i])
        prob[i] = sum
        if(sum > max_sum):
            max_sum = sum
            max_ind = i
    return max_ind + 1

def test(filename, dict, num_reviews_in_class, num_words_in_class, classProb, partnum = 0, stemming = False, addedFeatures = False):
    '''
        Testing the test data by predicting rating for every review
    '''
    predicted_rating = []
    actual_ratings = []
    correct_ratings = 0
    total_ratings = 0
    reader = json_reader(filename)
    max_freq_pred = maxfreqprediction(num_reviews_in_class)
    for line in reader:
        total_ratings += 1
        if(partnum == 0):
            if(addedFeatures):
                if(stemming):
                    r = predictRating(line['text'], dict, num_reviews_in_class, num_words_in_class, classProb, stemming = True, addedFeatures = True)
                else:
                    r = predictRating(line['text'], dict, num_reviews_in_class, num_words_in_class, classProb, addedFeatures = True)
            else:
                if(stemming):
                    r = predictRating(line['text'], dict, num_reviews_in_class, num_words_in_class, classProb, stemming = True)
                else:
                    r = predictRating(line['text'], dict, num_reviews_in_class, num_words_in_class, classProb)

        elif(partnum == 1):
            r = randomPrediction()
        else:
            r = max_freq_pred
        predicted_rating.append(r)
        actual_ratings.append(int(line['stars']))
        if(r == int(line['stars'])):
            correct_ratings += 1
    accuracy = correct_ratings / total_ratings * 100
    return accuracy, predicted_rating, actual_ratings

def draw_confusion_matrix(true_labels, predicted_labels, partnum = 0):
    c_matrix = confusion_matrix(true_labels, predicted_labels)
    print(c_matrix)
    fig = plt.figure()
    plt.imshow(c_matrix)
    plt.colorbar()
    plt.set_cmap('Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    plt.show()
    fig.savefig('ConfusionMatrixforNBforpart' + str(partnum) +'.png')

def macro_score(true_labels, predicted_labels):
    '''
        Calculating the F1 scores
    '''
    return f1_score(true_labels, predicted_labels, average = None)

if __name__ == '__main__':
    train_f = str(sys.argv[1])
    test_f = str(sys.argv[2])
    partnum = str(sys.argv[3])
    if(partnum == 'a'):
        dict, num_reviews_in_class, num_words_in_class = createDict(train_f)
        print('Training Done')
        classProb = classProbab(num_reviews_in_class)
        print('Completed Class probab')
        accuracy, pred, actu = test(train_f, dict, num_reviews_in_class, num_words_in_class, classProb)
        accuracy1, pred1, actu1 = test(test_f, dict, num_reviews_in_class, num_words_in_class, classProb)
        print('Train accuracy: ', accuracy)
        print('Test accuracy: ', accuracy1)
    elif(partnum == 'b'):
        dict, num_reviews_in_class, num_words_in_class = createDict(train_f)
        print('Training Done')
        classProb = classProbab(num_reviews_in_class)
        print('Completed Class probab')
        acc1, pred1, actu1 = test(test_f, dict, num_reviews_in_class, num_words_in_class, classProb, partnum = 1)
        acc2, pred2, actu2 = test(test_f, dict, num_reviews_in_class, num_words_in_class, classProb, partnum = 2)
        print('Accuracy for Random prediction: ', acc1)
        print('Accuracy for Max Frequency Prediction: ', acc2)
    elif(partnum == 'c'):
        dict, num_reviews_in_class, num_words_in_class = createDict(train_f)
        print('Training Done')
        classProb = classProbab(num_reviews_in_class)
        print('Completed Class probab')
        accuracy1, pred1, actu1 = test(test_f, dict, num_reviews_in_class, num_words_in_class, classProb)
        draw_confusion_matrix(actu1, pred1)
    elif(partnum == 'd'):
        dict, ratings, num_reviews_in_class, num_words_in_class = createDict(train_f, stemming = True)
        print('Training Done')
        classProb = classProbab(num_reviews_in_class)
        print('Completed Class probab')
        accuracy1, pred1, actu1 = test(test_f, dict, num_reviews_in_class, num_words_in_class, classProb, stemming = True)
        print('Accuracy : ', accuracy1)
    elif(partnum == 'e'):
        dict, num_reviews_in_class, num_words_in_class = createDict(train_f, stemming = False, addedFeatures = True)
        print('Training Done')
        classProb = classProbab(num_reviews_in_class)
        print('Completed Class probab')
        accuracy1, pred1, actu1 = test(test_f, dict, num_reviews_in_class, num_words_in_class, classProb, stemming = False, addedFeatures = True)
        print('Accuracy : ', accuracy1)
    elif(partnum == 'f'):
        dict, num_reviews_in_class, num_words_in_class = createDict(train_f, addedFeatures = True)
        print('Training Done')
        classProb = classProbab(num_reviews_in_class)
        print('Completed Class probab')
        accuracy1, pred1, actu1 = test(test_f, dict, num_reviews_in_class, num_words_in_class, classProb, addedFeatures = True)
        f1_score = macro_score(actu1, pred1)
        macro_f1 = sum(f1_score)/5.0
        print('f1_score: ', f1_score)
        print('macro_f1: ', macro_f1)
    elif(partnum == 'g'):
        dict, num_reviews_in_class, num_words_in_class = createDict(train_f, addedFeatures = True)
        print('Training Done')
        classProb = classProbab(num_reviews_in_class)
        print('Completed Class probab')
        accuracy1, pred1, actu1 = test(test_f, dict, num_reviews_in_class, num_words_in_class, classProb, addedFeatures = True)
        f1_score = macro_score(actu1, pred1)
        macro_f1 = sum(f1_score)/5.0
        print('Accuracy: ', accuracy1)
        print('f1_score: ', f1_score)
        print('macro_f1: ', macro_f1)
    print('DONE')
