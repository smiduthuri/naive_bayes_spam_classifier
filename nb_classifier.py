import csv
import collections
import numpy as np
import sys

class MultinomialNB():

    def __init__(self):
        self.y_log_apriori = None
        self.x_given_y_log_prob = None

    def fit(self, X, y):
        # Pr[Y|X] = Pr[X|Y]*Pr[Y]/Pr[X]
        # Pr[Y] = (number of spam or ham)/number of data points
        # Pr[X] = (number of occurrences of a word)/(total number of occurrences of all words)
        # Pr[X|Y] = (number of occurrences of a word)/(number of occurrences of all words in Y=spam/ham)
        row = X.shape[0]
        ham_0_spam_1 = list()
        for class_type in np.unique(y):
            ham_0_spam_1.append([x_list for x_list,y_val in zip(X,y) if y_val == class_type])
        # log(Pr[Y]) = [log(Prob(spam)), log(Prob(ham))]
        self.y_log_apriori = [np.log(float(len(i)) / float(row)) for i in ham_0_spam_1]
        # Number of times each feature occurs in each class
        freq = np.array([np.array(i).sum(axis=0) for i in ham_0_spam_1])
        # self.feature_log_prob = log(Pr[X|Y])
        self.x_given_y_log_prob = list()
        for i in range(freq.shape[0]):
            self.x_given_y_log_prob.append(freq[i].astype(float)/freq[i].sum().astype(float))
        self.x_given_y_log_prob = np.array(self.x_given_y_log_prob)

        return self

    def predict(self, X):
        # [Pr[Y=spam|X=input], Pr[Y=ham|X=input]] = Pr[X=input|Y]*Pr[Y]/Pr[X=input] But we don't divide by P[X]
        return np.argmax([(self.x_given_y_log_prob * x).sum(axis=1) for x in X], axis=1)
        # return np.argmax([(self.x_given_y_log_prob * x).sum(axis=1) + self.y_log_apriori for x in X], axis=1)

def process_dataset(filename):
    print 'Opening file:', filename
    with open(filename) as csvfile:
        emailID_list = list()
        spamStatus_list = list()
        wordCntPerDataPoint = list()
        words_list = collections.Counter()
        document_indices_per_word = dict()
        readCSV = csv.reader(csvfile)
        row_number = 0
        for row in readCSV:
            localWordCntr = collections.Counter()
            row = row[0].split(' ')
            emailID_list.append(row[0])
            spamStatus_list.append(int(row[1] == 'spam'))
            for word, count in zip(*[iter(row[2:])]*2):
                if not word.isdigit():
                    words_list[word] += int(count)
                    localWordCntr[word] += int(count)
                    if not document_indices_per_word.has_key(word):
                        document_indices_per_word[word] = [row_number]
                    else:
                        document_indices_per_word[word].append(row_number)
            wordCntPerDataPoint.append(dict(localWordCntr))
            row_number += 1
    processedData = dict()
    processedData['emailID'] = emailID_list                             # Size=numDataPoints
    processedData['spamStatus'] = spamStatus_list                       # Size=numDataPoints
    processedData['wordFrequencyPerDataPoint'] = wordCntPerDataPoint    # Size=numDataPoints
    processedData['vocabularyAndGlobalFrequency'] = dict(words_list)    # Size=vocabularySize
    processedData['dictOfDocumentsPerWord'] = document_indices_per_word # Size=vocabularySize
    return processedData

def generate_clean_data(processedData, trainingData, genTFIDF=False, tfidf_threshold=0.0001, listOfWordsToRemove=list(), test=False):
    docsPerWord = dict(processedData['dictOfDocumentsPerWord'])
    wordFrequencyPerDocument = list(processedData['wordFrequencyPerDataPoint'])
    globalWordVocabulary = dict(processedData['vocabularyAndGlobalFrequency'])
    ndocs = len(wordFrequencyPerDocument)
    if genTFIDF is True and test is False:
        # Do not perform TFIDF if running on Test Dataset
        # TFIDF for a term = tf(term,document) x idf(term) / l2_norm(document)
        # tf(term,document) = no. of times term occurs in a document (data-point) may be 0
        # df(document,term) = no. of documents where the term occurs != 0
        # ndocs = number of data-points                              != 0
        # idf(term) = log(ndocs/df(document,term)) + 1
        tfidfPerWord = dict()
        for i in range(ndocs):
            data_point = dict(wordFrequencyPerDocument[i])
            temp_data_point = dict()
            tfidf_square_sum = 0
            for word,tf in data_point.items():
                df = len(docsPerWord[word])
                idf = float(np.log(ndocs)) - (1+float(np.log(df))) + 1
                tfidf = tf*idf
                tfidf_square_sum += pow(tfidf,2)
                temp_data_point[word] = tfidf
            tfidf_l2_norm = pow(tfidf_square_sum,0.5)
            for word in temp_data_point.keys():
                temp_data_point[word] /= tfidf_l2_norm
            wordFrequencyPerDocument[i] = temp_data_point

            for word,tfidf in temp_data_point.items():
                if not tfidfPerWord.has_key(word):
                    tfidfPerWord[word] = [tfidf]
                else:
                    tfidfPerWord[word].append(tfidf)
        for word in tfidfPerWord.keys():
            tfidfPerWord[word] = float(np.mean(tfidfPerWord[word]))
        for word in globalWordVocabulary.keys():
            if tfidfPerWord[word] > tfidf_threshold:
                tfidfPerWord.pop(word)
        listOfWordsToRemove = list(set(listOfWordsToRemove).union(set(tfidfPerWord.keys())))

    if len(listOfWordsToRemove) == 0 and test is False:
        print 'Not removing any words'
    else:
        if test is True and trainingData is not None:
            print 'Need to consider words only from Training Dataset in the order which they exist in that dataset'
            newRemoveWords = set(processedData['vocabularyAndGlobalFrequency'].keys()).difference(set(trainingData['vocabularyAndGlobalFrequency'].keys()))
            listOfWordsToRemove += list(newRemoveWords)
        for i in range(ndocs):
            data_point = dict(wordFrequencyPerDocument[i])
            [data_point.pop(word) for word in listOfWordsToRemove if word in data_point.keys()]
        setOfWordsToRemove = set(listOfWordsToRemove).intersection(set(globalWordVocabulary.keys()))
        print 'Number of words to remove:', len(setOfWordsToRemove)
        print 'Removing words:', listOfWordsToRemove
        [globalWordVocabulary.pop(word) for word in setOfWordsToRemove]
        if test is False:
            processedData['vocabularyAndGlobalFrequency'] = globalWordVocabulary

    if test is True:
        globalWordVocabulary = dict(trainingData['vocabularyAndGlobalFrequency'])

    X_Matrix = list()
    for i in range(ndocs):
        temp_list = list()
        curr_doc = dict(wordFrequencyPerDocument[i])
        for word in globalWordVocabulary:
            if curr_doc.has_key(word):
                temp_list.append(curr_doc[word])
            else:
                temp_list.append(0)
        X_Matrix.append(np.array(temp_list))

    Y_Vector = processedData['spamStatus']
    X_Matrix = np.array(X_Matrix)
    return X_Matrix, Y_Vector

if __name__ == '__main__':

    if len(sys.argv) != 7:
        print 'Use as python', sys.argv[0], '-f1 <train_dataset> -f2 <test_dataset> -o <output_file>'
        exit()
    train_file = 'train'
    test_file = 'test'
    output_file = 'output'
    if sys.argv[1] == '-f1':
        train_file = sys.argv[2]
    elif sys.argv[1] == '-f2':
        test_file = sys.argv[2]
    elif sys.argv[1] == '-o':
        output_file = sys.argv[2]

    if sys.argv[3] == '-f1':
        train_file = sys.argv[4]
    elif sys.argv[3] == '-f2':
        test_file = sys.argv[4]
    elif sys.argv[3] == '-o':
        output_file = sys.argv[4]

    if sys.argv[5] == '-f1':
        train_file = sys.argv[6]
    elif sys.argv[5] == '-f2':
        test_file = sys.argv[6]
    elif sys.argv[5] == '-o':
        output_file = sys.argv[6]

    print 'First we will process the data'
    processedData = process_dataset(train_file)
    print 'Data is processed'
    print 'Now cleaning data'
    X_Train, Y_Train = generate_clean_data(processedData, test=False, trainingData=None,
                                           tfidf_threshold=0.0002, genTFIDF=True,
                                           listOfWordsToRemove=['the','and','a','an','is'])

    classifier_NB = MultinomialNB()
    classifier_NB.fit(X_Train, Y_Train)

    test_processedData = process_dataset(test_file)
    print 'Test Data processed'
    X_test, Y_test = generate_clean_data(test_processedData, test=True, trainingData=processedData)
    prediction = classifier_NB.predict(X_test)

    from sklearn.metrics import accuracy_score,precision_score
    accuracy = accuracy_score(Y_test, list(prediction))
    precision = precision_score(Y_test, list(prediction))
    print 'accuracy:', accuracy
    print 'precision:', precision

    f = open(output_file,'w')
    for email_id, prediction_val in zip(test_processedData['emailID'],prediction):
        f.write(str(email_id)+' '+str(prediction_val)+'\n')
    f.close()