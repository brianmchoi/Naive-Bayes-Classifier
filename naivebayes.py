import sys
import io
import math

class NaiveBayes(object):

    def __init__(self, trainingData):
        """
        Initializes a NaiveBayes object and the naive bayes classifier.
        :param trainingData: full text from training file as a single large string
        """
        self.red_dict = {}
        self.blue_dict = {}
        self.red_total_word_count = 0
        self.blue_total_word_count = 0
        self.red_probs = {}
        self.blue_probs = {}
        self.trainingData = trainingData
        #create word dictionaries for each type of speech
        self.create_dicts()
        #dictionary of probs for each word in red or blue speeches    
        self.create_prob_dicts()

    def create_dicts(self):
        speech_list = self.trainingData.split("\n")
        for text in speech_list:
            text_split = text.split("\t")
            label = text_split[0]
            speech = text_split[1]
            word_list = speech.split(" ")
            speech_length = len(word_list)
            #make counter dictionaries for both red and blue speeches
            if label == "RED":
                self.red_total_word_count += speech_length
                #calculate total number of words in red speeches
                for word in word_list:
                    if word not in self.red_dict:
                        self.red_dict[word] = 1
                    else:
                        self.red_dict[word] += 1
            else:
                self.blue_total_word_count += speech_length
                #calculate total number of words in blue speeches
                for word in word_list:
                    if word not in self.blue_dict:
                        self.blue_dict[word] = 1
                    else:
                        self.blue_dict[word] += 1

    def create_prob_dicts(self):
        for key in self.red_dict:
            self.red_probs[key] = (self.red_dict[key] + 1) / (self.red_total_word_count + len(self.red_dict))
        for key in self.blue_dict:
            self.blue_probs[key] = (self.blue_dict[key] + 1) / (self.blue_total_word_count + len(self.blue_dict))

    def estimateLogProbability(self, sentence):
        """
        Using the naive bayes model generated in __init__, calculate the probabilities that this sentence is in each category. Sentence is a single sentence from the test set. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary. 
        :param sentence: the test sentence, as a single string without label
        :return: a dictionary containing log probability for each category
        """
        red_prob = 0
        blue_prob = 0
        speech_words = sentence.split(" ")
        for word in speech_words:
            if word in self.red_dict:
                red_prob += math.log(self.red_probs[word])
            else: 
                red_prob += math.log(1 / (self.red_total_word_count + len(self.red_dict)))
            if word in self.blue_dict:
                blue_prob += math.log(self.blue_probs[word])
            else: 
                blue_prob += math.log(1 / (self.blue_total_word_count + len(self.blue_dict)))
        red_prob += math.log(self.red_total_word_count / (self.red_total_word_count + self.blue_total_word_count))
        blue_prob += math.log(self.blue_total_word_count / (self.red_total_word_count + self.blue_total_word_count))

        return {"red": red_prob, "blue": blue_prob}

    def testModel(self, testData):
        """
        Using the naive bayes model generated in __init__, test the model using the test data. You should calculate accuracy, precision for each category, and recall for each category. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary.
        :param testData: the test file as a single string
        :return: a dictionary containing each item as identified by the key
        """
        test_list = testData.split("\n")
        tr, fr, red = 0, 0, 0
        tb, fb, blue = 0, 0, 0

        line_counter = 1

        for speech_text in test_list:
            speech_split = speech_text.split("\t")
            true_label = speech_split[0]
            speech = speech_split[1]
            probs = self.estimateLogProbability(speech)
            print("Line " + str(line_counter) + " for red: " + str(probs["red"]))
            print("Line " + str(line_counter) + " for blue: " + str(probs["blue"]) + "\n")
            if probs["red"] > probs["blue"]:
                pred_label = "RED"
            else:
                pred_label = "BLUE"

            if true_label == pred_label:
                if true_label == "RED":
                    red += 1
                    tr += 1
                else:
                    blue += 1
                    tb += 1
            else:
                if true_label == "RED":
                    red += 1
                    fr += 1
                else:
                    blue += 1
                    fb += 1
            
            line_counter += 1
        
        red_precision = (tr) / (tr + fb)
        blue_precision = (tb) / (tb + fr)
        red_recall = (tr) / (tr+fr)
        blue_recall = (tb) / (tb+fb)
        accuracy = (tr+tb) / (red+blue)

        return {'overall accuracy': accuracy,
                'precision for red': red_precision,
                'precision for blue': blue_precision,
                'recall for red': red_recall,
                'recall for blue': blue_recall}

"""
The following code is used only on your local machine. The autograder will only use the functions in the NaiveBayes class.            

You are allowed to modify the code below. But your modifications will not be used on the autograder.
"""
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 naivebayes.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    with io.open(train_txt, 'r', encoding='utf8') as f:
        train_data = f.read()

    with io.open(test_txt, 'r', encoding='utf8') as f:
        test_data = f.read()

    model = NaiveBayes(train_data)
    evaluation = model.testModel(test_data)
    print("overall accuracy: " + str(evaluation['overall accuracy'])
        + "\nprecision for red: " + str(evaluation['precision for red'])
        + "\nprecision for blue: " + str(evaluation['precision for blue'])
        + "\nrecall for red: " + str(evaluation['recall for red'])
        + "\nrecall for blue: " + str(evaluation['recall for blue']))



