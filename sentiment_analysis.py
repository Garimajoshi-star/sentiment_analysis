# Import libraries
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

# Initialise different vocabulary classes
positive_vocab = [ 'amazing', 'adventure', 'beautiful', 'bravo', 'classical', 'delightful', 'excellent', 'exciting', 'fabulous', 'funny', 'fun', 'glamorous', 'knowledgeable', 'lovely', 'marvellous', 'motivating', 'positive', 'refreshing', 'rewarding', 'stunning', 'superb', 'terrific', 'thrilling', 'wonderful', 'wow', 'worthy', ':)' ]

negative_vocab = [   'annoying', 'bad', 'boring', 'dirty', 'fail', 'disgusting', 'unpredictable', 'uneven', 'dreadful', 'violent', 'senseless', 'confused', 'stupid', 'bland', 'tiresome', 'disappointing', 'uninteresting', 'outdated', 'terrible', 'useless', 'hate', ':(']

neutral_vocab = [   'ordinary', 'fast-moving', 'high-budget', 'dramatic','movie','the','sound','satisfactory','simple','was','is','actors','did','know','words','not' ]

# Convert words into features
def word_feats(words):
    return dict([(word, True) for word in words])

positive_features = [(word_feats(positive_pred), 'positive_pred') for positive_pred in positive_vocab]
negative_features = [(word_feats(negative_pred), 'negative_pred') for negative_pred in negative_vocab]
neutral_features = [(word_feats(neutral_pred), 'neutral_pred') for neutral_pred in neutral_vocab] 


# Generate training dataset 
train_set = positive_features + negative_features + neutral_features


# Classify training data features
classifier = NaiveBayesClassifier.train(train_set) 

# Split input text 
positive_pred = 0
negative_pred = 0
neutral_pred = 0 
sentence = "It was a great movie with an equal balance of humour and mystery. The story line was awesome with an unpredictable climax." 
words = sentence.split(' ')

# Generate output

for word in words:
    output = classifier.classify( word_feats(word))
    
    if output == 'positive_pred':
       positive_pred = positive_pred + 1

    if output == 'negative_pred':
        negative_pred = negative_pred + 1
  
    if output == 'neutral_pred':
        neutral_pred = neutral_pred + 1

print('Positive_review: ' + str(float(positive_pred)/len(words)))
print('Negative_review: ' + str(float(negative_pred)/len(words)))
print('Neutral_review: ' + str(float(neutral_pred)/len(words)))



 
