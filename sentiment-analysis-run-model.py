from sklearn.feature_extraction.text import CountVectorizer
import pickle

classifier_filename = 'final_classifier.pkl'
vectorizer_filename = 'final_vectorizer.pkl'

loaded_model = pickle.load(open(classifier_filename, 'rb'))
loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

testList = list()
testList.append("this is bullshit")
testList.append("this is boring")
testList.append("this is stupid")
testList.append("this is great")
testList.append("this is awful, the worst product I bought")
testList.append("this is not that good")
testList.append("this is not that good, I am disappointed")
testList.append("this is not good")

testJulien = loaded_vectorizer.transform(testList)

print(loaded_model.predict(testJulien))