# machine-learning-sentiment-analysis
Simple algorithm to classify English sentences as positive or negative

## What is it
Simple model that gives weight to words of a corpus based on IMDB reviews to predict sentiment (Positive vs Negative) on sentences in English

## usage
* Run `sentiment-analysis-create-model.py` to train and save your model on the data
* Run `sentiment-analysis-run-model.py` to make predictions on the samples provided in it

```python
loaded_model = pickle.load(open(classifier_filename, 'rb'))
loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

testList = list()
testList.append("this is great")
testList.append("this is awful, the worst product I bought")

testJulien = loaded_vectorizer.transform(testList)

print(loaded_model.predict(testJulien))
```

## Samples of results

```bash
python3 sentiment-analysis-run-model.py
[1 0 0 1 0 0 0 0 0 0 1]
```
