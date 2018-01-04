import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm

labeled_images = pd.read_csv('../train.csv')

#Take only 6000 of them to save time
images = labeled_images.iloc[0:6000,1:]
labels = labeled_images.iloc[0:6000,:1]

#Split the dataset into train set and test set
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, train_size=0.8, random_state=0)

#turn the value to 1 and 0.
images_test[images_test>0] = 1
images_train[images_train>0] = 1

clf = svm.SVC(C=100, kernel='rbf', gamma=0.001)
clf.fit(images_train, labels_train.values.ravel())
clf.score(images_test,labels_test)


#Use Grid search to find the best paramater combination, and put them back to clf
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf2 = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
clf2.fit(images_train, labels_train.values.ravel())
clf2.best_params_


#start predicting test_data
test_data=pd.read_csv('../test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data)
df = pd.DataFrame(results, columns=['label'])
df.to_csv('../results.csv', header=True, index=False)