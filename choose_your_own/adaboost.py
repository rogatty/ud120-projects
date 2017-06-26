from prep_terrain_data import makeTerrainData
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = AdaBoostClassifier(n_estimators=1000)
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print "Accuracy: {}".format(accuracy_score(labels_test, prediction))
