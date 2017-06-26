from prep_terrain_data import makeTerrainData
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, bootstrap=False)
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print "Accuracy: {}".format(accuracy_score(labels_test, prediction))
