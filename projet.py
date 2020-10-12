# -*-coding:Latin-1 -*
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

import pandas as pd


#Base de données

train = pd.read_csv("gisementLearn.txt",skiprows=1,header=None,names=["X_COORD","Y_COORD","BOUGUER","GRTOPISO","PROF_MOHO","S_DEPTH","TOPO_ISO","BENPENTE","DIST_SEIS","DIST_SUBS","DIST_VOL","NBR_SEISM","DIST_112","DIST_135","DIST_157","DIST_180","DIST_22","DIST_45","DIST_67","DIST_90","AGE","ROCK","OR"])

trainAge = []
for i in train["AGE"]:
    if i=="PALEOZOIC":
        trainAge.append(1)
    elif i=="PROTEROZOIC":
        trainAge.append(2)
    elif i=="MESOZOIC":
        trainAge.append(3)
train["AGE"] = trainAge

trainRock = []
for i in train["ROCK"]:
    if i=="VOLCANIC" :
        trainRock.append(1)
    elif i=="VOL_SEDIMENTARY":
        trainRock.append(2)
    elif i=="METAMORPHIC":
        trainRock.append(3)
    elif i=="PLUTONIC":
        trainRock.append(4)
    elif i=="SEDIMENTARY":
        trainRock.append(5)
    elif i=="UNDIFFERENT":
        trainRock.append(6)
train["ROCK"]=trainRock

trainOr = []
for i in train["OR"]:
    if i=="GISEMENT":
        trainOr.append(1)
    else :
        trainOr.append(0)
train["OR"] = trainOr


#Données �  prédire

toPredict = pd.read_csv("gisementTestNoLabel.txt",skiprows=1,header=None,names=["X_COORD","Y_COORD","BOUGUER","GRTOPISO","PROF_MOHO","S_DEPTH","TOPO_ISO","BENPENTE","DIST_SEIS","DIST_SUBS","DIST_VOL","NBR_SEISM","DIST_112","DIST_135","DIST_157","DIST_180","DIST_22","DIST_45","DIST_67","DIST_90","AGE","ROCK"])

predictAge = []
for i in toPredict["AGE"]:
    if i=="PALEOZOIC":
        predictAge.append(1)
    elif i=="PROTEROZOIC":
        predictAge.append(2)
    elif i=="MESOZOIC":
        predictAge.append(3)
toPredict["AGE"]=predictAge

predictRock = []
for i in toPredict["ROCK"]:
    if i=="VOLCANIC" :
        predictRock.append(1)
    elif i=="VOL_SEDIMENTARY":
        predictRock.append(2)
    elif i=="METAMORPHIC":
        predictRock.append(3)
    elif i=="PLUTONIC":
        predictRock.append(4)
    elif i=="SEDIMENTARY":
        predictRock.append(5)
    elif i=="UNDIFFERENT":
        predictRock.append(6)
toPredict["ROCK"]=predictRock


columns = list(train.columns[:22])

train_set, test_set = train_test_split(train,test_size=0.2)
gisement = train_set["OR"]

print("Recherche du meilleur résultat...")

optimum_samples_split_tree = 2
optimum_samples_split_forest = 2
final_result_tree = 0
final_result_forest = 0

for k in range (2,50):
    #Arbre de décision
    tree=DecisionTreeClassifier(min_samples_split=k)
    digit_tree=tree.fit(train_set[columns],gisement)
    pred = digit_tree.predict(test_set[columns])
    result = digit_tree.score(test_set[columns], test_set["OR"])
    if result>final_result_tree:
        final_result_tree=result
        optimum_samples_split_tree=k
    #Forêt aléatoire
    forest = RandomForestClassifier(n_estimators=100,min_samples_split=k)
    forest = forest.fit(train_set[columns],gisement)
    pred = forest.predict(test_set[columns])
    result = forest.score(test_set[columns],test_set["OR"])
    if result>final_result_forest:
        final_result_forest=result
        optimum_samples_split_forest=k

print("\n")

print("Recherche terminée!")

y = train["OR"]
X = train[columns]

print("\n")
if final_result_tree > final_result_forest :
    print("Méthode retenue : Arbre de décision pour min_samples_split =")
    print(optimum_samples_split_tree)
    print("\n")
    print("Accuracy :")
    print(final_result_tree)
    digit_tree = DecisionTreeClassifier(min_samples_split=optimum_samples_split_tree)
    digit_tree.fit(X, y)
    pred = digit_tree.predict(toPredict)
else:
    print("Méthode retenue : Forêt aléatoire pour min_samples_split =")
    print(optimum_samples_split_forest)
    print("\n")
    print("Accuracy : ")
    print(final_result_forest)
    forest = RandomForestClassifier(n_estimators=100,min_samples_split=optimum_samples_split_forest)
    forest.fit(X,y)
    pred = forest.predict(toPredict)

toPredict["OR"]=pred

predictAge = []
for i in toPredict["AGE"]:
    if i==1:
        predictAge.append("PALEOZOIC")
    elif i==2:
        predictAge.append("PROTEROZOIC")
    elif i==3:
        predictAge.append("MESOZOIC")
toPredict["AGE"]=predictAge

predictRock = []
for i in toPredict["ROCK"]:
    if i==1 :
        predictRock.append("VOLCANIC")
    elif i==2:
        predictRock.append("VOL_SEDIMENTARY")
    elif i==3:
        predictRock.append("METAMORPHIC")
    elif i==4:
        predictRock.append("PLUTONIC")
    elif i==5:
        predictRock.append("SEDIMENTARY")
    elif i==6:
        predictRock.append("UNDIFFERENT")
toPredict["ROCK"]=predictRock

predictOr = []
for i in toPredict["OR"]:
    if i==1:
        predictOr.append("GISEMENT")
    else :
        predictOr.append("STERILE")
toPredict["OR"] = predictOr


toPredict.to_csv("gisementTest.txt",index=False, header=True)
print("\n")
print("Résultat dans gisementTest.txt")



