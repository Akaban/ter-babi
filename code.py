
# coding: utf-8

# 
# # Projet: Facebook BaBi tasks
# Notebook de la semaine du 08/03
# Par Thierry Loesch et Bryce Tichit
# 
# Dans ce notebook nous allons commencer par tester le modèle sur différentes tâches et ajuste les paramètres (batch_size,epochs,...) afin de voir comment les résultats varient et ce que nous pouvons faire pour améliorer notre modèle.
# 
# Modifier la taille de l'embedding de 50 à 100 -> aucun effet on observe même une regression
# 
# # Rappel dernier notebook
# 
# Dans le dernier notebook nous avions fait de plus amples tests avec différents modèles, un avec deux réseaux LSTM en symétrique et un autre avec deux réseaux Bi-LSTM en symétrique ainsi qu'un autre modèle plus spécifique. Après vérification il s'avère que les modèls LSTM et Bi-LSTM n'arrivaient pas à généraliser et nous avions des courbes d'apprentissage montrant un sévère sur-apprentissage.
# 

# In[ ]:

from sys import argv
import os

challenge_id=argv[0]
directory="./computed_results/"
directory=os.path.abspath(directory)


# In[65]:

import re

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parserBabi(data,withoutNoise):
    ret=list()
    story=list() 
    
    for phrase in data:
        phrase = phrase.decode('utf-8').strip()
        id_phrase,phrase = phrase.split(' ',1)
        id_phrase= int(id_phrase)
        
        if id_phrase == 1: #Nouvelle story
            story=list()
            
        if '\t' in phrase: #Si tabulation alors il s'agit de la question ainsi que de la réponse
            q, a, justif = phrase.split('\t')
            q = tokenize(q)
            
            if withoutNoise:
                inds = map(int,justif.split())
                data_story = [story[x-1] for x in inds]
            else:
                data_story = [x for x in story if x] 
                                             
            ret.append((data_story,q,a)) #Nos données d'apprentissages
            story.append('')
        
        else: 
            #Alors la phrase est tout simplement un des élements de raisonnement et non une question
            story.append(tokenize(phrase))
            
    return ret

def readAndParse(f,withoutNoise=False):
    data = parserBabi(f.readlines(),withoutNoise)
    return [([substory for substories in story for substory in substories], q , a) for story,q,a in data]


# In[66]:

# # Récupération des données
# 
# On récupère les fichiers sur internet directement avec la fonction get_file, cette fonction a l'avantage de ne pas tout retelecharger si les données sont déjà sur la machines (dans ~/.keras/datasets). Puis on applique les fonctions du parser afin de récupérer les données dans la structure que l'on veut.

# In[16]:

from keras.utils.data_utils import get_file
import tarfile
from random import shuffle
import fnmatch

try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
except:
   #print("erreur pendant le telechargement")

def trunc(f):
    s = '{}'.format(f)
    i, p, d = s.partition('.')
    return int('.'.join([i, d][:1]))

def splitData(data,split_take):
    
    size=len(data)
    
    train_prop,test_prop,val_prop = trunc(split_take[0] * size),trunc(split_take[1] * size), trunc(split_take[2] * size)
    
    return data[0:train_prop],data[train_prop:train_prop+test_prop],data[train_prop+test_prop:train_prop+test_prop+val_prop]
   
equalProp = lambda size_data,num : tuple([num/float(size_data)] * 3)

def getChallenge(tar,num,_10k=False):
    
    if(_10k):
        srcdir = 'tasks_1-20_v1-2/en-10k/'
    else:
        srcdir = 'tasks_1-20_v1-2/en/'
        
    for member in tar.getnames():
        
        if fnmatch.fnmatch(member, srcdir+'*') and fnmatch.fnmatch(member,'*qa'+str(num)+'_*'):
            
            #return member
            return '_'.join(member.split('_')[0:-1]) + '_{}.txt'
    

tar = tarfile.open(path)

#challenge = 'tasks_1-20_v1-2/en-10k/qa14_time-reasoning_{}.txt'
#challenge = 'tasks_1-20_v1-2/en/qa19_path-finding_{}.txt'
challenge = getChallenge(tar,challenge_id,_10k=True)
withoutNoise=False

#print "doing train"
train = readAndParse(tar.extractfile(challenge.format('train')),withoutNoise=withoutNoise)
#print "doing test"
test = readAndParse(tar.extractfile(challenge.format('test')),withoutNoise=withoutNoise)
data = train+test

#print "Nombre d'exemples total: {}".format(len(data))

#proportions=(0.1,0.1,0.1) 
proportions=equalProp(len(data),1000) #on en veut 1000 pour chaque ensemble
train,test,validation=splitData(data,proportions)

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test + validation)))
vocab_size = len(vocab) + 1
word_idx = dict(((w,i+1) for i,w in enumerate(vocab)))

story_max = max((len(x) for x,_,_ in train+test+validation))
question_max = max((len(x) for _,x,_ in train+test+validation))

#print "Training: {} samples , Test: {} samples , Validation: {} samples".format(len(train),len(test),len(validation))


# # Vectorisation
# 
# Comme vu en cours on doit vectoriser nos données (les assimiler a des nombres et les mettre dans des matrices) afin de pouvoir entrainer notre réseau neuronal.
# 
# La fonction pad_sequences plus bas permet de transformer notre liste de liste en matrice numpy en ajoutant des 0. pour complèter quand il n'y a pas de données.

# In[17]:

#Mettons tout ça dans des matrices
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def vectorize(data,wordidx,storymaxlen,questionmaxlen):
    X,Xq,Y = list(),list(),list()
    
    lookup = lambda m : wordidx[m]
    
    for story, question, reponse in data:
        X.append(map(lookup,story))
        Xq.append(map(lookup,question))
        Yline = np.zeros(vocab_size)
        Yline[wordidx[reponse]] = 1
        Y.append(Yline)
        
    return pad_sequences(X, maxlen=story_max), pad_sequences(Xq, maxlen=question_max), np.array(Y)
        
    
X,Xq,Y = vectorize(train,word_idx,story_max,question_max)
X_test,Xq_test,Y_test = vectorize(test,word_idx,story_max,question_max)
X_val,Xq_val,Y_val = vectorize(validation,word_idx,story_max,question_max)


# In[18]:

#print X[0:10]
#print Xq[0:10]
#print Y[0:10]


# # Modèle
# 
# Ensuite nous crééons notre modèle keras, nous avions choisi donc de faire un modèle pour les story et pour les questions afin de pouvoir raisonner sur chacun d'eux différement puis de les combiner en un modèle, ainsi nous avons un seul modèle mais qui traite différement les story des questions. On a un LSTM dans le modèle des questions car il faut raisonner d'abord sur les questions seules puis les histoires avec les questions (raisonner sur les histoires seules ne veut pas dire grand chose du point de vue d'un raisonnement)
# 
# Schema modèle:
# 
#                     question_model                              story_model
#                          |                                           |
#                       Embedding                                  Embedding
#                           |                                          |
#                                                                     /
#                         LSTM                                       /
#                           |                                       /
#                         RepeatVector                            /
#                                     \                          /
#                                       \_______________________/
#                                                    |
#                                                    |
#                                               Merge(mode=concat)
#                                                    |
#                                                  LSTM
#                                                    |
#                                                  SoftMax
# 
# 

# In[19]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def plotLearningCurves_acc(history_model,save=''):
    plt.plot(history_model.history['acc'])
    plt.plot(history_model.history['val_acc'])
    plt.title('Precision du modele')
    plt.ylabel('Precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig(save,bbox_inches='tight')
    #plt.show()
    
def plotLearningCurves_loss(history_model,save=''):
    plt.plot(history_model.history['loss'])
    plt.plot(history_model.history['val_loss'])
    plt.title('Perte sur le modele')
    plt.ylabel('Perte')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig(save,bbox_inches='tight')
    #plt.show()


# In[20]:

from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge,RepeatVector,Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam


embed_size = 50
batch_size=64
epochs=25

story_model = Sequential()
story_model.add(Embedding(vocab_size,embed_size,input_length=story_max))

question_model = Sequential()
question_model.add(Embedding(vocab_size,embed_size,input_length=question_max))
question_model.add(LSTM(embed_size))
question_model.add(RepeatVector(story_max)) #permet d'ajuster la taille du modèle afin de préparer un merge


model = Sequential()
model.add(Merge([story_model, question_model], mode='concat'))
model.add(LSTM(embed_size))
model.add(Dense(vocab_size))
model.add(Activation("softmax"))

model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit([X, Xq], Y, batch_size=batch_size, nb_epoch=epochs,validation_data=([X_val,Xq_val],Y_val))

# In[21]:

#Calcul de précision sur l'ensemble de test

challenge_dir=os.path.join(directory,"challenge/{}".format(challenge_id))

loss,acc = model.evaluate([X_test,Xq_test],Y_test, batch_size=batch_size)
ps="\nPerte = {:.3f}".format(loss)
ass= "Précision = {:.0f}%".format(acc*100)

if not os.path.exists(challenge_dir):
        os.makedirs(challenge_dir)

plotLearningCurves_acc(history,save=os.path.join(challenge_dir,"accuracy.png"))
plotLearningCurves_loss(history,save=os.path.join(challenge_dir,"loss.png"))
model.save(os.path.join(challenge_dir,"model.h5"))

with open(os.path.join(challenge_dir,"info"),'w+') as file:
    file.write(ps)
    file.write(ass)



