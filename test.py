import wfdb
import numpy as np
import os

from scipy.io import wavfile
import scipy.signal
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

'''
pcg = []
labels = []

def getLabel(name):
    lbl = 0
    if name == 'Abnormal':
        lbl = 1
    return lbl    

dataset = 'training-a'
for root, dirs, directory in os.walk(dataset):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if '.dat' in directory[j]:
            fname = directory[j].split(".")
            signals, fields = wfdb.rdsamp(root+"/"+fname[0], sampfrom=10000, sampto=15000)
            signals = signals.ravel()
            label = getLabel(fields.get('comments')[0])
            pcg.append(signals)
            labels.append(label)
            print(directory[j]+" "+fname[0]+" "+str(signals.shape)+" "+str(label))
pcg = np.asarray(pcg)
labels = np.asarray(labels)

np.save("model/pcg",pcg)
np.save("model/pcg_label",labels)
'''
'''
tt = 0
time_steps = 450
nfft = 1203

def getLabel(name):
    lbl = 0
    if name == 'Abnormal':
        lbl = 1
    return lbl 

wavData = []
labels = []
dataset = 'training-a'
for root, dirs, directory in os.walk(dataset):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if '.wav' in directory[j]:
            fname = directory[j].split(".")
            sampling_freq, audio = wavfile.read(root+"/"+directory[j])
            lbl = 0
            st = root +"/"+fname[0]+".hea"
            with open(st,'r') as f:
                for line in f:
                    for word in line.split():
                        if(word=="Abnormal"):
                            lbl = 1
            f.close()
            labels.append(lbl)
            audio1 = audio/32768
            temp = mfcc(audio1, sampling_freq, nfft=nfft)
            temp = temp[tt:tt+time_steps,:]
            wavData.append(temp)
            print(directory[j]+" "+str(temp.shape)+" "+str(lbl))

wavData = np.asarray(wavData)
labels = np.asarray(labels)
np.save("model/wav",wavData)
np.save("model/wav_label",labels)
'''
'''
pcg_X = np.load("model/pcg.npy")
pcg_Y = np.load("model/pcg_label.npy")

print(pcg_X)
print(pcg_Y)
pcg_X = np.nan_to_num(pcg_X)
X_train, X_test, y_train, y_test = train_test_split(pcg_X, pcg_Y, test_size=0.2)

rfc = RandomForestClassifier(n_estimators=200, random_state=0)
rfc.fit(pcg_X, pcg_Y)
predict = rfc.predict(X_test)
acc = accuracy_score(y_test,predict)*100
print(acc)
'''
audio_X = np.load("model/wav.npy")
audio_Y = np.load("model/wav_label.npy")
audio_Y = to_categorical(audio_Y)
print(audio_X)
print(audio_Y)

audio_X = np.reshape(audio_X, (audio_X.shape[0], audio_X.shape[1], audio_X.shape[2], 1))
print(audio_X.shape)

if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    json_file.close()    
    classifier.load_weights("model/model_weights.h5")
    classifier._make_predict_function()       
else:
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape = (audio_X.shape[1], audio_X.shape[2], audio_X.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = audio_Y.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(audio_X, audio_Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
    classifier.save_weights('model/model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
                     









