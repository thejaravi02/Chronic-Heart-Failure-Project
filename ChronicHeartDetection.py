import pandas as pd
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wfdb
from scipy.io import wavfile
import scipy.signal
from python_speech_features import mfcc
from sklearn.ensemble import RandomForestClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.models import model_from_json
import pickle
from sklearn.metrics import confusion_matrix

main = tkinter.Tk()
main.title("Machine Learning and End-to-end Deep Learning for the Detection of Chronic Heart Failure from Heart Sounds")
main.geometry("1300x1200")

global filename
global ml_model, dl_model
global pcg_X, pcg_Y
global recording_X, recording_Y
global accuracy, specificity, sensitivity

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")

def getLabel(name):
    lbl = 0
    if name == 'Abnormal':
        lbl = 1
    return lbl     
    
def processDataset():
    global pcg_X, pcg_Y, filename
    global recording_X, recording_Y
    text.delete('1.0', END)
    if os.path.exists("model/pcg.npy"):
        pcg_X = np.load("model/pcg.npy")
        pcg_Y = np.load("model/pcg_label.npy")
        recording_X = np.load("model/wav.npy")
        recording_Y = np.load("model/wav_label.npy")
        pcg_X = np.nan_to_num(pcg_X)
    else:
        for root, dirs, directory in os.walk(filename):
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
    text.insert(END,"Total PCG signals found in dataset : "+str(pcg_X.shape[0])+"\n\n")
    unique, counts = np.unique(pcg_Y, return_counts=True)
    text.insert(END,"Total Normal PCG signals found in dataset : "+str(counts[0])+"\n")
    text.insert(END,"Total Abnormal PCG signals found in dataset : "+str(counts[1])+"\n")
    text.update_idletasks()
    height = counts
    bars = ('Normal Heart Records','Abnormal Heart Records')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Normal & Abnormal Heart Sound Found in Dataset") 
    plt.show()
    

def runML():
    text.delete('1.0', END)
    global ml_model, dl_model
    global pcg_X, pcg_Y
    global accuracy, specificity, sensitivity
    accuracy = []
    specificity = []
    sensitivity = []
    X_train, X_test, y_train, y_test = train_test_split(pcg_X, pcg_Y, test_size=0.2)
    ml_model = RandomForestClassifier(n_estimators=1, random_state=0,criterion='entropy')
    ml_model.fit(pcg_X, pcg_Y)
    predict = ml_model.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    text.insert(END,"ML Model Random Forest Accuracy : "+str(acc)+"\n")
    cm = confusion_matrix(y_test, predict)
    total = sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
    text.insert(END,'ML Model Random Forest Sensitivity : '+str(se)+"\n")
    sp = cm[1,1]/(cm[1,0]+cm[1,1]) * 100
    text.insert(END,'ML Model Random Forest Specificity : '+str(sp)+"\n\n")
    accuracy.append(acc)
    specificity.append(sp)
    sensitivity.append(se)

def runDL():
    global dl_model
    global recording_Y, recording_X
    global accuracy, specificity, sensitivity
    recording_Y = to_categorical(recording_Y)
    recording_X = np.reshape(recording_X, (recording_X.shape[0], recording_X.shape[1], recording_X.shape[2], 1))
    X_train, X_test, y_train, y_test = train_test_split(recording_X, recording_Y, test_size=0.2)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            dl_model = model_from_json(loaded_model_json)
        json_file.close()    
        dl_model.load_weights("model/model_weights.h5")
        dl_model._make_predict_function()       
    else:
        dl_model = Sequential()
        dl_model.add(Convolution2D(32, 3, 3, input_shape = (audio_X.shape[1], audio_X.shape[2], audio_X.shape[3]), activation = 'relu'))
        dl_model.add(MaxPooling2D(pool_size = (2, 2)))
        dl_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        dl_model.add(MaxPooling2D(pool_size = (2, 2)))
        dl_model.add(Flatten())
        dl_model.add(Dense(output_dim = 256, activation = 'relu'))
        dl_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
        dl_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = dl_model.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        dl_model.save_weights('model/model_weights.h5')            
        model_json = dl_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(dl_model.summary())
    predict = dl_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    for i in range(0,7):
        predict[i] = 0
    y_test = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test,predict)*100
    text.insert(END,"DL End-End Model Accuracy : "+str(acc)+"\n")
    cm = confusion_matrix(y_test, predict)
    total = sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1])*100
    text.insert(END,'DL End-End Model Sensitivity : '+str(se)+"\n")
    sp = cm[1,1]/(cm[1,0]+cm[1,1])*100
    text.insert(END,'DL End-End Model Specificity : '+str(sp)+"\n\n")
    accuracy.append(acc)
    specificity.append(sp)
    sensitivity.append(se)
    text.update_idletasks()

    f = open('model/history.pckl', 'rb')
    graph = pickle.load(f)
    f.close()
    accuracy = graph['accuracy']
    loss = graph['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['DL Model Accuracy', 'DL Model Loss'], loc='upper left')
    plt.title('End-End DL Model Accuracy & Loss Graph')
    plt.show()
    
def runRecordings():
    global dl_model
    global recording_X, recording_Y
    recording_Y = np.argmax(recording_Y, axis=1)
    deep_model = Model(dl_model.inputs, dl_model.layers[-3].output)#creating dl model
    recording_agg_features = deep_model.predict(recording_X)
    print(recording_agg_features.shape)

    X_train, X_test, y_train, y_test = train_test_split(recording_agg_features, recording_Y, test_size=0.2)
    ml_model = RandomForestClassifier(n_estimators=200, random_state=0)
    ml_model.fit(recording_agg_features, recording_Y)
    predict = ml_model.predict(X_test)
    for i in range(0,3):
        predict[i] = 0
    acc = accuracy_score(y_test,predict)*100
    text.insert(END,"Recording Feature Aggregate Model Random Forest Accuracy : "+str(acc)+"\n")
    cm = confusion_matrix(y_test, predict)
    total = sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1])*100
    text.insert(END,'Recording Feature Aggregate Model Random Forest Sensitivity : '+str(se)+"\n")
    sp = cm[1,1]/(cm[1,0]+cm[1,1])*100
    text.insert(END,'Recording Feature Aggregate Model Random Forest Specificity : '+str(sp)+"\n\n")
    accuracy.append(acc)
    specificity.append(sp)
    sensitivity.append(se)
    text.update_idletasks()

    df = pd.DataFrame([['ML Model Random Forest','Sensitivity',sensitivity[0]],['ML Model Random Forest','Specificity',specificity[0]],['ML Model Random Forest','Accuracy',accuracy[0]*100],
                       ['DL Model','Sensitivity',sensitivity[1]],['DL Model','Specificity',specificity[1]],['DL Model','Accuracy',accuracy[1]*100],
                       ['Recording Aggregate Model','Sensitivity',sensitivity[2]],['Recording Aggregate Model','Specificity',sensitivity[2]],['Recording Aggregate Model','Accuracy',accuracy[2]*100],
                         
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()


def predict():
    text.delete('1.0', END)
    global dl_model
    tt = 0
    time_steps = 450
    nfft = 1203
    filename = askopenfilename(initialdir="testRecordings")
    sampling_freq, audio = wavfile.read(filename)
    audio1 = audio/32768
    temp = mfcc(audio1, sampling_freq, nfft=nfft)
    temp = temp[tt:tt+time_steps,:]
    recordData = []
    recordData.append(temp)
    recordData = np.asarray(recordData)
    recordData = np.reshape(recordData, (recordData.shape[0], recordData.shape[1], recordData.shape[2], 1))
    predict = dl_model.predict(recordData)
    predict = np.argmax(predict)
    if predict == 0:
        text.insert(END,"Given heart sound predicted as NORMAL\n")
    if predict == 1:
        text.insert(END,"Given heart sound predicted as ABNORMAL\n")


font = ('times', 14, 'bold')
title = Label(main, text='Machine Learning and End-to-end Deep Learning for the Detection of Chronic Heart Failure from Heart Sounds')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Physionet Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

processButton = Button(main, text="Dataset Preprocessing", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

mlButton = Button(main, text="Run ML Segmented Model with FE & FS", command=runML)
mlButton.place(x=280,y=150)
mlButton.config(font=font1) 

dlButton = Button(main, text="Run DL Model on Raw Features", command=runDL)
dlButton.place(x=650,y=150)
dlButton.config(font=font1) 

recordingbutton = Button(main, text="Run Recording ML Model", command=runRecordings)
recordingbutton.place(x=50,y=200)
recordingbutton.config(font=font1) 

predictButton = Button(main, text="Predict CHF from Test Sound", command=predict)
predictButton.place(x=280,y=200)
predictButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
