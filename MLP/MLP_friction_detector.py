from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from sklearn.metrics import f1_score
from tqdm import tqdm

# written by Stephan Raaijmakers

#preprocessing
def featurize_data(sentences): 
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    print("Featurizing data...be patient")

    X=[]
    Words={}
    for sentence in tqdm(sentences):
        #print(sentence)
        emb=model.encode([sentence])[0]
        X.append(list(emb))
        for word in sentence.split(" "):
            Words[word]=1
    return np.array(X), len(Words)


def read_sentence_data(fn): # format: dialogueID<TAB>turn<TAB>sentence<TAB>label
    sentences=[]
    labels=[]
    pos = 0
    fp=open(fn,"r")
    for line in fp:
        fields=line.rstrip().split("\t")
        if fields:
            sentences.append(fields[2])
            if fields[3] == '0' or fields[3] == '3':
              labels.append(float(0))
            else: 
              labels.append(float(1))
            
    fp.close()
    return sentences, labels    


def preprocess(fn):
    sentences, y=read_sentence_data(fn)
    X, vocab_len=featurize_data(sentences)  
    return X, np.array(y), vocab_len   


def create_model():
  model = Sequential()
  model.add(Dense(128,input_shape=(X.shape[1],1)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(32))
  model.add(Activation('sigmoid'))
  model.add(Dropout(0.1))

  model.add(Flatten())    
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  model.summary()
  opt=tf.keras.optimizers.Adam()
  
  # Compile the model
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model


def main(fn):
    X, y, vocab_len = preprocess(fn)
    max_len=len(X[0]) 
    acc_per_fold = []
    loss_per_fold = []
    f1_per_fold= []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=46)
    fold_no = 1
    for train, test in skf.split(X, y):
      X_train=np.array(X)[train.astype(int)]
      X_test=np.array(X)[test.astype(int)]
      y_train=np.array(y)[train.astype(int)]
      y_test=np.array(y)[test.astype(int)]
      model = create_model()

      # Generate a print
      print(f'Training for fold {fold_no} ...')

      # Fit data to model
      model.fit(X_train, y_train,batch_size=20,epochs=80,verbose=2)
      
      # Generate generalization metrics
      scores = model.evaluate(X_test, y_test, verbose=0)
      print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
      acc_per_fold.append(scores[1] * 100)
      loss_per_fold.append(scores[0])
      pred=model.predict(X_test) 
      pred=[p[0] for p in pred]
      pred=list(np.vectorize(lambda x: int(x >= 0.5))(pred))
      f1_per_fold.append(f1_score(y_test, pred))
      #print(classification_report(y_test, pred))
      # Increase fold number
      fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
      print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% - F1: {f1_per_fold[i]}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print(f'> F1: {np.mean(f1_per_fold)} (+- {np.std(f1_per_fold)})')
    print('------------------------------------------------------------------------')



if __name__ =="__main__":

  main("data/sentences_3cat_1:5.txt")
