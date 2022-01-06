import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ###Using GPU 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ###Only warning and error information of tensorflow is displayed


def DeepRetention(len_depthseq, len_bpemb):

    inputDepthSeq = layers.Input(shape=(len_depthseq,1))
    inputBpEmb = layers.Input(shape=(len_bpemb,))
    inputLength = layers.Input(shape=(1,))

    DepthSeq = layers.Conv1D(4, 16, strides=2, activation='relu')(inputDepthSeq)
    DepthSeq = layers.Conv1D(4, 16, strides=2, activation='relu', padding="same")(DepthSeq)
    DepthSeq = layers.MaxPooling1D(2)(DepthSeq)
    DepthSeq = layers.Conv1D(16, 8, strides=2, activation='relu', padding="same")(DepthSeq)
    DepthSeq = layers.Conv1D(16, 8, strides=2, activation='relu', padding="same")(DepthSeq)
    DepthSeq = layers.MaxPooling1D(2)(DepthSeq)
    DepthSeq = layers.Conv1D(64, 1, strides=1, activation='relu', padding="same")(DepthSeq)
    DepthSeq = layers.Conv1D(64, 1, strides=1, activation='relu', padding="same")(DepthSeq)
    DepthSeq = layers.GlobalAveragePooling1D()(DepthSeq)

    concatall = layers.Concatenate()([DepthSeq, inputBpEmb, inputLength])
    hidden1 = layers.Dense(128, activation='relu')(concatall)
    hidden1 = layers.BatchNormalization()(hidden1)
    hidden1 = layers.Dropout(0.5)(hidden1)
    hidden2 = layers.Dense(32, activation='relu')(hidden1)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(0.3)(hidden2)
    output = layers.Dense(1, activation='sigmoid')(hidden2)

    model = Model(inputs=[inputDepthSeq, inputBpEmb, inputLength], outputs=output)
    model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])

    return model


if __name__ == "__main__":

    main_dir = "~/DeepRetention"
    os.chdir(main_dir)
    data_name = sys.argv[1]
    model_name = sys.argv[2]

    for p in range(2, 11): ### all
        '''parameter'''
        epochs = 1000
        lr = 0.1
        batch_size = 128
        n_splits = 10

        seed = 1024
        np.random.seed(seed)


        '''path'''
        data_dir = os.path.join(main_dir, "data")
        model_dir = os.path.join(main_dir, "point_model")

        train_file = os.path.join(data_dir, data_name, data_name+".dataset")
        converage = os.path.join(data_dir, data_name, data_name+'.ConveraeSeq.center'+'.'+str(p)+'p'+'.pkl')
        
        if "Human" in model_name:
            embedding = os.path.join(data_dir, data_name, data_name+".ReadSeq_human.pkl")
        elif "Mouse" in model_name:
            embedding = os.path.join(data_dir, data_name, data_name+".ReadSeq_mouse.pkl")
        else:
            print("please select correct ReadSeq.pkl")
            exit(1)


        '''dataset'''
        intron_data = pd.read_csv(train_file, header=None, sep="\t", low_memory=False)
        intron_data.columns = ["chr", "start", "end", "iREAD_label", "iREAD_score",
                            "IRFinder_label", "IRFinder_score", "or_label", "from"]
        bpCount = pd.read_pickle(converage).values # shape: (n, seq_len)

        bpEmb = pd.read_pickle(embedding).values
        length = (intron_data["end"]-intron_data["start"]).values.reshape(-1,1)
        label = intron_data["or_label"].values.reshape(-1,1)

        '''callbacks'''
        earlystop = callbacks.EarlyStopping(patience=10)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

        '''model'''
        len_depthseq = bpCount.shape[1]
        len_bpemb = bpEmb.shape[1]

        '''training'''
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for i, (tidx, vidx) in enumerate(kfold.split(length, label)):
            print(os.path.join(model_dir, model_name +'_'+str(p)+'p'+"_fold-"+str(i+1)))
            
            checkpoint = callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, model_name +'_'+str(p)+'p'+"_fold-"+str(i+1)),monitor='val_loss',mode="min",save_best_only='True')
            bpCount_t, bpEmb_t, len_t, label_t = bpCount[tidx], bpEmb[tidx], length[tidx], label[tidx]
            bpCount_v, bpEmb_v, len_v, label_v = bpCount[vidx], bpEmb[vidx], length[vidx], label[vidx]
            model = DeepRetention(len_depthseq, len_bpemb)
            history = model.fit([bpCount_t, bpEmb_t, len_t], label_t,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=([bpCount_v, bpEmb_v, len_v], label_v),
                            callbacks=[earlystop,reduce_lr,checkpoint],
                            class_weight=None,
                            sample_weight=None,
                            verbose=2)

            val_loss, val_acc = history.history['val_loss'][-1], history.history['val_accuracy'][-1]
            print("val_acc: %.2f%%, val_loss: %.4f" % (val_acc*100, val_loss))
        K.clear_session()
