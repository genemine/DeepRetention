import os
import sys
import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from gensim.models.doc2vec import Doc2Vec
K.clear_session()

parser = argparse.ArgumentParser(description='model to predict intron retention')
parser.add_argument('-b','--bed', required=True, help='intron bed file, with processed sequence')
parser.add_argument('-c','--cov', required=True, help='converage sequence pkl')
parser.add_argument('-r','--read', required=True, help='read sequence pkl')
parser.add_argument('-m','--model', default='../model/DeepRetention_Human', help='DeepRetention dir')
parser.add_argument('-g','--gpu', default='-1', help='gpu id')
parser.add_argument('-o','--out', default='intron_retention.res', help='IR detection result')
parser.add_argument('-t','--threshold', type=float, default=0.5, help='the threshold to determine the IR label')
args = parser.parse_args()

gpu_id = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

bed_file = args.bed
converage_seq = args.cov
read_seq = args.read
model_name = args.model
result = args.out
threshold = args.threshold


if __name__ == "__main__":
    
    bpCount = pd.read_pickle(converage_seq) # shape: (n, seq_len)
    bpEmb = pd.read_pickle(read_seq)
    
    lens = bpCount.shape[1] # converage sequence length
    batch_size = 1024
    
    intron_bed = pd.read_csv(bed_file, header=None, sep="\t", low_memory=False)
    column_names = intron_bed.columns.tolist()
    column_names[:3] = ["chr", "start", "end"]
    dataset = tf.data.experimental.make_csv_dataset(bed_file, 
                                                    batch_size=batch_size, column_names=column_names,
                                                    label_name=None, select_columns=["start","end"],
                                                    field_delim="\t", header=False,
                                                    shuffle = False,
                                                    na_value="?", num_epochs=1)
    
    length = dataset.map(lambda x: tf.reshape(x["end"]-x["start"], [-1,1]))
    bpCount = tf.data.Dataset.from_tensor_slices(bpCount).map(lambda x: tf.reshape(x, [lens,1])).batch(batch_size)
    bpEmb = tf.data.Dataset.from_tensor_slices(bpEmb).batch(batch_size)

    # note: the input of predict is (data, target, weights), so we should regard three dataset as one tuple
    data = tf.data.Dataset.zip(((bpCount, bpEmb, length),)).prefetch(tf.data.experimental.AUTOTUNE)
    

    model_prefix = model_name+'_fold'
    scores = []
    for fold in range(1,11):
        model_name = model_prefix + "-" + str(fold)
        model = load_model(model_name)
        score = model.predict(data)
        val = [s[0] for s in score]
        scores.append(val)

    average_score = np.sum(scores, axis=0)/10
    average_score_pd = pd.DataFrame(average_score)
    average_score_pd.columns = ['IRprob']

    DeepRetention_label = [1 if score > threshold else 0 for score in average_score]
    DeepRetention_label_pd = pd.DataFrame(DeepRetention_label)
    DeepRetention_label_pd.columns = ['IR_label']

    pd_finally = pd.concat([intron_bed, average_score_pd, DeepRetention_label_pd], axis=1)
    pd_finally.to_csv(result, index=None, sep="\t")
    