import numpy as np
import pandas as pd
import wave
import librosa
import re
# from allennlp.commands.elmo import ElmoEmbedder
import os
prefix = '/home/youjiajun/data/EATD/EATD-Corpus'
from elmoformanylangs import Embedder
# import pkuseg
# import thulac
# from pyhanlp import HanLP
import jieba
# seg = pkuseg.pkuseg()
# thu1 = thulac.thulac(seg_only=True)
elmo = Embedder('/home/youjiajun/data/pre_trained_model/Text/zhs.model')

topics = ['positive', 'neutral', 'negative']
answers = {}
text_features = []
text_targets = []

def extract_features(text_features, text_targets, path):
    for index in range(114):
        if os.path.isdir(os.path.join(prefix, f'{path}_{str(index+1)}')):
            answers[index+1] = []
            for topic in topics:
                with open(os.path.join(prefix, f'{path}_{str(index+1)}', '%s.txt'%(topic)) ,'r') as f:
                    lines = f.readlines()[0]
                    # seg_text = seg.cut(lines) 
                    # seg_text = thu1.cut(lines)
                    # seg_text_iter = HanLP.segment(lines) 
                    seg_text_iter = jieba.cut(lines, cut_all=False) 
                    answers[index+1].append([item for item in seg_text_iter])
                    # answers[dir].append(seg_text)
            with open(os.path.join(prefix, '{1}_{0}/new_label.txt'.format(index+1, path))) as fli:
                target = float(fli.readline())
            # text_targets.append(1 if target >= 53 else 0)
            text_targets.append(1 if target >= 53 else 0)
            text_features.append([np.array(item).mean(axis=0) for item in elmo.sents2elmo(answers[index+1])])

extract_features(text_features, text_targets, 't')
extract_features(text_features, text_targets, 'v')

print("Saving npz file locally...")
np.savez(os.path.join('/home/youjiajun/data/EATD', 'Features/TextWhole/whole_samples_clf_avg.npz'), text_features)
np.savez(os.path.join('/home/youjiajun/data/EATD', 'Features/TextWhole/whole_labels_clf_avg.npz'), text_targets)
    