import numpy as np
import pandas as pd
import math
import sys
import os
from random import shuffle
import h5py
import pickle
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from utilities import read_audio, create_folder
import config


parser = argparse.ArgumentParser(description='Example of parser. ')
subparsers = parser.add_subparsers(dest='mode')

parser_train = subparsers.add_parser('reading')
parser_train.add_argument('--audio_folder', type=str, required=True)
parser_train.add_argument('--meta_folder', type=str, required=True)
parser_train.add_argument('--workspace', type=str, required=True)
args = parser.parse_args()


#meta_folder = "../../data_split/demos_data/evaluation_setup/"
#audio_folder = "../../demos_data/wav_DEMoS/"
#workspace = "../../workspace"

meta_folder = args.meta_folder
audio_folder = args.audio_folder
workspace = args.workspace

create_folder(workspace)
hdf5_path = os.path.join(workspace, 'meta.h5')

# read csv
train_list = np.squeeze(pd.read_csv(os.path.join(meta_folder,"fold1_train.csv"), delimiter='\t').to_numpy())
devel_list = np.squeeze(pd.read_csv(os.path.join(meta_folder, "fold1_devel.csv"), delimiter='\t').to_numpy())
test_list = np.squeeze(pd.read_csv(os.path.join(meta_folder, "fold1_test.csv"), delimiter='\t').to_numpy())

# read audio to hdf5
x_train = []
x_devel = []
x_test = []
y_train = []
y_devel = []
y_test = []
filename_train = []
filename_devel = []
filename_test = []

print("start reading ...")
for i, audio_item in enumerate(train_list):
	audio_path = os.path.join(audio_folder, audio_item[0])
	audio, _ = read_audio(audio_path, target_fs=config.sample_rate)	
	x_train.append(audio)
	y_train.append(audio_item[1])
	filename_train.append(audio_item[0])
print("finishing reading %d training data ..." % (i+1))

for i, audio_item in enumerate(devel_list):
	audio_path = os.path.join(audio_folder, audio_item[0])
	audio, _ = read_audio(audio_path, target_fs=config.sample_rate)
	x_devel.append(audio)
	y_devel.append(audio_item[1])
	filename_devel.append(audio_item[0])
print("finishing reading %d devel data ..." % (i+1))

for i, audio_item in enumerate(test_list):
	audio_path = os.path.join(audio_folder, audio_item[0])
	audio, _ = read_audio(audio_path, target_fs=config.sample_rate)
	x_test.append(audio)
	y_test.append(audio_item[1])
	filename_test.append(audio_item[0])
print("finishing reading %d test data ..." % (i+1))


x_train = np.array(x_train, dtype=object)
x_devel = np.array(x_devel, dtype=object)
x_test = np.array(x_test, dtype=object)
y_train = np.array(y_train, dtype='S80')
y_devel = np.array(y_devel, dtype='S80')
y_test = np.array(y_test, dtype='S80')
filename_train = np.array(filename_train, dtype='S80')
filename_devel = np.array(filename_devel, dtype='S80')
filename_test = np.array(filename_test, dtype='S80')

# save data
dt = h5py.special_dtype(vlen=np.dtype('float32'))
with h5py.File(hdf5_path, 'w') as hf:
	hf.create_dataset("train_audio",  data=x_train, dtype=dt)
	hf.create_dataset("train_y", data=y_train, dtype='S80')
	hf.create_dataset("train_filename", data=filename_train, dtype='S80')
	hf.create_dataset("devel_audio",  data=x_devel, dtype=dt)
	hf.create_dataset("devel_y", data=y_devel, dtype='S80')
	hf.create_dataset("devel_filename", data=filename_devel, dtype='S80')
	hf.create_dataset("test_audio",  data=x_test, dtype=dt)
	hf.create_dataset("test_y", data=y_test, dtype='S80')
	hf.create_dataset("test_filename", data=filename_test, dtype='S80')
print('Save train, devel and test audio arrays to hdf5 located at {}'.format(hdf5_path))

hf.close()




