import numpy as np
import soundfile
import librosa
import os
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):

    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1

    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def read_audio(path, target_fs=None):

    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def calculate_scalar(x):

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """

    samples_num = len(target)
    
    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):
        
        total[target[n]] += 1
        
        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy
        
    elif average == 'macro':
        return np.mean(accuracy)
        
    else:
        raise Exception('Incorrect average!')


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes

    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num, classes_num))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def print_accuracy(class_wise_accuracy, labels):

#    print('{:<30}{}'.format('Scene label', 'accuracy'))
#    print('------------------------------------------------')
#    for (n, label) in enumerate(labels):
#        print('{:<30}{:.3f}'.format(label, class_wise_accuracy[n]))
#    print('------------------------------------------------')
#    print('{:<30}{:.3f}'.format('Average', np.mean(class_wise_accuracy)))
    logging.info('{:<30}{}'.format('Emotion label', 'accuracy'))
    logging.info('------------------------------------------------')
    for (n, label) in enumerate(labels):
        logging.info('{:<30}{:.3f}'.format(label, class_wise_accuracy[n]))
    logging.info('------------------------------------------------')
    logging.info('{:<30}{:.3f}'.format('Average', np.mean(class_wise_accuracy)))

def plot_confusion_matrix(confusion_matrix, title, labels, values, path):
    """Plot confusion matrix.

    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal

    Ouputs:
      None
    """

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')
#    plt.show()


def model_summary(model, logging, detail=False):
    logging.info("model_summary")
    logging.info("Layer_name"+"\t\t\t\t\t\t"+"Number of Parameters")
    logging.info("========================================")
    if detail:
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            logging.info("{}: \t {}".format(name, params))
            total_params += params

        logging.info("Total Params: {}".format(total_params))      
    else:
        total_params = 0
        params_feat_extract = 0
        params_feat_project = 0
        params_encoder = 0
        params_encoder_layer = np.zeros((12))
        name_prior = ""
        idx = -1
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_params += params

            if "wav2vec2.feature_extractor" in name:
                params_feat_extract += params
                name_prior = name

            elif "wav2vec2.feature_projection" in name:
                if not "wav2vec2.feature_projection" in name_prior:
                    logging.info("{}: \t {}".format(".".join(name_prior.split('.')[:3]), params_feat_extract))
                params_feat_project +=params
                name_prior = name
          
            elif "wav2vec2.encoder.pos_conv_embed" in name or "wav2vec2.encoder.layer_norm" in name:
                if not "wav2vec2.encoder" in name_prior:
                    logging.info("{}: \t {}".format(".".join(name_prior.split('.')[:3]), params_feat_project))
                params_encoder +=params
                name_prior = name

            elif "wav2vec2.encoder.layers" in name:
                if not "wav2vec2.encoder.layers" in name_prior:
                    logging.info("{}: \t {}".format(".".join(name_prior.split('.')[:3]), params_encoder))

                if (not int(name.split('.')[4]) == idx) and idx != -1:                    
                    logging.info("{}: \t {}".format(".".join(name_prior.split('.')[:4]), params_encoder_layer[idx]))

                idx = int(name.split('.')[4])
                params_encoder_layer[idx] += params
                name_prior = name

            else:
                if "wav2vec2.encoder.layers" in name_prior:
                    logging.info("{}: \t {}".format(".".join(name_prior.split('.')[:4]), params_encoder_layer[-1]))
                logging.info("{}: \t {}".format(name, params))
                name_prior = name


def freeze_parameters(model, layer_all=False):
    
    if layer_all == True:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:   
        for name, param in model.named_parameters():
            if 'wav2vec2.encoder.layers' in name:
                layer_num = int(name.split('.')[4])
                if layer_num >= 2:  # freeze wav2vec2 from the 3rd encoder layer 
                    param.requires_grad = False



def freeze_parameters_partial(model, layer=None): 
    for name, param in model.named_parameters():
        if 'wav2vec2.encoder.layers' in name:
            layer_num = int(name.split('.')[4])
            if layer_num >= layer:  # freeze wav2vec2 from the 3rd encoder layer 
                param.requires_grad = False



