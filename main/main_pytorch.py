import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import label_binarize

from utilities import (create_folder, get_filename, create_logging, calculate_accuracy, print_accuracy, calculate_confusion_matrix, model_summary, freeze_parameters, freeze_parameters_partial)
import config

from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining

from models_pytorch import move_data_to_gpu, Wav2vec2DistillCNN, Wav2vec2DistillRNNnew, Wav2vec2DistillRNNTeacherStudent, Wav2vec2DistillCNNTeacherStudent, Wav2vec2FineTune

from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_metric

batch_size = 16
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
#distill_layer = [4, 8, 12] # 0-11
Model = Wav2vec2FineTune
distill_modeltype = 'GRU' # only when Model = Wav2vec2DistillRNNnew 'LSTM' 'GRU'
#feature_level = "embedding" # "embedding" "linear"
layer_fintune = 10

def fusion_pred(output1, output2):
    output = np.zeros((output1.shape[0], output1.shape[1]), dtype=np.float32)

    conf1 = np.sort(output1, axis=-1)
    confid1 = conf1[:,-1] - conf1[:, -2]

    conf2 = np.sort(output2, axis=-1)
    confid2 = conf2[:,-1] - conf2[:, -2]

    for i in range(0, len(output1)):
        if confid1[i] >= confid2[i]:
            output[i] = output1[i]
        else:
            output[i] = output2[i]

    return output

def evaluate(model, data_loader, distill_layer, cuda, class_wise=False):
    """Evaluate
    
    Returns:
      accuracy: float
    """
    if class_wise == True:
        avg = None
    else:
        avg = 'macro'

    outputs, targets, outputs_distills, outputs_feature, outputs_distfeature = forward(model, data_loader, distill_layer, cuda)

    # loss
    loss_fct = nn.CrossEntropyLoss()
    loss = float(loss_fct(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy())
    
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    softm = nn.Softmax(dim=-1)
    loss_mse = nn.MSELoss()
    loss_distill = 0
    loss_kld = 0
    loss_fea = 0
    for (idx, output_distill) in enumerate(outputs_distills, 0):
        loss_distill = loss_distill + float(loss_fct(Variable(torch.Tensor(output_distill)), Variable(torch.LongTensor(targets))).data.numpy())
        loss_kld = loss_kld + float(loss_kl(softm(Variable(torch.Tensor(output_distill))), softm(Variable(torch.Tensor(outputs)))))
        loss_fea = loss_fea + loss_mse(Variable(torch.Tensor(outputs_distfeature[idx])), Variable(torch.Tensor(outputs_feature)))           


    # UAR
    classes_num = outputs.shape[-1]
    predictions = np.argmax(outputs, axis=-1)
    uar = calculate_accuracy(targets, predictions, classes_num, average=avg)

    uar_distill = []
    for (idx, output_distill) in enumerate(outputs_distills, 0):
        predictions_distill = np.argmax(output_distill, axis=-1)
        uar_d = calculate_accuracy(targets, predictions_distill, classes_num, average=avg)
        uar_distill.append(uar_d)

        if idx == 0: 
            fusion = output_distill
        if idx > 0:
            fusion = fusion_pred(fusion, output_distill)

    if len(distill_layer) > 1:    
        predictions_distill = np.argmax(fusion, axis=-1)
        uar_d = calculate_accuracy(targets, predictions_distill, classes_num, average=avg)
        uar_distill.append(uar_d)    


    return [loss, loss_distill, loss_kld, loss_fea], uar, uar_distill
            

def forward(model, data_loader, distill_layer, cuda):
    
    outputs = []
    targets = []    
    outputs_feature = []

    outputs_distill = []
    outputs_distfeature = []
    for i in range(0, len(distill_layer)):
        outputs_distill.append([])
        outputs_distfeature.append([])


    for (idx, (batch_x, batch_y)) in enumerate(data_loader, 0):
        
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        
        model.eval()
        batch_output, batch_output_distill, batch_feature = model(batch_x)

        outputs.append(batch_output.data.cpu().numpy())
        targets.append(batch_y.data.cpu().numpy())
        outputs_feature.append(batch_feature[0].data.cpu().numpy())
        for i in range(0, len(distill_layer)):
            outputs_distill[i].append(batch_output_distill[i].data.cpu().numpy())
            outputs_distfeature[i].append(batch_feature[i+1].data.cpu().numpy())


    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0) 
    outputs_feature = np.concatenate(outputs_feature, axis=0)

    for i in range(0, len(distill_layer)):
        outputs_distill[i] = np.concatenate(outputs_distill[i], axis=0) 
        outputs_distfeature[i] = np.concatenate(outputs_distfeature[i], axis=0)         

    return outputs, targets, outputs_distill, outputs_feature, outputs_distfeature




def audio_unify(audio_wav, seq_len=config.seq_len):
    if len(audio_wav) < seq_len:
        stack_n1 = math.floor(seq_len / len(audio_wav))
        #stack_n2 = int(seq_len % len(audio_wav))
        audio_new = np.tile(audio_wav, stack_n1+1)
        audio_new = audio_new[:seq_len]

        #audio_temp = audio_wav
        #for i in range(1, stack_n1):
        #   audio_temp = np.hstack((audio_temp, audio_wav))
        #audio_new = np.hstack((audio_temp, audio_wav[0:stack_n2]))
    else:
        audio_new = audio_wav[:seq_len]

    #if True in np.isnan(audio_new):
    #    print('There is nan in the new audio')
    #if len(audio_new) != seq_len:
    #    print('Wrongly unifiying audio length!')

    return audio_new

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

def data_generater(hdf5_path, validation, classes):
    '''Read data into a dict'''
    with h5py.File(hdf5_path, 'r') as hf:
        x_train = hf['train_audio'][:]
        y_train = hf['train_y'][:]
        x_devel = hf['devel_audio'][:]
        y_devel = hf['devel_y'][:]
        x_test = hf['test_audio'][:]
        y_test = hf['test_y'][:]

    hf.close()

    if validation:
        y_train = label_binarize([x.decode() for x in y_train], classes=classes)
        y_train = np.argmax(y_train, axis=-1)
        y_devel = label_binarize([x.decode() for x in y_devel], classes=classes)
        y_devel = np.argmax(y_devel, axis=-1)

        d = {'train':Dataset.from_dict({'label':y_train,'audio':x_train}), 'test':Dataset.from_dict({'label':y_devel,'audio':x_devel})}

    else:
        x_train = np.concatenate((x_train, x_devel), axis=0)
        y_train = np.concatenate((y_train, y_devel), axis=None)

        y_train = label_binarize([x.decode() for x in y_train], classes=classes)
        y_train = np.argmax(y_train, axis=-1)
        y_test = label_binarize([x.decode() for x in y_test], classes=classes)
        y_test = np.argmax(y_test, axis=-1)

        d = {'train':Dataset.from_dict({'label':y_train,'audio':x_train}), 'test':Dataset.from_dict({'label':y_test,'audio':x_test})}

    return d


def preprocess_function(audio_examples):
    audio_arrays = [x for x in audio_examples['audio']]
    inputs = feature_extractor(audio_arrays, sampling_rate=feature_extractor.sampling_rate)
    return inputs



########################### Fine tune ###############################

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric("recall")

    return metric.compute(predictions=predictions, references=labels, average='macro')


def evaluate_finetune(model, data_loader, cuda):
    """Evaluate
    
    Returns:
      accuracy: float
    """

    outputs, targets= forward_finetune(model, data_loader, cuda)

    # loss
    loss_fct = nn.CrossEntropyLoss()
    loss = float(loss_fct(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy())

    # UAR
    classes_num = outputs.shape[-1]
    predictions = np.argmax(outputs, axis=-1)
    uar = calculate_accuracy(targets, predictions, classes_num, average='macro')

    return loss, uar
            

def forward_finetune(model, data_loader, cuda):
    
    outputs = []
    targets = []    

    for (idx, (batch_x, batch_y)) in enumerate(data_loader, 0):
        
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        
        model.eval()
        batch_output = model(batch_x)

        outputs.append(batch_output.data.cpu().numpy())
        targets.append(batch_y.data.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)       

    return outputs, targets

def train(args):

    # Arugments & parameters
    workspace = args.workspace
    validation = args.validation
    epoch = args.epoch
    mini_data = args.mini_data
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    hdf5_path = os.path.join(workspace, "meta.h5")

    if validation:                     
        models_dir = os.path.join(workspace, 'models','train_devel')
                                        
    else:
        models_dir = os.path.join(workspace, 'models','traindevel_test')

    create_folder(models_dir)

    # data
    data = data_generater(hdf5_path, validation, labels)
    dataset = DatasetDict(data)
    dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=batch_size)

    # model loading
    model_deep = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
    model = Model(model_deep, classes_num, layer_fintune)
    freeze_parameters_partial(model, layer=layer_fintune) 
    model_summary(model, logging, detail=True)
    # model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=classes_num)

    if cuda:
        model.cuda()

    # unify data
    print('Data unifying')
    dataset_train = TensorDataset(torch.Tensor([audio_unify(x) for x in dataset['train']['input_values']]), torch.LongTensor(dataset['train']['label']))
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_test = TensorDataset(torch.Tensor([audio_unify(x) for x in dataset['test']['input_values']]), torch.LongTensor(dataset['test']['label']))
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)    

    del data
    del dataset

    # training
    print('Start training ...')
    lr = 3e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    loss_fct = nn.CrossEntropyLoss()  

    for epoch_idx in range(0, epoch): 
        logging.info('epoch: {}'.format(epoch_idx))
        for (idx, (batch_x, batch_y)) in enumerate(trainloader, 0):
            batch_x = move_data_to_gpu(batch_x, cuda)
            batch_y = move_data_to_gpu(batch_y, cuda)

            model.train()
            batch_output = model(batch_x)
        
            loss = loss_fct(batch_output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        tr_loss, tr_uar = evaluate_finetune(model, trainloader, cuda)
        te_loss, te_uar = evaluate_finetune(model, testloader, cuda)
        logging.info('train_acc: {:.3f}, train_loss: {:.3f}'.format(tr_uar, tr_loss))
        logging.info('test_acc: {:.3f}, test_loss: {:.3f}'.format(te_uar, te_loss))

        # save model
        if epoch_idx + 1 == epoch:
            save_out_dict = {'epoch': epoch_idx + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                            }
            save_out_path = os.path.join(models_dir, "md_{}_epoch.tar".format(epoch_idx + 1))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))

    #model.freeze_feature_extractor() 
    '''
    training_args = TrainingArguments(
        output_dir=models_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=3e-5,
        warmup_ratio = 0.1,
        save_total_limit = 2,
        num_train_epochs=epoch,
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )
    trainer.train()
    '''
    # trainer.evaluate()



####################### Self Distill #######################

def train_distill(args):
    # Arugments & parameters
    workspace = args.workspace
    validation = args.validation
    epoch = args.epoch
    distill_layer_str = args.distill_layer_str
    feature_level = args.feature_level
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    logging.info('Distill layer: {}'.format(distill_layer_str))
    distill_layer_list = distill_layer_str[1:-1].split(',')
    distill_layer = [int(i) for i in distill_layer_list]
    #distill_layer = [int(distill_layer_str[i]) for i in range(1, len(distill_layer_str)-1) if distill_layer_str[i] != ',']

    hdf5_path = os.path.join(workspace, "meta.h5")

    if validation:                     
        models_dir = os.path.join(workspace, 'models_'+feature_level, distill_layer_str, 'train_devel')
                                        
    else:
        models_dir = os.path.join(workspace, 'models_'+feature_level, distill_layer_str, 'traindevel_test')

    create_folder(models_dir)
    logging.info('Feature level: {}'.format(feature_level))
    logging.info('Model folder: {}'.format(models_dir))

    # data
    data = data_generater(hdf5_path, validation, labels)
    dataset = DatasetDict(data)
    dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=batch_size)

    # model loading
    #model_deep = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", num_labels=classes_num, output_hidden_states=True)  
    model_deep = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)

    debug = False
    if debug:
        a = dataset['train']['input_values'][0][:15898]
        data_test = torch.Tensor([a, dataset['train']['input_values'][1]])
        print(data_test.size())

        print(model_deep)
        for name, param in model_deep.named_parameters():
            if param.requires_grad:
                print(name)
                print(param.size())

        output = model_deep(data_test)
        hidden_states = output.hidden_states
        print(len(hidden_states))
        for layer, hidden_state in enumerate(hidden_states):
            print(f"Layer {layer} hidden: ")
            print(hidden_state.size())
    
    model = Model(model_deep, classes_num, distill_layer, feature_level, distill_modeltype)
    model_summary(model, logging)
    #if torch.cuda.device_count()>1:
    #    print('CUDA device count: ' + str(torch.cuda.device_count()))
    #    models = nn.DataParallel(model, device_ids=[0,1])
    if cuda:
        model.cuda()

    debug = False
    if debug:
        data_test = move_data_to_gpu(torch.rand(2, 10000), cuda)
        outputs, outputs_distill = model(data_test)    
        print(outputs.size()) 
        print(outputs_distill.size()) 

    # unify data
    print('Data unifying')
    #x_train = [audio_unify(x) for x in dataset['train']['input_values']]
    dataset_train = TensorDataset(torch.Tensor([audio_unify(x) for x in dataset['train']['input_values']]), torch.LongTensor(dataset['train']['label']))
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    #x_test = [audio_unify(x) for x in dataset['test']['input_values']]
    dataset_test = TensorDataset(torch.Tensor([audio_unify(x) for x in dataset['test']['input_values']]), torch.LongTensor(dataset['test']['label']))
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)    

    del data
    del dataset

    print('Start training ...')
    lr = 3e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    loss_fct = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    softm = nn.Softmax(dim=-1)
    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()
    loss_cos = nn.CosineSimilarity()    

    for epoch_idx in range(0, epoch): 
        logging.info('epoch: {}'.format(epoch_idx))
        for (idx, (batch_x, batch_y)) in enumerate(trainloader, 0):
            batch_x = move_data_to_gpu(batch_x, cuda)
            batch_y = move_data_to_gpu(batch_y, cuda)

            model.train()
            batch_output, batch_output_distill, batch_output_feature = model(batch_x)
        
            loss = loss_fct(batch_output, batch_y)
            loss_distill = 0
            loss_kld = 0
            loss_fea = 0
            for (output_id, batch_output_d) in enumerate(batch_output_distill, 0):            
                loss_distill = loss_distill + loss_fct(batch_output_d, batch_y)
                loss_kld = loss_kld + loss_kl(softm(batch_output_d), softm(batch_output))
                
                loss_fea = loss_fea + loss_mse(batch_output_feature[output_id+1], batch_output_feature[0]) 
                #loss_fea = loss_fea + loss_l1(batch_output_feature[output_id+1], batch_output_feature[0])
                #loss_cos_value = torch.log(torch.special.expit(torch.mean(loss_cos(batch_output_feature[output_id+1], batch_output_feature[0]))))
                #loss_fea = loss_fea - loss_cos_value
                #loss_fea = loss_fea + loss_mse(batch_output_feature[output_id+1], batch_output_feature[0]) - loss_cos_value
                #loss_fea = loss_fea + loss_l1(batch_output_feature[output_id+1], batch_output_feature[0]) - loss_cos_value
              
            loss_distill = loss_distill/len(distill_layer)
            loss_kld = loss_kld/len(distill_layer)
            loss_fea = loss_fea/len(distill_layer)
            
            loss = loss + loss_distill + loss_kld + loss_fea

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        tr_loss, tr_uar, tr_uar_d = evaluate(model, trainloader, distill_layer, cuda)
        te_loss, te_uar, te_uar_d = evaluate(model, testloader, distill_layer, cuda)
        logging.info('deepest_layer, train_acc: {:.3f}, train_loss: {:.3f}, train_loss_distill: {:.3f}, train_loss_kl: {:.3f}, train_loss_fea: {:.3f}'.format(tr_uar, tr_loss[0], tr_loss[1], tr_loss[2], tr_loss[3]))
        for d_i in range(0, len(distill_layer)):
            logging.info('distill_layer: {:.1f}, train_acc_d: {:.3f}'.format(distill_layer[d_i], tr_uar_d[d_i]))
        if len(distill_layer) > 1: 
            logging.info('fusion, train_acc_d: {:.3f}'.format(tr_uar_d[-1]))


        logging.info('deepest layer, test_acc: {:.3f}, test_loss: {:.3f}, test_loss_distill: {:.3f}, test_loss_kl: {:.3f}, test_loss_fea: {:.3f}'.format(te_uar, te_loss[0], te_loss[1], te_loss[2], te_loss[3]))
        for d_i in range(0, len(distill_layer)):
            logging.info('distill_layer: {:.1f}, test_acc_d: {:.3f}'.format(distill_layer[d_i], te_uar_d[d_i]))
        if len(distill_layer) > 1: 
            logging.info('fusion, test_acc_d: {:.3f}'.format(te_uar_d[-1]))

        if epoch_idx + 1 == epoch:
            te_loss, te_uar, te_uar_d = evaluate(model, testloader, distill_layer, cuda, class_wise=True)
            logging.info('\ndeepest layer:')
            print_accuracy(te_uar, labels)
            for d_i in range(0, len(distill_layer)):
                logging.info('\ndistill_layer: {:.1f}'.format(distill_layer[d_i]))                
                print_accuracy(te_uar_d[d_i], labels)
            if len(distill_layer) > 1:
                logging.info('\nfusion:')
                print_accuracy(te_uar_d[-1], labels)

        # save model
        if epoch_idx + 1 == epoch:
            save_out_dict = {'epoch': epoch_idx + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                            }
            save_out_path = os.path.join(models_dir, "md_{}_epoch.tar".format(epoch_idx + 1))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))


#################################

################## Distill Teacher Student ######################

def evaluate_distill_teacher_student(model, model_deep, data_loader, distill_layer, cuda):

    outputs, targets, outputs_distill, target_distill = forward_distill_teacher_student(model, model_deep, data_loader, distill_layer, cuda)

    # loss
    loss_fct = nn.CrossEntropyLoss()
    loss = float(loss_fct(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy())
    
    loss_fea = 0
    loss_mse = nn.MSELoss()
    for idx in range(0, len(distill_layer)):
        loss_fea = loss_fea + loss_mse(Variable(torch.Tensor(outputs_distill[idx])), Variable(torch.Tensor(target_distill[idx])))           


    # UAR
    classes_num = outputs.shape[-1]
    predictions = np.argmax(outputs, axis=-1)
    uar = calculate_accuracy(targets, predictions, classes_num, average='macro')

    return loss, loss_fea, uar

def forward_distill_teacher_student(model, model_deep, data_loader, distill_layer, cuda):
    '''Return:
    outputs: Prediction on classes of hidden layer 2
    targets: labels
    outputs_distill: distill layers 
    target_distill: predicted distill layers
    '''
    
    outputs = []
    targets = []    
    outputs_distill = []
    target_distill = []
    for i in range(0, len(distill_layer)):
        outputs_distill.append([])
        target_distill.append([])

    for (idx, (batch_x, batch_y)) in enumerate(data_loader, 0):
        
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        
        model.eval()
        batch_output, batch_distill = model(batch_x)  

        batch_distill_target = []
        model_deep.eval()
        batch_hidden = model_deep(batch_x).hidden_states
        for i in range(0, len(distill_layer)):
            batch_d = torch.transpose(batch_hidden[distill_layer[i]], 1, 2)
            batch_d = F.avg_pool1d(batch_d, kernel_size=batch_d.shape[2])
            batch_d = batch_d.view(batch_d.shape[0:2])
            batch_distill_target.append(batch_d)

        outputs.append(batch_output.data.cpu().numpy())
        targets.append(batch_y.data.cpu().numpy())
        for i in range(0, len(distill_layer)):
            outputs_distill[i].append(batch_distill[i].data.cpu().numpy())
            target_distill[i].append(batch_distill_target[i].data.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0) 
    for i in range(0, len(distill_layer)):
        outputs_distill[i] = np.concatenate(outputs_distill[i], axis=0) 
        target_distill[i] = np.concatenate(target_distill[i], axis=0)         

    return outputs, targets, outputs_distill, target_distill




# https://aaai-sas-2022.github.io/static/media/DistilHuBERT_AAAI_SAS_2022.7d068153.pdf
def train_distill_teacherstudent(args):
    # Arugments & parameters
    workspace = args.workspace
    validation = args.validation
    epoch = args.epoch
    epoch_trans = args.epoch_trans
    distill_layer_str = args.distill_layer_str
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    logging.info('Distill layer: {}'.format(distill_layer_str))
    distill_layer_list = distill_layer_str[1:-1].split(',')
    distill_layer = [int(i) for i in distill_layer_list]
    #distill_layer = [int(distill_layer_str[i]) for i in range(1, len(distill_layer_str)-1) if distill_layer_str[i] != ',']

    hdf5_path = os.path.join(workspace, "meta.h5")

    if validation:                     
        models_dir = os.path.join(workspace, 'models',distill_layer_str, 'train_devel')
                                        
    else:
        models_dir = os.path.join(workspace, 'models',distill_layer_str, 'traindevel_test')

    create_folder(models_dir)

    # data
    data = data_generater(hdf5_path, validation, labels)
    dataset = DatasetDict(data)
    dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=batch_size)

    # model loading
    model_deep = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
    model_copy = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
    model = Model(model_copy, classes_num, distill_layer, distill_modeltype)

    freeze_parameters(model_deep, layer_all=True)
    freeze_parameters(model)
    model_summary(model, logging, detail=True)
    
    #if torch.cuda.device_count()>1:
    #    print('CUDA device count: ' + str(torch.cuda.device_count()))
    #    models = nn.DataParallel(model, device_ids=[0,1])
    if cuda:
        model_deep.cuda()
        model.cuda()

    # unify data
    print('Data unifying')
    dataset_train = TensorDataset(torch.Tensor([audio_unify(x) for x in dataset['train']['input_values']]), torch.LongTensor(dataset['train']['label']))
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_test = TensorDataset(torch.Tensor([audio_unify(x) for x in dataset['test']['input_values']]), torch.LongTensor(dataset['test']['label']))
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)    

    del data
    del dataset

    print('Start training ...')
    lr = 3e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    loss_fct = nn.CrossEntropyLoss()
    #loss_kl = nn.KLDivLoss(reduction="batchmean")
    #softm = nn.Softmax(dim=-1)
    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()
    loss_cos = nn.CosineSimilarity()

    for epoch_idx in range(0, epoch): 
        logging.info('epoch: {}'.format(epoch_idx))
        if epoch_idx == epoch_trans:
            logging.info('Start finetuning from the {}-th epoch'.format(epoch_idx))        
        for (idx, (batch_x, batch_y)) in enumerate(trainloader, 0):
            batch_x = move_data_to_gpu(batch_x, cuda)
            batch_y = move_data_to_gpu(batch_y, cuda)

            model.train()
            batch_output, batch_output_feature = model(batch_x)

            batch_distill = []
            model_deep.eval()
            batch_hidden = model_deep(batch_x).hidden_states
            for i in range(0, len(distill_layer)):
                batch_d = torch.transpose(batch_hidden[distill_layer[i]], 1, 2)
                batch_d = F.avg_pool1d(batch_d, kernel_size=batch_d.shape[2])
                batch_d = batch_d.view(batch_d.shape[0:2])
                batch_distill.append(batch_d)

        
            if epoch_idx < epoch_trans:
                loss = 0
                for (output_id, batch_output_f) in enumerate(batch_output_feature, 0):            
                    loss_mse_value = loss_mse(batch_output_f, batch_distill[output_id])
                    loss_l1_value = loss_l1(batch_output_f, batch_distill[output_id])
                    loss_cos_value = torch.mean(loss_cos(batch_output_f, batch_distill[output_id]))
                    
                    #loss = loss + loss_mse_value
                    #loss = loss + loss_l1_value 
                    loss = loss + loss_l1_value - torch.log(torch.special.expit(loss_cos_value)) # torch.special.expit = torch.sigmoid
                    #loss = loss + loss_mse_value - torch.log(torch.special.expit(loss_cos_value))

                loss = loss/len(distill_layer)

            else: 
                loss = loss_fct(batch_output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        tr_loss, tr_loss_distill, tr_uar = evaluate_distill_teacher_student(model, model_deep, trainloader, distill_layer, cuda)
        te_loss, te_loss_distill, te_uar = evaluate_distill_teacher_student(model, model_deep, testloader, distill_layer, cuda)
        logging.info('deepest_layer, train_acc: {:.3f}, train_loss: {:.3f}, tr_loss_distill: {:.3f}'.format(tr_uar, tr_loss, tr_loss_distill))
        logging.info('deepest layer, test_acc: {:.3f}, test_loss: {:.3f}, te_loss_distill: {:.3f}'.format(te_uar, te_loss, te_loss_distill))

        # save model
        if epoch_idx + 1 == epoch:
            save_out_dict = {'epoch': epoch_idx + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                            }
            save_out_path = os.path.join(models_dir, "md_{}_epoch.tar".format(epoch_idx + 1))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validation', action='store_true', default=False)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    
    parser_train_distill = subparsers.add_parser('train_distill')
    parser_train_distill.add_argument('--workspace', type=str, required=True)
    parser_train_distill.add_argument('--validation', action='store_true', default=False)
    parser_train_distill.add_argument('--epoch', type=int, required=True)
    parser_train_distill.add_argument('--distill_layer_str', type=str, required=True)
    parser_train_distill.add_argument('--feature_level', type=str, required=True)
    parser_train_distill.add_argument('--cuda', action='store_true', default=False)


    parser_train_distill_teacherstudent = subparsers.add_parser('train_distill_teacherstudent')
    parser_train_distill_teacherstudent.add_argument('--workspace', type=str, required=True)
    parser_train_distill_teacherstudent.add_argument('--validation', action='store_true', default=False)
    parser_train_distill_teacherstudent.add_argument('--epoch', type=int, required=True)
    parser_train_distill_teacherstudent.add_argument('--epoch_trans', type=int, required=True)
    parser_train_distill_teacherstudent.add_argument('--distill_layer_str', type=str, required=True)
    parser_train_distill_teacherstudent.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    if args.mode == 'train_distill':
        logs_dir = os.path.join(args.workspace, 'logs_'+args.feature_level, args.filename)
    else:
        logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'train_distill':
        train_distill(args)

    elif args.mode == 'train_distill_teacherstudent':
        train_distill_teacherstudent(args)        

    else:
        raise Exception('Error argument!')

