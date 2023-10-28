import time
import sys
import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from sklearn import metrics

import scipy as sp
import shap
import pickle
import math

def fmt_gpu_mem_info(gpu_id=0, brief=True) -> str:
    import torch.cuda.memory

    if torch.cuda.is_available():
        report = ""
        t = torch.cuda.get_device_properties(gpu_id).total_memory
        c = torch.cuda.memory.memory_reserved(gpu_id)
        a = torch.cuda.memory_allocated(gpu_id)
        f = t - a

        report += f"[Allocated {a} | Free {f} | Cached {c} | Total {t}]\n"
        if not brief:
            report += torch.cuda.memory_summary(device=gpu_id, abbreviated=True)
        return report
    else:
        return f"CUDA not available, using CPU"

# setting the seed for reproducibility, same seed is set to ensure the reproducibility of the result
def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 

# specify GPU
device = torch.device("cuda")

# read the parameters 

dataset_path = sys.argv[1]
model_weights_path = sys.argv[2]
results_file = sys.argv[3]
data_name = sys.argv[4]
where_data_comes=data_name.split("-")[0]

# define CodeBERT model
#model_name = "microsoft/graphcodebert-base"
model_name = "microsoft/codebert-base"
model_config = AutoConfig.from_pretrained(model_name, return_dict=False, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
auto_model = AutoModel.from_pretrained(model_name, config=model_config)

# convert code into tokens and then vector representation
def tokenize_data(train_text, val_text, test_text):

    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=1024,
        pad_to_max_length=True,
        truncation=True)

    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=1024,
        pad_to_max_length=True,
        truncation=True)
    #print(test_text)
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=1024,
        pad_to_max_length=True,
        truncation=True)
    return tokens_train, tokens_val, tokens_test

# convert vector representation to tensors
def text_to_tensors(tokens_train, tokens_val, tokens_test):
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    return train_seq, train_mask, val_seq, val_mask, test_seq, test_mask

# sett seed for data_loaders for output reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
seed = 42 # any number 
set_deterministic(seed)

def data_loaders(train_seq, train_mask, train_y, val_seq, val_mask, val_y):
    # define a batch size
    batch_size = 16
    print('From data_loader')
    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    return train_dataloader, val_dataloader

# set up the neural network for CodeBERT fine-tuning
class BERT_Arch(nn.Module):

    def __init__(self, bert):

        super(BERT_Arch, self).__init__()

        self.bert = auto_model

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()

        #self.aap=nn.AdaptiveAvgPool1d(768)
        #self.fc1 = nn.Linear(768,512)
        # dense layer 1
        self.fc1 = nn.Linear(2*768, 512)

        # dense layer 2 (Output layer), For FlakiCat=4, IDOFT=6, binary-classification=2
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=-1)

    # define the forward pass
    def forward(self, sent_id, mask):

        # pass the inputs to the model
        #print(sent_id.shape)
        #print(mask.shape)
        chunk_size = 512
        #overlap_size = 256
        total_seq_length = sent_id.size(1)
        #print('***************=',total_seq_length)
        # Split the sequence into chunks of size chunk_size
        #cls_hs_list = []
        for start in range(0, total_seq_length, chunk_size):
            end = min(start + chunk_size, total_seq_length)
            chunk_sent_id = sent_id[:, start:end]
            chunk_mask = mask[:, start:end]
        # pass the inputs to the model
            cls_hs_current = self.bert(chunk_sent_id, attention_mask=chunk_mask)[1]
            #print('shantp=',cls_hs_current.shape)
            if start == 0:
                cls_hs = cls_hs_current.clone()
            else:
                cls_hs = torch.cat([cls_hs, cls_hs_current], dim=1)
        #cls_hs = self.bert(sent_id, attention_mask=mask)[1]
        #print(cls_hs.shape)
        #self.fc1.in_features = cls_hs.shape[1]

        #cls_hs_aap = self.aap(cls_hs)
        #fc1_output = self.fc1(cls_hs_aap)
        #print('shanpe outside=',cls_hs.shape)
        fc1_output = self.fc1(cls_hs)

        relu_output = self.relu(fc1_output)

        dropout_output = self.dropout(relu_output)

        # output layer
        fc2_output = self.fc2(dropout_output)

        # apply softmax activation
        final_output = self.softmax(fc2_output)

        return final_output

def format_time(time=None, format=None, rebase=True):
    """See :meth:`I18n.format_time`."""
    return get_i18n().format_time(time, format, rebase)


def get_evaluation_scores(tn, fp, fn, tp):
    print("get_score method is defined")
    if(tp == 0):
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        Precision = 0
        Recall = 0
        F1 = 0
    else:
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        Precision = tp/(tp+fp)
        Recall = tp/(tp+fn)
        F1 = 2*((Precision*Recall)/(Precision+Recall))
    return accuracy, F1, Precision, Recall

# give test data to the model in chunks to avoid Cuda out of memory error
'''def give_test_data_in_chunks(x_test_nparray): #BERT
    #X_test = X_test.tolist() if isinstance(X_test, np.ndarray) else X_test
    #x_test = pd.DataFrame(x_test_nparray) 
    x_test = x_test_nparray
    n = len(x_test) / 50 
    preds_chunks = None
    for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
        if isinstance(x_test_chunk, str):  # Check if x_test_chunk is a string
            x_test_chunk = [x_test_chunk]  # Convert it to a list
        tokens_test = tokenizer.batch_encode_plus(
            x_test_chunk.squeeze().tolist(), max_length=500, pad_to_max_length=True, truncation=True)
        #tokens_test = tokenizer.batch_encode_plus(_test_chunk.squeeze().tolist() if isinstance(x_test_chunk, pd.Series) else x_test_chunk,max_length=500, pad_to_max_length=True, truncation=True)
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()
        preds_chunks = preds_chunk if preds_chunks is None else np.append(
            preds_chunks, preds_chunk, axis=0)
        preds = np.argmax(preds_chunks, axis=1)

    return preds'''


def give_test_data_in_chunks(X_test_nparray):
    print("size=",X_test_nparray.shape)
    chunk_size = 50  # Number of examples per chunk
    num_chunks = math.ceil(len(X_test_nparray) / chunk_size)
    preds_chunks = None

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(X_test_nparray))
        x_test_chunk = X_test_nparray[start_idx:end_idx]

        tokens_test = tokenizer.batch_encode_plus(x_test_chunk.tolist(), max_length=1024, pad_to_max_length=True, truncation=True, return_tensors='pt')

        test_seq = tokens_test['input_ids'].squeeze()
        test_mask = tokens_test['attention_mask'].squeeze()
        print('#chunks=',num_chunks)
        print('test_seq_shape=',test_seq.shape)
        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()

        preds_chunks = preds_chunk if preds_chunks is None else np.append(preds_chunks, preds_chunk, axis=0)

    preds = np.argmax(preds_chunks, axis=1)
    return preds

execution_time = time.time()
print("Start time of the experiment", execution_time)
no_splits=10 # For FlakiCat=4, IDOFT=10
skf = StratifiedKFold(n_splits=no_splits,shuffle=True)
TN = FP = FN = TP = 0
fold_number = 0

exists1=os.path.exists(where_data_comes+"-result")
if not exists1:
    os.mkdir(where_data_comes+"-result")

if not os.path.exists(data_name):
    os.mkdir(data_name)

if not os.path.exists(data_name+'/data_splits'):
    os.mkdir(data_name+'/data_splits')

total_execution_time = 0
total_execution_time_for_feature_extraction = 0
Total_auc = 0
fold=0
#for train_index, test_index in skf.split(input_data, target_data):
for fold in range(no_splits):
    X_test=np.load(data_name+'/data_splits/X_test_fold'+str(fold)+'.npy',allow_pickle=True)
    y_test=np.load(data_name+'/data_splits/y_test_fold'+str(fold)+'.npy',allow_pickle=True)

    X_train=np.load(data_name+'/data_splits/X_train_fold'+str(fold)+'.npy',allow_pickle=True)
    y_train=np.load(data_name+'/data_splits/y_train_fold'+str(fold)+'.npy',allow_pickle=True)

    X_valid=np.load(data_name+'/data_splits/X_valid_fold'+str(fold)+'.npy',allow_pickle=True)
    y_valid=np.load(data_name+'/data_splits/y_valid_fold'+str(fold)+'.npy',allow_pickle=True)

    #X_train, y_train, X_valid, y_valid = sampling(X_train, y_train, X_valid, y_valid)
    
    tokens_train, tokens_val, tokens_test = tokenize_data(X_train, X_valid, X_test)

    '''with open("Example.txt", "a") as file:
        file.write("FOLD============"+str(fold))
        file.write(np.array2string(X_test))
        file.write('****************Tokens***************************')
        file.write(str(tokens_test))
        file.write("\n")'''


    Y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_valid)
    y_test = pd.DataFrame(y_test)

    Y_train.columns = ['which_tests']
    y_val.columns = ['which_tests']
    y_test.columns = ['which_tests']
    # convert labels of train, validation and test into tensors
    train_y = torch.tensor(Y_train['which_tests'].values)
    val_y = torch.tensor(y_val['which_tests'].values)
    test_y = torch.tensor(y_test['which_tests'].values)
    train_seq, train_mask, val_seq, val_mask, test_seq, test_mask = text_to_tensors(tokens_train, tokens_val, tokens_test)
    # create data_loaders for train and validation dataset
    train_dataloader, val_dataloader = data_loaders(train_seq, train_mask, train_y, val_seq, val_mask, val_y)
    # compute the class weights
    class_weights = compute_class_weight('balanced', np.unique(Y_train.values), y=np.ravel(Y_train.values))
    # convert list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)
    # push to GPU
    weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)
    model = BERT_Arch(auto_model)
    # push the model to GPU
    model = model.to(device)

    # define the optimizer
    #optimizer = AdamW(model.parameters(), lr=1e-5)
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(model_weights_path+str(fold_number)+'.pt'))
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        preds = give_test_data_in_chunks(X_test) 
        #preds = give_test_data_in_chunks(X_valid) #X_valid, y_valid
        #preds = give_test_data_in_chunks(X_train) #X_train, y_train
        total_execution_time+=(time.time() - start_time)

    cr=classification_report(test_y, preds)
    #cr=classification_report(y_valid, preds)
    #cr=classification_report(y_train, preds)
    print('I AM=',type(X_test))
        
   #np.savetxt('X_test.txt', X_test, delimiter=',', fmt='%f')
    X_test_str_array = X_test.astype(str)
    X_test_before_sep = [x.split('<SEP>')[0] for x in X_test_str_array]
    print(len(X_test_before_sep))
    key_value_pairs = {}
    
	# Create key-value pairs with indices
    for idx, item in enumerate(X_test_before_sep):
        if item in key_value_pairs:
            key_value_pairs[item].append(idx)
        else:
            key_value_pairs[item] = [idx]

    lists_of_duplicate_indices = list(key_value_pairs.values())
    print(lists_of_duplicate_indices)
    # Convert the list of extracted strings to a single string with a separator ('$* ' in this case)
    X_test_str = '$* '.join(X_test_before_sep)

    count_uniq_victim = 0
    uniq_victim_which_has_atleast_one_correct_polluter = 0
    with open(where_data_comes+"-result/Flaky-Test_Victim_polluter_"+str(no_splits)+".txt", "a") as file:
        file.write("Fold="+str(fold_number)+"\n")
        #file.write(str(lists_of_duplicate_indices))
        #file.write("\n")
        
        numpy_array_test_y=test_y.numpy()
        print('numpy_arr=',numpy_array_test_y)

        for indices in lists_of_duplicate_indices:
            count_uniq_victim +=1
            #print(indices)
            flag_matched=False
            for idx in indices:
                if numpy_array_test_y[idx] == preds[idx]:
                    #print('Matched')
                    flag_matched=True
                    break
            if flag_matched:
                uniq_victim_which_has_atleast_one_correct_polluter += 1
        print('Total-uniq-victim='+ str(count_uniq_victim) +", uniq_victim_which_has_atleast_one_correct_polluter="+str(uniq_victim_which_has_atleast_one_correct_polluter))


        file.write('Total-uniq-victim='+ str(count_uniq_victim) +", uniq_victim_which_has_atleast_one_correct_polluter="+str(uniq_victim_which_has_atleast_one_correct_polluter))
        file.write("\n")

    with open(where_data_comes+"-result/Bert_classification_report_"+str(no_splits)+".txt", "a") as file:
        file.write("Fold="+str(fold_number)+"\n")
        file.write(cr)
        file.write("\n")

    cm = confusion_matrix(test_y, preds)
    #cm = confusion_matrix(y_valid, preds)
    #cm = confusion_matrix(y_train, preds)
    #print(cm)
	
    with open(where_data_comes+"-result/Bert_confusion_matrix_"+str(no_splits)+".txt", "a") as file:
        file.write("Fold="+str(fold_number)+"\n")
        file.write(np.array2string(cm))
        file.write("\n")
        
    tn, fp, fn, tp = confusion_matrix(test_y, preds, labels=[0, 1]).ravel()
    #tn, fp, fn, tp = confusion_matrix(y_valid, preds, labels=[0, 1]).ravel()
    #tn, fp, fn, tp = confusion_matrix(y_train, preds, labels=[0, 1]).ravel()
    TN = TN + tn
    FP = FP + fp
    FN = FN + fn
    TP = TP + tp

    fpr, tpr, thresholds = metrics.roc_curve(test_y, preds, pos_label=1)
    #fpr, tpr, thresholds = metrics.roc_curve(y_valid, preds, pos_label=1)
    #fpr, tpr, thresholds = metrics.roc_curve(y_train, preds, pos_label=1)
    Total_auc += metrics.auc(fpr, tpr)

    print("delete model")
    del model
    torch.cuda.empty_cache()

    fold_number = fold_number+1

avg_auc=Total_auc/no_splits
# This part is needed for flaky_vs_non-flaky
accuracy, F1, Precision, Recall = get_evaluation_scores(TN, FP, FN, TP)
result = pd.DataFrame(columns = ['Accuracy','F1', 'Precision', 'Recall', 'TN', 'FP', 'FN', 'TP','Total_execution_time(Sec)','AUC'])
result = result.append(pd.Series([accuracy, F1, Precision, Recall, TN, FP, FN, TP, total_execution_time/no_splits,avg_auc], index=result.columns), ignore_index=True)
result.to_csv(results_file,  index=False)

print("The processed is completed in : (%s) seconds. " % round((time.time() - execution_time), 5))





