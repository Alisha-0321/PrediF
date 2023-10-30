import time
import sys
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy as sp
import shap
import pickle

def custom_masked_bar_plot(class_index,mask_type,viz_type):
    #determine type of operation on the explanation object
    if viz_type=='mean':
        compute_shap=copy.copy(shap_values_multiclass.mean(0))
    if viz_type=='sum':
        compute_shap=copy.copy(shap_values_multiclass.sum(0))
    if viz_type=='abs_mean':
        compute_shap=copy.copy(shap_values_multiclass.abs.sum(0))
    if viz_type=='abs_sum':
        compute_shap=copy.copy(shap_values_multiclass.abs.sum(0))
    #create a mask to visualize either positively or negatively contributing features
    if mask_type=='pos':
        mask=compute_shap.values[:,class_index]>=0
    else:
        mask=compute_shap.values[:,class_index]<=0
    #slice values related to a given class
    compute_shap.values=compute_shap.values[:,class_index][mask]
    compute_shap.feature_names=list(np.array(compute_shap.feature_names)[mask])

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

def f(x_test):
    n = len(x_test) / 50 
    preds_chunks = None
    #for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
    tokens_test = tokenizer.batch_encode_plus(
        x_test.tolist(), max_length=1024, pad_to_max_length=True, truncation=True)
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    preds_chunk = model(test_seq.to(device), test_mask.to(device))
    preds_chunk = preds_chunk.detach().cpu().numpy()
    preds_chunks = preds_chunk if preds_chunks is None else np.append(
        preds_chunks, preds_chunk, axis=0)

    scores = (np.exp(preds_chunks).T / np.exp(preds_chunks).sum(-1)).T
    val = sp.special.logit(scores) 
    return val

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

df = pd.read_csv(dataset_path)
#input_data = df['full_code'] # use the 'full_code' column to run Flakify using the full code instead of pre-processed code
#target_data = df['category']
if data_name == "Victim_Polluter_Pair-Data":
    target_data = df['isVictimPolluterPair']
    df['Merged-Content'] = df['victim_code'] + ' ' + df['p_or_np_code']

elif data_name == "Victim_Cleaner_Polluter_Pair-Data":
    target_data = df['isVictimPolluterCleanerPair']
    df['Merged-Content'] = df['victim_code'] + ' ' + df['c_or_nc_code'] + ' ' + df['polluter_code']

elif data_name == "Brittle_State-Setter_Pair-Data":
    target_data = df['isBSSPair']
    df['Merged-Content'] = df['brittle_code'] + ' ' + df['ss_or_nss_code']

input_data=df['Merged-Content']
df.head()

# balance dataset,converts a 1D Pandas DataFrame into a 2D NumPy array
def sampling(X_train, y_train, X_valid, y_valid):
    
    oversampling = RandomOverSampler(
        sampling_strategy='minority', random_state=49)
    x_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    x_val = X_valid.values.reshape(-1, 1)
    y_val = y_valid.values.reshape(-1, 1)
    
    x_train, y_train = oversampling.fit_resample(x_train, y_train)
    x_val, y_val = oversampling.fit_resample(x_val, y_val)
    x_train = pd.Series(x_train.ravel())
    y_train = pd.Series(y_train.ravel())
    x_val = pd.Series(x_val.ravel())
    y_val = pd.Series(y_val.ravel())

    return x_train, y_train, x_val, y_val


# define CodeBERT model
model_name = "microsoft/graphcodebert-base"
model_config = AutoConfig.from_pretrained(model_name)
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

        # dense layer 1
        self.fc1 = nn.Linear(2*768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=-1)

    # define the forward pass
    def forward(self, sent_id, mask):

        # pass the inputs to the model
        #print(sent_id.shape)
        #print(mask.shape)
        
        #cls_hs = self.bert(sent_id, attention_mask=mask)[1]
        #print(cls_hs.shape)
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

            if start == 0:
                cls_hs = cls_hs_current.clone()
            else:
                cls_hs = torch.cat([cls_hs, cls_hs_current], dim=1)
        
        fc1_output = self.fc1(cls_hs)

        relu_output = self.relu(fc1_output)

        dropout_output = self.dropout(relu_output)

        # output layer
        fc2_output = self.fc2(dropout_output)

        # apply softmax activation
        final_output = self.softmax(fc2_output)

        return final_output

# train the model
def train():

    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)
    
        #print(preds.shape)
        #print(labels.shape)
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
            print('loss=',loss)
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds

def format_time(time=None, format=None, rebase=True):
    """See :meth:`I18n.format_time`."""
    return get_i18n().format_time(time, format, rebase)

# evaluate the model
def evaluate():

    print("\nEvaluating..")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:

            # Calculate elapsed time in minutes.
            # elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(
                step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

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
'''def give_test_data_in_chunks(x_test):
    n = len(x_test) / 50 
    preds_chunks = None
    for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
        tokens_test = tokenizer.batch_encode_plus(
            x_test_chunk.tolist(), max_length=1024, pad_to_max_length=True, truncation=True)
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()
        preds_chunks = preds_chunk if preds_chunks is None else np.append(
            preds_chunks, preds_chunk, axis=0)
        preds = np.argmax(preds_chunks, axis=1)

    return preds'''

def give_test_data_in_chunks(X_test_nparray):
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

        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()

        preds_chunks = preds_chunk if preds_chunks is None else np.append(preds_chunks, preds_chunk, axis=0)

    preds = np.argmax(preds_chunks, axis=1)
    return preds

execution_time = time.time()
print("Start time of the experiment", execution_time)
skf = StratifiedKFold(n_splits=5,shuffle=True)
TN = FP = FN = TP = 0
n_folds = 10

for i in range(n_folds): #fol
    X_test=np.load(data_name+'/data_splits/X_test_fold'+str(i)+'.npy' ,allow_pickle=True)
    y_test=np.load(data_name+'/data_splits/y_test_fold'+str(i)+'.npy' ,allow_pickle=True)
    X_test=np.asarray(X_test)
    print(X_test.shape)
    model = BERT_Arch(auto_model)
    # push the model to GPU
    model = model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    # load weights of best model
    model.load_state_dict(torch.load(model_weights_path+str(i)+'.pt'))

    ##Explainablibility, Shap
    id2label = {0: 'Not-Pair', 1: 'True-Pair'}
    labels=list(id2label.values())
    label2id = {}
    for j,label in enumerate(labels):
        label2id[label]=j
    
    with torch.no_grad():
        explainer = shap.Explainer(f, tokenizer, output_names=labels, output_shape=(X_test.shape[0], 2))
        shap_values_multiclass = explainer(X_test)

    with open("shap_model/shap_values_multiclass"+str(i)+data_name+".pkl", "wb") as fl:
        pickle.dump(shap_values_multiclass, fl)

    # Load the saved shap_values_multiclass object
    with open("shap_model/shap_values_multiclass"+str(i)+data_name+".pkl", "rb") as fl:
        shap_values_multiclass = pickle.load(fl)

    shap_values = shap_values_multiclass.mean(0)[:, label2id['True-Pair']] 
    cohorts = {"": shap_values}
    #print('MUSTU')
    shap.plots.bar(shap_values)
    plt.savefig('shap_model/shap_plot_True-Pair'+str(i)+data_name+'.png')
    plt.close()
    
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))]).T

    true_pair_df = pd.DataFrame({'Data':feature_names, 'Values': values[:,0]})
    csv_filename ='shap_features/fold_' + str(i)+ '_' + data_name + '_shap_values.csv'  # Added underscore
    true_pair_df.to_csv(csv_filename, index=False)

    shap.plots.bar(shap_values_multiclass.mean(0)[:,label2id['Not-Pair']])
    plt.savefig('shap_model/shap_plot_Not-Pair'+str(i)+data_name+'.png')
    plt.close()

    '''shap.plots.bar(shap_values_multiclass.mean(0)[:,label2id['OD']])
    plt.savefig('shap_model/shap_plot_OD'+str(i)+'.png')
    plt.close()

    shap.plots.bar(shap_values_multiclass.mean(0)[:,label2id['ID']])
    plt.savefig('shap_model/shap_plot_ID'+str(i)+'.png')
    plt.close()

    shap.plots.bar(shap_values_multiclass.mean(0)[:,label2id['NIO']])
    plt.savefig('shap_model/shap_plot_NIO'+str(i)+'.png')
    plt.close()

    shap.plots.bar(shap_values_multiclass.mean(0)[:,label2id['UD']])
    plt.savefig('shap_model/shap_plot_UD'+str(i)+'.png')
    plt.close()
    shap.plots.bar(shap_values_multiclass.mean(0)[:,label2id['OD']])
    print(shap_values_multiclass.shape)'''


    print("delete model")
    del model
    torch.cuda.empty_cache()


print("The processed is completed in : (%s) seconds. " % round((time.time() - execution_time), 5))






