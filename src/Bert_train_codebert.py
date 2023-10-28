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
from transformers import LongformerModel
from transformers import LongformerForSequenceClassification, LongformerTokenizer
import scipy as sp
import shap
import pickle
import math

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

df = pd.read_csv(dataset_path)
#input_data = df['victim_code'] # use the 'full_code' column to run Flakify using the full code instead of pre-processed code
#target_data = df['category'] # For multi-class
if data_name == "Victim_Polluter_Pair-Data":
    target_data = df['isVictimPolluterPair']
    df['Merged-Content'] = df['victim_code'] + '<SEP>' + df['p_or_np_code']
    #print(len(df))
    print('DONE**')

elif data_name == "Victim_Cleaner_Polluter_Pair-Data":
    columns_to_check = ['victim_code', 'c_or_nc_code', 'polluter_code'] 
    df.dropna(subset=columns_to_check, inplace=True)
    target_data = df['isVictimPolluterCleanerPair']
    df['Merged-Content'] = df['victim_code'] + '<SEP>' + df['c_or_nc_code'] + '<SEP>' + df['polluter_code']

elif data_name == "Brittle_State-Setter_Pair-Data":
    target_data = df['isBSSPair']
    df['Merged-Content'] = df['brittle_code'] + '<SEP>' + df['ss_or_nss_code']

input_data=df['Merged-Content']
#nan_val=df[df['c_or_nc_code'].isna()]

#df1 = pd.DataFrame(data,columns=['first_set','second_set'])
#print(nan_val)

df.head()

# balance dataset,converts a 1D Pandas DataFrame into a 2D NumPy array
'''def sampling(X_train, y_train, X_valid, y_valid):
    
    oversampling = RandomOverSampler(
        sampling_strategy='minority', random_state=49)
    x_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    x_val = X_valid.values.reshape(-1, 1)
    y_val = y_valid.values.reshape(-1, 1)
   
    x_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_val = X_valid.reshape(-1, 1)
    y_val = y_valid.reshape(-1, 1)

    x_train, y_train = oversampling.fit_resample(x_train, y_train)
    x_val, y_val = oversampling.fit_resample(x_val, y_val)
    x_train = pd.Series(x_train.ravel())
    y_train = pd.Series(y_train.ravel())
    x_val = pd.Series(x_val.ravel())
    y_val = pd.Series(y_val.ravel())

    return x_train, y_train, x_val, y_val'''


# define CodeBERT model
model_name = "microsoft/codebert-base"
model_config = AutoConfig.from_pretrained(model_name, return_dict=False, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
auto_model = AutoModel.from_pretrained(model_name, config=model_config)

def chunk_text(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        print(chunk)
        chunks.append(chunk)
    return chunks

def tokenize_data(train_text, val_text, test_text):
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=512,
        pad_to_max_length=True,
        truncation=True)
    
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=512,
        pad_to_max_length=True,
        truncation=True)

    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=512,
        pad_to_max_length=True,
        truncation=True)
    return tokens_train, tokens_val, tokens_test

# convert code into tokens and then vector representation
'''def tokenize_data(train_text, val_text, test_text):
    max_chunk_size = 512
    chunked_train_text = []
    chunked_val_text = []
    chunked_test_text = []
    #print(len(train_text))
    for text in train_text:
        chunks = chunk_text(text, max_chunk_size)
        for chunk in chunks: 
            tokens = tokenizer.batch_encode_plus(
    	        [chunk],
        	    max_length=512,
            	pad_to_max_length=True,
                truncation=True
            )
            #print(len(tokens))
            #print('*************HELLO FROM TRAIN DATA**********')
            chunked_train_text.append(tokens)

    for text in val_text:
        chunks = chunk_text(text, max_chunk_size)
        for chunk in chunks: 
            tokens = tokenizer.batch_encode_plus(
                #val_text.tolist(),
                [chunk],
                max_length=512,
                pad_to_max_length=True,
                truncation=True)
            #print(len(tokens))
            #print('*************HELLO FROM TEST DATA**********')
            chunked_val_text.append(tokens)

    for text in test_text:
        chunks = chunk_text(text, max_chunk_size)
        for chunk in chunks:
            tokens_test = tokenizer.batch_encode_plus(
    		    #test_text.tolist(),
                [chunk],
        	    max_length=512,
        		pad_to_max_length=True,
		        truncation=True)
            print(len(tokens))
            chunked_test_text.append(tokens)

    return tokens_train, tokens_val, tokens_test'''

# convert vector representation to tensors
def text_to_tensors(tokens_train, tokens_val, tokens_test):
    count_tmp=0
    count_tmp +=1
    if count_tmp==1:
        print(tokens_test['attention_mask'][0])
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
    batch_size = 32
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
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer), For FlakiCat=5, IDOFT=6, binary-classification=2
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=-1)

    # define the forward pass
    def forward(self, sent_id, mask):

        # pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)[1]
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
    
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
            print('loss=',loss.item())
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
'''def give_test_data_in_chunks(x_test): #BERT
    n = len(x_test) / 50 
    preds_chunks = None
    for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
        tokens_test = tokenizer.batch_encode_plus(
            x_test_chunk.tolist(), max_length=510, pad_to_max_length=True, truncation=True)
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

        tokens_test = tokenizer.batch_encode_plus(
            x_test_chunk.tolist(), max_length=500, pad_to_max_length=True, truncation=True,
            return_tensors='pt', padding='longest')

        test_seq = tokens_test['input_ids'].squeeze()
        test_mask = tokens_test['attention_mask'].squeeze()

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

for train_index, test_index in skf.split(input_data, target_data):

    print(" NOW IN FOLD NUMBER", fold_number)
    #for q in range(len(test_index)):
        #print(test_index[q])
    '''X_train, X_test = input_data.iloc[list(train_index)], input_data.iloc[list(test_index)]
    y_train, y_test = target_data.iloc[list(train_index)], target_data.iloc[list(test_index)]
    np.save(data_name+'/data_splits/X_test_fold'+str(fold_number)+'.npy', X_test)
    np.save(data_name+'/data_splits/y_test_fold'+str(fold_number)+'.npy', y_test)
    #"test_data"

    X_train, X_valid, y_train, y_valid=train_test_split(X_train, y_train, random_state=49, test_size=0.1, stratify=y_train)
    np.save(data_name+'/data_splits/X_train_fold'+str(fold_number)+'.npy', X_train)
    np.save(data_name+'/data_splits/y_train_fold'+str(fold_number)+'.npy', y_train)
    
    np.save(data_name+'/data_splits/X_valid_fold'+str(fold_number)+'.npy', X_valid)
    np.save(data_name+'/data_splits/y_valid_fold'+str(fold_number)+'.npy', y_valid)'''

    X_test=np.load(data_name+'/data_splits/X_test_fold'+str(fold_number)+'.npy',allow_pickle=True)
    y_test=np.load(data_name+'/data_splits/y_test_fold'+str(fold_number)+'.npy',allow_pickle=True)

    X_train=np.load(data_name+'/data_splits/X_train_fold'+str(fold_number)+'.npy',allow_pickle=True)
    y_train=np.load(data_name+'/data_splits/y_train_fold'+str(fold_number)+'.npy',allow_pickle=True)

    X_valid=np.load(data_name+'/data_splits/X_valid_fold'+str(fold_number)+'.npy',allow_pickle=True)
    y_valid=np.load(data_name+'/data_splits/y_valid_fold'+str(fold_number)+'.npy',allow_pickle=True)

    #X_train, y_train, X_valid, y_valid = sampling(X_train, y_train, X_valid, y_valid)
    print(data_name)
    tokens_train, tokens_val, tokens_test = tokenize_data(X_train, X_valid, X_test)
    #tokens_train_1, tokens_val_1, tokens_test_1 = tokenize_data(X_train, X_valid, X_test)
    #tokens_train_2, tokens_val_2, tokens_test_2 = tokenize_data(X_train, X_valid, X_test)

#Concatenate

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

    # number of training epochs
    epochs = 20

    print("Class Weights:", class_weights)

    model = BERT_Arch(auto_model)

    # push the model to GPU
    model = model.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    gc.collect()
    torch.cuda.empty_cache()
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train the model
        train_loss, _ = train()

        # evaluate the model
        valid_loss, _ = evaluate()

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_weights_path+str(fold_number)+'.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
    
    # load weights of best model
    model.load_state_dict(torch.load(model_weights_path+str(fold_number)+'.pt'))
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        preds = give_test_data_in_chunks(X_test)
        total_execution_time+=(time.time() - start_time)

    cr=classification_report(test_y, preds)
    #print(type(cr))
    with open(where_data_comes+"-result/classification_report_"+str(no_splits)+"folds_"+str(epochs)+"_epoch.txt", "a") as file:
        file.write("Fold="+str(fold_number)+"\n")
        file.write(cr)
        file.write("\n")

    cm = confusion_matrix(test_y, preds)
    #print(cm)
	
    with open(where_data_comes+"-result/confusion_matrix_"+str(no_splits)+"folds_"+str(epochs)+"_epoch.txt", "a") as file:
        file.write("Fold="+str(fold_number)+"\n")
        file.write(np.array2string(cm))
        file.write("\n")
        
    # Create a heatmap plot of the confusion matrix
    #fig, ax = plt.subplots(figsize=(10, 8))
    #sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    
    # Set the plot title and labels
    #ax.set_xlabel("Predicted labels")
    #ax.set_ylabel("True labels")
    #ax.set_title("Confusion Matrix")
    
    # Set the axis tick labels
    #class_names = [0, 1, 2, 3, 5, 6] # replace with your own class names
    #class_names = [0, 1, 2, 3] # replace with your own class names
    #ax.xaxis.set_ticklabels(class_names)
    #ax.yaxis.set_ticklabels(class_names)
    
    #plt.show()
    #plt.savefig('Confusion-Matrix/confusion_matrix_val_30_epoch_fold'+str(fold_number)+'.png')
    #avg_execution_time=total_execution_time/no_splits

    tn, fp, fn, tp = confusion_matrix(test_y, preds, labels=[0, 1]).ravel()
    TN = TN + tn
    FP = FP + fp
    FN = FN + fn
    TP = TP + tp

    #fpr, tpr, thresholds = metrics.roc_curve(y_test_label, predicted_labels, pos_label=1)
    #Total_auc += metrics.auc(fpr, tpr)

    print("delete model")
    del model
    torch.cuda.empty_cache()

    fold_number = fold_number+1







