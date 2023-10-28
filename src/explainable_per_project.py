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
import scipy as sp
import shap
import pickle

def calculate_filtered_avg(arr, th=-1000):
    filtered_arr = [x for x in arr  if x != th]
    avg = sum(filtered_arr) / len(filtered_arr)
    return avg

def compute_metrics(test_y,preds):
    
    total_tp=0
    total_fp=0
    total_fn=0
    total_tn=0
    accuracies=[]
    f_scores=[]
    precisions=[]
    recalls=[]
    print(test_y)
    print(preds)
    for category in range(6):
        y=np.where(test_y==category,1,0) #taking y only for the specific category
        print(y)
        #print(test_y)
        p=np.where(preds==category,1,0)
        print("P====:w,"+str(category))
        print(p)
        n_p=np.sum(y==1) #positive
        n_n=np.sum(y==0) #negative
        '''if (n_p == 0):
            recalls.append(-1000)
            precisions.append(-1000)
            f_scores.append(-1000)
            continue'''
        
        cm = confusion_matrix(y, p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print(cm.ravel())
        if (tp == 0.0):
            precision=0.0
            recall=0.0
            f_score=0.0
        else:
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            f_score=2*(precision*recall)/(precision+recall)
        recalls.append(recall)
        precisions.append(precision)
        f_scores.append(f_score)
        total_tp +=tp
        total_tn +=tn
        total_fp +=fp
        total_fn +=fn

    accuracy=(total_tp+total_tn)/len(test_y)
    return total_tp, total_tn, total_fp, total_fn, precisions, recalls, f_scores, accuracy


# setting the seed for reproducibility
def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 

# specify GPU
#device = torch.device("cpu")
device = torch.device("cuda")

#reading the parameters 

dataset_path = sys.argv[1]
model_weights_path = sys.argv[2]
data_name = sys.argv[3]

df = pd.read_csv(dataset_path)
'''if data_name == "Victim_Polluter_Pair-Data":
    target_data = df['isVictimPolluterPair']
    #df['Merged-Content'] = df['victim_code'] + ' ' + df['P_or_NP_code']

elif data_name == "Victim_Cleaner_Polluter_Pair-Data":
    target_data = df['isVictimPolluterCleanerPair']
    #df['Merged-Content'] = df['victim_code'] + ' ' + df['C_or_NC_code'] + ' ' +df['polluter_code']

elif data_name == "Brittle_State-Setter_Pair-Data":
    target_data = df['isBSSPair']
    df['Merged-Content'] = df['brittle_code'] + ' ' + df['SS_or_NSS_code']'''

#input_data=df['Merged-Content']

#input_data = df['full_code'] # use the 'full_code' column to run Flakify using the full code instead of pre-processed code
#target_data = df['category']
df.head()

# get project names

project_name=df['project'].unique()

# balance dataset
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


# defining CodeBERT model
model_name = "microsoft/graphcodebert-base"
model_config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
auto_model = AutoModel.from_pretrained(model_name, config=model_config)




# converting code into tokens and then vector representation
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





# converting vector representation to tensors
def text_to_tensors(tokens_train, tokens_val, tokens_test):
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    return train_seq, train_mask, val_seq, val_mask, test_seq, test_mask





# setting seed for data_loaders for output reproducibility
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



# setting up the neural network for CodeBERT fine-tuning
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

        # pass the inputs to the model
        #cls_hs = self.bert(sent_id, attention_mask=mask)[1]

        fc1_output = self.fc1(cls_hs)

        relu_output = self.relu(fc1_output)

        dropout_output = self.dropout(relu_output)

        # output layer
        fc2_output = self.fc2(dropout_output)

        # apply softmax activation
        final_output = self.softmax(fc2_output)

        return final_output



# training the model
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



# evaluating the model
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

#performing per project analysis

result = pd.DataFrame(columns = ['project_name','Accuracy','F1', 'Precision', 'Recall', 'TN', 'FP', 'FN', 'TP','avg_precision','avg_recall','avg_f1', 'total_test'])
execution_time_full = time.time()
print("Start time of  complete experiment", execution_time_full)
TN = FP = FN = TP = 0
#x='full_code'
#y='category'

if data_name == "Victim_Polluter_Pair_Per_project-Data":
    y = 'isVictimPolluterPair'
    x1='victim_code'
    x2='p_or_np_code'
    y1='victim'
    y2='p_or_np'

elif data_name == "Victim_Cleaner_Polluter_Pair_Per_project-Data":
    y = 'isVictimPolluterCleanerPair'
    x1='victim_code'
    x2='c_or_nc_code'
    x3='polluter_code'
    y1='victim'
    y2='polluter'
    y3='c_or_nc'

elif data_name == "Brittle_State-Setter_Pair_Per_project-Data":
    y = 'isBSSPair'
    x1='brittle_code'
    x2='ss_or_nss_code'
    y1='brittle'
    y2='ss_or_nss'

project_index=0
total_weighted_avg_scores=[0, 0, 0]
total_support=0
weighted_avg_arrays_list=[]
category_dict={}
for i in sorted(project_name):
#for i in project_name:
    project_index +=1
    print('*************** '+str(project_index)+' testing on project: ', i)
    project_Name=i
    
    train_dataset=  df.loc[(df['project'] != i)]
    test_dataset= df.loc[(df['project']== i)]

    model = BERT_Arch(auto_model)

    # push the model to GPU
    model = model.to(device)
    # load weights of best model
    ii=i.replace("/","-")
    model.load_state_dict(torch.load(model_weights_path+"_"+str(ii)+".pt", map_location=device))

    full_file_name='../dataset/All-Pairs-Per-Project/predicting-flakies/Unbalanced/PerProj_Unbalanced_no_Comments/VP_Per_Victim/VP_'+i.replace("/","_")+'.csv'
    df_full_proj_tests = pd.read_csv(full_file_name)
    test_dataset_per_project = df_full_proj_tests.loc[(df_full_proj_tests['project']== i)]
    test_dataset_per_module = test_dataset_per_project.groupby('module')
    print(test_dataset_per_project['module'])
    # Iterate over groups and access group data
    for module_name, test_dataset in test_dataset_per_module:
        #print(test_dataset)
        predicted_victim_or_brittle_count = 0
        predicted_victim_and_polluter_count = 0

        if data_name == "Victim_Cleaner_Polluter_Pair_Per_project-Data":
            test_dataset['vic_polluter'] = test_dataset[y1] + '<AND>' +test_dataset[y2] 
            uniq_victim_and_polluter_count = test_dataset['vic_polluter'].unique()
            test_dataset.to_csv('vic_polluter.csv', index=False)
            #uniq_victim_and_polluter_count.to_csv('vic_polluter_uniq', index=False)
            for vic_polluter in test_dataset['vic_polluter'].unique(): 
                selected_test_dataset = test_dataset.loc[(test_dataset['vic_polluter'] == vic_polluter)]
                selected_test_dataset['merged_code'] = selected_test_dataset[x1] + '<SEP>' +selected_test_dataset[x2] + '<SEP>' +test_dataset[x3] + '<VIC>' +selected_test_dataset[y1]+ ',' +selected_test_dataset[y2] +',' +test_dataset[y3]
                #found_at_least_one_cleaner=do_inference(selected_test_dataset, ii, vic_polluter, module_name, "0")

        else: #For Victim-polluter or Brittle-statesetter
            uniq_victim_or_brittle_count = len(test_dataset[y1].unique())
            for vic_or_brittle in test_dataset[y1].unique(): 
                predicted_victim_or_brittle_count = 0
                predicted_victim_and_polluter_count = 0
                for failing_order in test_dataset['FAIL_Order'].unique():
                    selected_test_dataset = test_dataset.loc[(test_dataset[y1] == vic_or_brittle) & (test_dataset['FAIL_Order'] == failing_order)]
                    selected_test_dataset['merged_code'] = selected_test_dataset[x1] + '<SEP>' +selected_test_dataset[x2] + '<VIC>' +selected_test_dataset[y1]+ ',' +selected_test_dataset[y2]
                    test_x=selected_test_dataset['merged_code']
                    ##Explainablibility, Shap
                    id2label = {0: 'Not-Pair', 1: 'True-Pair'}
                    labels=list(id2label.values())
                    label2id = {}
                    for j,label in enumerate(labels):
                        label2id[label]=j

                    with torch.no_grad():
                        explainer = shap.Explainer(f, tokenizer, output_names=labels, output_shape=(test_x.shape[0], 2))
                        shap_values_multiclass = explainer(test_x)

                    with open("shap_model_per_project/shap_values_multiclass_"+ii+"_"+vic_or_brittle+"_"+str(failing_order)+"_.pkl", "wb") as fl:
                        pickle.dump(shap_values_multiclass, fl)

                    # Load the saved shap_values_multiclass object
                    with open("shap_model_per_project/shap_values_multiclass_"+ii+"_"+vic_or_brittle+"_"+str(failing_order)+"_.pkl", "rb") as fl:
                        shap_values_multiclass = pickle.load(fl)

                    shap_values = shap_values_multiclass.mean(0)[:, label2id['True-Pair']] 
                    cohorts = {"": shap_values}
                    #print('MUSTU')
                    shap.plots.bar(shap_values)
                    plt.savefig("shap_model_per_project/shap_plot_True-Pair_"+ii+"_"+vic_or_brittle+"_"+str(failing_order)+"_.png")
                    plt.close()
    
                    cohort_labels = list(cohorts.keys())
                    cohort_exps = list(cohorts.values())
                    features = cohort_exps[0].data
                    feature_names = cohort_exps[0].feature_names
                    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))]).T

                    true_pair_df = pd.DataFrame({'Data':feature_names, 'Values': values[:,0]})
                    csv_filename ="shap_features_per_project/"+ii+"_"+vic_or_brittle+"_"+str(failing_order)+"_shap_values.csv"  # Added underscore
                    true_pair_df.to_csv(csv_filename, index=False)

                    shap.plots.bar(shap_values_multiclass.mean(0)[:,label2id['Not-Pair']])
                    plt.savefig("shap_model_per_project/shap_plot_Not-Pair_"+ii+"_"+vic_or_brittle+"_"+str(failing_order)+".png")
                    plt.close()

                    #found_at_least_one_polluter_or_state_setter=do_inference(selected_test_dataset, ii, vic_or_brittle, module_name, str(failing_order))


    print("delete model")
    del model
    torch.cuda.empty_cache()

  # define the optimizer
    '''optimizer = AdamW(model.parameters(), lr=1e-5)

    gc.collect()
    torch.cuda.empty_cache()
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []
    print('Categories=',np.unique(y_test['category'].values))'''
    
    #model.cpu()
    #model.eval()
    #print("The training process for each project is completed in : (%s) seconds. " % round((time.time() - execution_time_full), 5))
   

    '''with torch.no_grad():
        preds = give_test_data_in_chunks(test_x)

    print("test_y=",test_y.numpy())
    print("predicted=",preds)
    cr=classification_report(test_y, preds)
    print(cr)

    lines = cr.strip().split('\n')

    # parse the class names and metrics
    classes = []
    metrics = []
    for line in lines[2:-4]:  # skip the first 2 and last 3 lines
        t = line.strip().split()
        classes.append(t[0])
        key=t[0]
        values=[float(x) for x in t[1:]]

        with open("../Per_project_result/per_Category_Evaluation_"+y+".txt", "a") as file:
            file.write(i+":"+key+":" + str(values))
            file.write("\n")

        metrics.append(values)
        if key in category_dict:
            existing_values=category_dict[key]
            updated_values=[existing_values[k] + (values[k]*values[-1]) for k in range(len(values)-1)]
            updated_values.append(existing_values[-1] + values[-1]) #This is for adding support
            category_dict[key] = updated_values
        else:
            initial_val = [(values[i]*values[-1]) for i in range(len(values)-1)]
            initial_val.append(values[-1])
            category_dict[key] = initial_val

    
    print('metrics=',metrics)
    third_last_line = lines[-3].strip().split()

    accuracy = [float(x) for x in third_last_line[1:]]

    second_last_line = lines[-2].strip().split()
    macro_avg = [float(x) for x in second_last_line[2:]]

    # parse the overall scores
    last_line = lines[-1].strip().split()
    weighted_avg = [float(x) for x in last_line[2:]]
    # print the results
    print('Classes:', classes)
    #print('Metrics:', metrics)
    #print('Overall scores:', weighted_avg)

    total_weighted_avg_scores =  [ total_weighted_avg_scores[idx] + (weighted_avg[idx] * weighted_avg[-1]) for idx in range(3)] 
    #weighted_avg_arrays_list.append(weighted_avg)
    total_support +=weighted_avg[-1]

    with open("../Per_project_result/weighted_avg_per_project_test_"+y+".txt", "a") as file:
        file.write(i+",")
        file.write(str(weighted_avg))
        file.write("\n")

    del model
    torch.cuda.empty_cache()
    
avg_score=[round(i / (total_support), 4) for i in total_weighted_avg_scores] #
total_weighted_avg_scores.append((total_support))
avg_score.append(total_support)

with open("../Per_project_result/weighted_avg_per_project_test_"+y+".txt", "a") as file:
    file.write("Total_Weighted_score_for_all_project,")    
    file.write(str(total_weighted_avg_scores))
    file.write("\nWeighted_avg_score_for_all_project,")
    file.write(str(avg_score))
    file.write("\n")


with open("../Per_project_result/per_Category_Evaluation_"+y+".txt", "a") as file:
    file.write("WEIGHTED AVG==>")
    for key, value in category_dict.items():
        avg_score_per_category = [val/value[-1] for val in value[0:-1]]
        avg_score_per_category.append(value[-1])
        #file.write(key+",")
        file.write(key+":" + str(avg_score_per_category))
        file.write("\n")'''

