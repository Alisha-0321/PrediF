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
    #print(test_y)
    #print(preds)
    for category in range(2):
        y=np.where(test_y==category,1,0) #taking y only for the specific category
        #print(y)
        #print(test_y)
        p=np.where(preds==category,1,0)
        #print(p)
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
device = torch.device("cuda")
#device = torch.device("cpu")

#reading the parameters 

dataset_path = sys.argv[1]
model_weights_path = sys.argv[2]
results_file = sys.argv[3]
data_name = sys.argv[4]
df = pd.read_csv(dataset_path)
#input_data = df['full_code'] # use the 'full_code' column to run Flakify using the full code instead of pre-processed code
#target_data = df['category']
df.head()

# get project names

project_name=df['project'].unique()
print('*************',project_name)

# defining CodeBERT model
model_name = "microsoft/graphcodebert-base"
model_config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
auto_model = AutoModel.from_pretrained(model_name, config=model_config)

# converting code into tokens and then vector representation
def tokenize_data(test_text):
    print(test_text)
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=1024,
        pad_to_max_length=True,
        truncation=True)
    return tokens_test

# converting vector representation to tensors
def text_to_tensors(tokens_test):
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    return test_seq, test_mask


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
        total_seq_length = sent_id.size(1)
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
def give_test_data_in_chunks(x_test, project_name):
    n = len(x_test) / 50 
    paired_data = []
    preds_chunks = None
    for g, x_test_chunk_org in x_test.groupby(np.arange(len(x_test)) // n):
        x_test_chunk = x_test_chunk_org.str.split("<VIC>").str[0]
        tokens_test = tokenizer.batch_encode_plus(x_test_chunk.tolist(), max_length=1024, pad_to_max_length=True, truncation=True)
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()
        preds_chunks = preds_chunk if preds_chunks is None else np.append(
            preds_chunks, preds_chunk, axis=0)
        for row, x_test_data in zip(preds_chunk, x_test_chunk_org):
            paired_data.append((row, x_test_data))

    with open("../Per_project_result/Ranking_"+y+"_"+project_name+"_Class1.txt", "w") as Class1File, open("../Per_project_result/Ranking_"+y+"_"+project_name+"_Class0.txt", "w") as Class0File:
        for row, x_test_data in paired_data:
            item0, item1 = row
            #print(row)
            if item1 > item0: # as the score is negative, hence upper value indiates the actual class 
                #print(x_test_data)
                #print('************I AM GREATER THAN ITEM1,'+str(x_test_data.split("<VIC>")[1:]))
                Class1File.write(str(item1) + ",<VIC>"+ str(x_test_data.split("<VIC>")[1:]))
                Class1File.write("\n")
            else:
                #print('*************I AM LESS THAN ITEM0,'+str(x_test_data.split("<VIC>")[1:]))
                Class0File.write(str(item0) + ",<VIC>"+ str(x_test_data.split("<VIC>")[1:]))
                Class0File.write("\n")
    preds = np.argmax(preds_chunks, axis=1)
    return preds

print(torch.cuda.is_available())


def do_inference(test_dataset, ii, vic_or_brittle):
    test_x = test_dataset['merged_code']
    test_y=test_dataset[y]
    #print(test_x)
     
    tokens_test = tokenize_data(test_x)
    #print('**************************=============Tokens_test=======********************************************')
    #print(tokens_test)
    #print('**************************=============END=======********************************************')

    y_test = pd.DataFrame(test_y)
    y_test.columns = ['category']
    test_y = torch.tensor(y_test['category'].values)

    test_seq, test_mask = text_to_tensors(tokens_test)
    
    
    start = time.time()
    with torch.no_grad():
        preds = give_test_data_in_chunks(test_x, ii)
    end = time.time()
    
    required_time = end - start
    #with open("../Per_project_result/Per-Victim-Runtime.txt", "a") as file:
    #with open("../Per_project_result/Per-Brittle-Runtime.txt", "a") as file:
    with open("../Per_project_result/Per-Victim-Cleaner-Runtime.txt", "a") as file:
        file.write(vic_or_brittle + "," + str(round(required_time,2)))
        file.write('\n')

    #X_test_str_array = test_x.astype(str)
    
    pred_len=len(preds) 
    print(pred_len)

    flag_matched=False

    for idx in range(pred_len):
        if test_y[idx] == preds[idx] and test_y[idx] ==1:
            #print('****************idx*****************')
            flag_matched=True
            break


    '''text_after_vic_and_before_comma = []
    for x in X_test_str_array:
        parts = x.split("<VIC>")
        # Check if there are two parts (text after <VIC> and text after the comma)
        if len(parts) == 2:
            # Split the second part at the comma and get the first part
            text = parts[1].rsplit(",",1)[0]
            text_after_vic_and_before_comma.append(text)

    key_value_pairs = {}
    
	# Create key-value pairs with indices
    for idx, item in enumerate(text_after_vic_and_before_comma):
        print('item='+str(item))
        print('idx='+str(idx))
        if item in key_value_pairs:
            key_value_pairs[item].append(idx)
        else:
            key_value_pairs[item] = [idx]

    lists_of_duplicate_indices = list(key_value_pairs.values())
    

    count_uniq_v_or_b_or_vp = 0 # v=victim, b=brittle, vp=victim-polluter
    uniq_victim_which_has_atleast_one_correct_polluter = 0'''
    '''How many uniq victim exists in a project? Answer in the caller of this method
    
    with open("../Per_project_result/Flaky-Test_Victim_polluter.txt", "a") as file, open("../Per_project_result/SanityCheck_"+y + "_"+ ii +".txt", "w") as sanityCheckFile: #y=BSS, ii=project_name
        numpy_array_test_y=test_y.numpy()

        for indices in lists_of_duplicate_indices:
            print('************indices************')
            flag_matched = False
            true_victim_polluter_pair = False
            #print('indices=' +str(indices))
            for idx in indices:
                #print(test_x[idx])
                if numpy_array_test_y[idx]==1: #1 means true victim-polluter/brittle-state_setter/ pairs
                    if true_victim_polluter_pair == False: # Counting a victim for one time
                        print('Entered *********')
                        count_uniq_v_or_b_or_vp +=1 
                        true_victim_polluter_pair = True
                        sanityCheckFile.write(str(test_x[idx].split("<VIC>")[1:]))
                        sanityCheckFile.write("\n")
                    if numpy_array_test_y[idx] == preds[idx]:
                        print('****************idx*****************')
                        #print()
                        flag_matched=True
                        break
            if flag_matched:
                uniq_victim_which_has_atleast_one_correct_polluter += 1
        #print('project-name='+ str(i) +',Total-uniq-victim='+ str(count_uniq_v_or_b_or_vp) +", uniq_victim_which_has_atleast_one_correct_polluter="+str(uniq_victim_which_has_atleast_one_correct_polluter))
        file.write('project-name='+ str(i) + ',Total-uniq-victim='+ str(count_uniq_v_or_b_or_vp) +", uniq_victim_which_has_atleast_one_correct_polluter="+str(uniq_victim_which_has_atleast_one_correct_polluter))
        file.write("\n")'''

    return flag_matched


result = pd.DataFrame(columns = ['project_name','Accuracy','F1', 'Precision', 'Recall', 'TN', 'FP', 'FN', 'TP','avg_precision','avg_recall','avg_f1', 'total_test'])
execution_time_start = time.time()

total_weighted_avg_scores=[0, 0, 0]
total_support=0
TN = FP = FN = TP = 0

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
weighted_avg_arrays_list=[]
category_dict={}
for i in sorted(project_name):
#for i in project_name:
    #if i != "Apache/Struts":
    #    continue
    project_index +=1
    print('*************** '+str(project_index)+' testing on project: ', i)
    project_Name=i

    # load weights of best model
    ii=i.replace("/","-")

    model = BERT_Arch(auto_model)
    model = model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(torch.load(model_weights_path+"_"+str(ii)+".pt", map_location=device))
    #model.cpu()
    model.eval()


    train_dataset=  df.loc[(df['project'] != i)]
    #full_file_name='../dataset/All-Pairs-Per-Project/predicting-flakies/Unbalanced/PerProj_Unbalanced_no_Comments/VP/VP_'+i.replace("/","_")+'.csv'
    #full_file_name='../dataset/All-Pairs-Per-Project/predicting-flakies/Unbalanced/PerProj_Unbalanced_no_Comments/VC/VC_'+i.replace("/","_")+'.csv'
    #full_file_name='../dataset/All-Pairs-Per-Project/predicting-flakies/Unbalanced/PerProj_Unbalanced_no_Comments/BSS/BSS_'+i.replace("/","_")+'.csv'
    #df_full_proj_tests = pd.read_csv(full_file_name)
    #test_dataset= df_full_proj_tests.loc[(df_full_proj_tests['project']== i)]
    #test_dataset.to_csv('original.csv', index=False)
    test_dataset_per_project = df.loc[(df['project']== i)]
    test_dataset_per_module = test_dataset_per_project.groupby('module')
    # Iterate over groups and access group data
    for module_name, test_dataset in test_dataset_per_module:
        print(test_dataset)
        predicted_victim_or_brittle_count = 0
        predicted_victim_and_polluter_count = 0

        if data_name == "Victim_Cleaner_Polluter_Pair_Per_project-Data":
            test_dataset['vic_polluter'] = test_dataset[y1] + '<AND>' +test_dataset[y2] 
            uniq_victim_and_polluter_count = test_dataset['vic_polluter'].unique()
            test_dataset.to_csv('vic_polluter.csv', index=False)
            #print('ALLL*********')
            #print(len(uniq_victim_and_polluter_count))

            for vic_polluter in test_dataset['vic_polluter'].unique(): 
                with open("../Per_project_result/NameOfUniqVictimAndPolluter_"+y + "_"+ ii +"_"+ module_name.replace("/","_") +".txt", "a") as sanityCheckFileForUniqVictimAndPolluter:
                    sanityCheckFileForUniqVictimAndPolluter.write(vic_polluter)
                    sanityCheckFileForUniqVictimAndPolluter.write('\n')

                #print(vic_polluter)            
                selected_test_dataset = test_dataset.loc[(test_dataset['vic_polluter'] == vic_polluter)]
                selected_test_dataset['merged_code'] = selected_test_dataset[x1] + '<SEP>' +selected_test_dataset[x2] + '<VIC>' +selected_test_dataset[y1]+ ',' +selected_test_dataset[y2] +',' +test_dataset[y3]
                #print('selected_test_dataset')
                #print(vic)
                #print(len(selected_test_dataset))
                found_at_least_one_cleaner=do_inference(selected_test_dataset, ii, vic_polluter)
                if found_at_least_one_cleaner:
                    predicted_victim_and_polluter_count += 1

            with open("../Per_project_result/Flaky-Test_Victim_polluter_and_cleaner.txt", "a") as file:
                file.write('project-name,'+ str(i)+"_"+ module_name.replace("/","_")  + ',Total-uniq-victim,'+ str(len(uniq_victim_and_polluter_count)) +", uniq_victim_which_has_atleast_one_correct_polluter,"+str(predicted_victim_and_polluter_count))
                file.write('\n')

            test_dataset['merged_code'] = test_dataset[x1] + '<SEP>' +test_dataset[x2] + '<SEP>' +test_dataset[x3] + '<VIC>' + test_dataset[y1]+ ',' +test_dataset[y2]+',' +test_dataset[y3]
            #print(test_dataset[y1]+ ',' +test_dataset[y2]+',' +test_dataset[y3])
        else: #For Victim-polluter or Brittle-statesetter
            uniq_victim_or_brittle_count = len(test_dataset[y1].unique())
            #print(uniq_victim_or_brittle_count)
            print('*********************-=============**************************')
            for vic_or_brittle in test_dataset[y1].unique(): 
                with open("../Per_project_result/NameOfUniqVictimOrBrittle_"+y + "_"+ ii+"_"+ module_name.replace("/","_") +".txt", "a") as sanityCheckFileForUniqVictimOrBrittle:
                    sanityCheckFileForUniqVictimOrBrittle.write(vic_or_brittle)
                    sanityCheckFileForUniqVictimOrBrittle.write('\n')

                selected_test_dataset = test_dataset.loc[(test_dataset[y1] == vic_or_brittle)]
                #print(selected_test_dataset)
                selected_test_dataset['merged_code'] = selected_test_dataset[x1] + '<SEP>' +selected_test_dataset[x2] + '<VIC>' +selected_test_dataset[y1]+ ',' +selected_test_dataset[y2]
                found_at_least_one_polluter_or_state_setter=do_inference(selected_test_dataset, ii, vic_or_brittle)

                if found_at_least_one_polluter_or_state_setter:
                    predicted_victim_or_brittle_count += 1
            with open("../Per_project_result/Flaky-Test_Victim_polluter.txt", "a") as file:
                file.write('project-name,'+ str(i)+"_"+ module_name.replace("/","_")  + ',Total-uniq-victim,'+ str(uniq_victim_or_brittle_count) +", uniq_victim_which_has_atleast_one_correct_polluter,"+str(predicted_victim_or_brittle_count))
                file.write('\n')
            
            #For computing precision and recall and fscore
            test_dataset['merged_code'] = test_dataset[x1] + '<SEP>' + test_dataset[x2] + '<VIC>' + test_dataset[y1]+ ',' + test_dataset[y2]

        test_x = test_dataset['merged_code']
        test_y=test_dataset[y]
        tokens_test = tokenize_data(test_x)
        y_test = pd.DataFrame(test_y)
        y_test.columns = ['category']
        test_y = torch.tensor(y_test['category'].values)

        test_seq, test_mask = text_to_tensors(tokens_test)
        
        with torch.no_grad():
            preds = give_test_data_in_chunks(test_x, ii)

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
                file.write(i+","+ module_name.replace("/","_")  +":"+key+":" + str(values))
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
            file.write(i+","+ module_name+",")
            file.write(str(weighted_avg))
            file.write("\n")

    del model
    torch.cuda.empty_cache()

execution_time_end = time.time()

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
        file.write("\n")

time_difference = execution_time_end - execution_time_start

print("Time taken:", time_difference, "seconds")


