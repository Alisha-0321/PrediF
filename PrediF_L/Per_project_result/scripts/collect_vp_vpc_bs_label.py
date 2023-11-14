import sys
import pandas as pd

file_name = sys.argv[1];
df = pd.read_csv(file_name) 
only_filename=file_name.rsplit("/",1)[1]
category=only_filename.split("_")[0]
print(file_name)
print(category)
result_df=pd.DataFrame()

if category == "VP":
    column1 = df['project']
    column2 = df['sha']
    column3 = df['module']
    column4 = df['victim']
    column5 = df['p_or_np']
    column6 = df['isVictimPolluterPair']
    result_df = pd.DataFrame({'project':column1, 'sha':column2, 'module':column3, 'victim':column4, 'p_or_np':column5, 'isVictimPolluterPair':column6})

elif category == "VC":
    column1 = df['project']
    column2 = df['sha']
    column3 = df['module']
    column4 = df['victim']
    column5 = df['polluter']
    column6 = df['c_or_nc']
    column7 = df['isVictimPolluterCleanerPair']
    #result_df = pd.DataFrame({'victim':column1, 'polluter':column2, 'c_or_nc':column3, 'isVictimPolluterCleanerPair':column4})
    result_df = pd.DataFrame({'project':column1, 'sha':column2, 'module':column3, 'victim':column4, 'polluter':column5, 'c_or_nc':column6, 'isVictimPolluterCleanerPair':column7 }) #For Bala


elif category == "BSS":
    column1 = df['project']
    column2 = df['sha']
    column3 = df['module']     
    column4 = df['brittle']
    column5 = df['ss_or_nss']
    column6 = df['isBSSPair']
    result_df = pd.DataFrame({'project':column1, 'sha':column2, 'module':column3, 'brittle':column4, 'ss_or_nss':column5, 'isBSSPair':column6})

result_df.to_csv("X.txt", index=False)

