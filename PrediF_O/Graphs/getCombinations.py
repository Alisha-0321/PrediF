import os
import glob
from lxml import etree as ET
import csv
import json
import random 
import itertools
import copy
from urllib.parse import urlparse
import re
import math
import sys
import matplotlib.pyplot as plt
import numpy as np

"""
data={
    "gitURL":{
        "sha":"dasdasd",
        "module1":{
            "methods":["m1","m2"]-set(),
            "victims":["m1","m2"]-set()
            "brittles":[]-set()
            "polluters":{
                "victim1":["polluter1","polluter2"]-set(),
                "victim2":["polluter1","polluter2"]-set()
            },
            "cleaners":{
                "victim1":{
                    "polluter":["cleaner1","cleaner2"]-set()
                }
            },
            "statesetters":{
                "brittle":["statesetter1","statesetter2"]-set()
            }
            "codes":{
                "m1":"code"
            },
            "polluterCount":34,
            "cleanerCount":543,
            "statesetterCount":132
        }
    }
}
"""

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path) 

def readCSV(input_file, headers=False):
    with open(input_file, 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        csv_data = [row for row in reader]
        if headers:
            csv_data.pop(0)
    return csv_data

def getProjName(url):
    parts = url.split('/')
    project_name = parts[-1].split('.')[0]
    return project_name

def createCSV(csv_file,data):
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def writeFile(fileName,content):
    with open(fileName, 'w', encoding='utf-8') as f:
        f.write(content)

def appendFile(fileName,content):
    with open(fileName, 'a', encoding='utf-8') as f:
        f.write(content+"\n")

def readFile(fileName):
    with open(fileName, 'r') as file:
        contents = file.read()
    return contents

def printJSON(data):
    data=json.dumps(data,indent=4)
    print(data)
    return data

def createData(csv_data,saveCodes=False):
    data={}
    i=1
    for row in csv_data:
        #print("Processing row: "+str(i))
        git=row[0]
        if not git.endswith(".git"):
            git=git+".git"
        git=extract_github_info(git)
        if git not in data.keys():
            data[git]={}
        sha=row[1]
        data[git]["sha"]=sha
        module=row[2]
        if module.startswith("."):
            module=module[1:]
        if module.startswith("/"):
            module=module[1:]
        if module == "":
            module="NA"
        if module not in data[git].keys():
            data[git][module]={}
            data[git][module]["methods"]=set()
            data[git][module]["polluters"]={}
            data[git][module]["cleaners"]={}
            if saveCodes:
                data[git][module]["codes"]={}
            data[git][module]["victims"]=set()
            data[git][module]["brittles"]=set()
            data[git][module]["statesetters"]={}
        if row[6]=="victim":        
            victim=row[3]
            polluter=row[4]
            cleaner=row[5]
            victimCode=row[7]
            polluterCode=row[8]
            cleanerCode=row[9]
            if victim != "":
                data[git][module]["methods"].add(victim)
                if saveCodes:
                    if victim not in data[git][module]["codes"].keys():
                        data[git][module]["codes"][victim]=victimCode
                data[git][module]["victims"].add(victim)
            if polluter != "":
                data[git][module]["methods"].add(polluter)
                if saveCodes:
                    if polluter not in data[git][module]["codes"].keys():
                        data[git][module]["codes"][polluter]=polluterCode
                if victim not in data[git][module]["polluters"].keys():
                    data[git][module]["polluters"][victim]=set()
                data[git][module]["polluters"][victim].add(polluter)
            if cleaner != "":
                data[git][module]["methods"].add(cleaner)
                if saveCodes:
                    if cleaner not in data[git][module]["codes"].keys():
                        data[git][module]["codes"][cleaner]=cleanerCode
                if victim not in data[git][module]["cleaners"].keys():
                    data[git][module]["cleaners"][victim]={}
                if polluter not in data[git][module]["cleaners"][victim].keys():
                    data[git][module]["cleaners"][victim][polluter]=set()
                data[git][module]["cleaners"][victim][polluter].add(cleaner)
        elif row[6]=="brittle":
            brittle=row[3]
            statesetter=row[4]
            brittleCode=row[7]
            statesetterCode=row[8]
            if brittle!="":
                data[git][module]["methods"].add(brittle)
                data[git][module]["brittles"].add(brittle)
                if saveCodes:
                    if brittle not in data[git][module]["codes"].keys():
                        data[git][module]["codes"][brittle]=brittleCode
            if statesetter!="":
                data[git][module]["methods"].add(statesetter)
                if saveCodes:
                    if statesetter not in data[git][module]["codes"].keys():
                        data[git][module]["codes"][statesetter]=statesetterCode
                if brittle not in data[git][module]["statesetters"].keys():
                    data[git][module]["statesetters"][brittle]=set()
                data[git][module]["statesetters"][brittle].add(statesetter)
        else:
            appendFile("log.txt","Failed to process row: "+str(i))
        #print("Completed processing row: "+str(i))
        i=i+1
    return data

def generateVPCombinationsCsv(csv_data, fileName):
    mkdir("output/VP")
    csv_data = [["project","sha","module","victim","p_or_np","isVictimPolluterPair","victim_code","p_or_np_code"]] + csv_data
    fileName = os.path.join("output/VP",fileName)
    createCSV(fileName,csv_data)

def generateCombinations(l,n):
    data = list(itertools.permutations(l, n))
    return data

def preProcessCode(code):
    code=removeTestFromCode(code)
    code=removeJavaComments(code)
    return code

def populateCombinations_seperate(data,saveCodes,balanced=False):
    newDataP=[]
    newDataC=[]
    newDataSS=[]
    for git in data.keys():
        sha=data[git]["sha"]
        for module in data[git].keys():
            if module != "sha":
                methods=list(data[git][module]["methods"])
                victims=list(data[git][module]["victims"])
                for victim in victims:
                    for method in methods:
                        if victim!=method:
                            try:
                                if method in data[git][module]["polluters"][victim]:
                                    isVictimPolluterPair = 1
                                else:
                                    isVictimPolluterPair = 0
                            except:
                                isVictimPolluterPair = 0
                            if saveCodes:
                                vCode = data[git][module]["codes"][victim]
                                mCode = data[git][module]["codes"][method]
                            else:
                                vCode = ""
                                mCode = ""
                            if module == "NA":
                                newMod=""
                            else:
                                newMod=module
                            #preProcess Code
                            vCode=preProcessCode(vCode)
                            mCode=preProcessCode(mCode)
                            newDataP.append([git,sha,newMod,victim,method,isVictimPolluterPair,vCode,mCode])
    for git in data.keys():
        sha=data[git]["sha"]
        for module in data[git].keys():
            if module!="sha":
                victims=list(data[git][module]["victims"])
                methods=list(data[git][module]["methods"])
                for victim in victims:
                    polluters=data[git][module]["polluters"][victim]
                    for polluter in polluters:
                        for method in methods:
                            if victim!=method and polluter!=method:
                                try:
                                    if method in data[git][module]["cleaners"][victim][polluter]:
                                        isVPCTripplet=1
                                    else:
                                        isVPCTripplet=0
                                except:
                                    isVPCTripplet=0
                                if saveCodes:
                                    vCode = data[git][module]["codes"][victim]
                                    pCode = data[git][module]["codes"][polluter]
                                    mCode = data[git][module]["codes"][method]
                                else:
                                    vCode = ""
                                    pCode = ""
                                    mCode = ""
                                if module == "NA":
                                    newMod=""
                                else:
                                    newMod=module
                                vCode=preProcessCode(vCode)
                                pCode=preProcessCode(pCode)
                                mCode=preProcessCode(mCode)
                                newDataC.append([git,sha,newMod,victim,polluter,method,isVPCTripplet,vCode,pCode,mCode])
    for git in data.keys():
        sha=data[git]["sha"]
        for module in data[git].keys():
            if module!="sha":
                brittles=data[git][module]["brittles"]
                methods=data[git][module]["methods"]
                for brittle in brittles:
                    for method in methods:
                        if brittle!=method:
                            try:
                                if method in data[git][module]["statesetters"][brittle]:
                                    isBSSPair=1
                                else:
                                    isBSSPair=0
                            except:
                                isBSSPair=0
                            if saveCodes:
                                bCode=data[git][module]["codes"][brittle]
                                mCode=data[git][module]["codes"][method]
                            else:
                                bCode=""
                                mCode=""
                            if module=="NA":
                                newMod=""
                            else:
                                newMod=module
                            bCode=preProcessCode(bCode)
                            mCode=preProcessCode(mCode)
                            newDataSS.append([git,sha,module,brittle,method,isBSSPair,bCode,mCode])
    return newDataP,newDataC,newDataSS

def generateVCCombinationsCsv(csv_data, fileName):
    mkdir("output/VC")
    csv_data = [["project","sha","module","victim","polluter","c_or_nc","isVictimPolluterCleanerPair","victim_code","polluter_code","c_or_nc_code"]] + csv_data
    fileName = os.path.join("output/VC",fileName)
    createCSV(fileName,csv_data)

def populateCombinations(data, saveCodes):
    newData=[]
    for git in data.keys():
        sha=data[git]["sha"]
        for module in data[git].keys():
            if module != "sha":
                methods=list(data[git][module]["methods"])
                combinations = generateCombinations(methods,3)
                for combi in combinations:
                    print("Processing: "+git+", Module: "+module+", Methods: "+str(combi))
                    possibleVictim = combi[0]
                    possiblePolluter = combi[1]
                    possibleCleaner = combi[2]
                    try:
                        if possiblePolluter in data[git][module]["polluters"][possibleVictim]:
                            isVictimPolluterPair = 1
                        else:
                            isVictimPolluterPair = 0
                    except:
                        isVictimPolluterPair = 0
                    try:
                        if possibleCleaner in data[git][module]["cleaners"][possibleVictim]:
                            isVictimCleanerPair = 1
                        else:
                            isVictimCleanerPair = 0
                    except:
                        isVictimCleanerPair = 0
                    if module == "NA":
                        newMod=""
                    else:
                        newMod=module
                    if saveCodes:
                        vCode = data[git][module]["codes"][possibleVictim]
                        pCode = data[git][module]["codes"][possiblePolluter]
                        cCode = data[git][module]["codes"][possibleCleaner]
                    else:
                        vCode = ""
                        pCode = ""
                        cCode = ""
                    newData.append([git,sha,newMod,possibleVictim,possiblePolluter,possibleCleaner,isVictimPolluterPair,isVictimCleanerPair,vCode,pCode,cCode])
    return newData    

def populateCounts(csv_data):
    for git in csv_data.keys():
        for module in csv_data[git].keys():
            p=set()
            c=set()
            if module != "sha":
                for victim in csv_data[git][module]["polluters"].keys():
                    p=p.union(csv_data[git][module]["polluters"][victim])
                for victim in csv_data[git][module]["cleaners"].keys():
                    for polluter in csv_data[git][module]["cleaners"][victim].keys():
                        c=c.union(csv_data[git][module]["cleaners"][victim][polluter])
                csv_data[git][module]["polluterCount"]=len(p)
                csv_data[git][module]["cleanerCount"]=len(c)
    return csv_data

def countPosNeg(data,isCleaner=False):
    pos=0
    neg=0
    for row in data:
        if not isCleaner:
            if row[5]==1 or row[5]=="1":
                pos=pos+1
            else:
                neg=neg+1
        else:
            if row[6]==1 or row[6]=="1":
                pos=pos+1
            else:
                neg=neg+1
    return pos,neg

def validateBalanceBrittle(BSS,bBSS):
    oldPosBSS,oldNegBSS=countPosNeg(BSS)
    newPosBSS,newNegBSS=countPosNeg(bBSS)
    print("Unbalanced BSS -> Positive : "+str(oldPosBSS)+" Negative : "+str(oldNegBSS))
    print("Balanced BSS -> Positive : "+str(newPosBSS)+" Negative : "+str(newNegBSS))
    if oldPosBSS == newPosBSS and newNegBSS<=oldPosBSS:
        print("Unbalanced BSS and Balanced BSS are valid")
    else:
        print("Unbalanced BSS and Balanced BSS are not valid")

def validateBalance(VC,VP,bVC,bVP):
    oldPosVC,oldNegVC=countPosNeg(VC,True)
    newPosVC,newNegVC=countPosNeg(bVC,True)
    oldPosVP,oldNegVP=countPosNeg(VP)
    newPosVP,newNegVP=countPosNeg(bVP)
    print("Unbalanced VPC -> Positive : "+str(oldPosVC)+" Negative : "+str(oldNegVC))
    print("Balanced VPC -> Positive : "+str(newPosVC)+" Negative : "+str(newNegVC))
    print("Unbalanced VP -> Positive : "+str(oldPosVP)+" Negative : "+str(oldNegVP))
    print("Balanced VP -> Positive : "+str(newPosVP)+" Negative : "+str(newNegVP))
    if oldPosVC == newPosVC and newNegVC<=oldPosVC:
        print("Unbalanced VPC and Balanced VPC are valid")
    else:
        print("Unbalanced VPC and Balanced VPC are not valid")
    if oldPosVP == newPosVP and newNegVP<=oldPosVP:
        print("Unbalanced VP and Balanced VP are valid")
    else:
        print("Unbalanced VP and Balanced VP are not valid")

def separateCSV(data,isCleaner):
    """
    separate = {
        "git":{
            "module":{
                positiveList:[]
                negativeList:[]
            }
        }
    }
    """
    separate={}
    for row in data:
        git=row[0]
        module=row[2]
        if module=="":
            module="NA"
        if isCleaner:
            isPair=row[6]
        else:
            isPair=row[5]
        if git not in separate.keys():
            separate[git]={}
        if module not in separate[git].keys():
            separate[git][module]={
                "positiveList":[],
                "negativeList":[]
            }
        if isPair == "1" or isPair == 1:
            separate[git][module]["positiveList"].append(row)
        else:
            separate[git][module]["negativeList"].append(row)
    return separate

def createBalance(data,methods_data,vpc_data,isPolluter,isCleaner,isBrittle):
    separate=separateCSV(data,isCleaner)
    newData=[]
    for git in separate.keys():
        for module in separate[git].keys():
            newData=newData+separate[git][module]["positiveList"]
            pLen = len(separate[git][module]["positiveList"])
            nLen = len(separate[git][module]["negativeList"])
            if nLen==pLen:
                newData=newData+separate[git][module]["negativeList"]
            elif nLen<pLen:
                newData=newData+separate[git][module]["negativeList"]
                #diff = pLen-nLen
                #additionalData = getAdditionalData4VPCB(git,module,methods_data,vpc_data,diff,isPolluter,isCleaner,isBrittle)
                #newData=newData+additionalData
            else:
                newData=newData+separate[git][module]["negativeList"][:pLen]
    return newData

def getAdditionalData4VPCB(git,module,methods_data,vpc_data,diff,isPolluter,isCleaner,isBrittle):
    print("Add -> Module: "+module+" Git:"+git)
    if module=="NA":
        newMod=""
    else:
        newMod=module            
    additionalData=[]
    noNeedMethods=vpc_data[git][module]["methods"]
    victims=vpc_data[git][module]["victims"]
    brittles=vpc_data[git][module]["brittles"]
    try:
        for file in methods_data[git][module].keys():
            for method in methods_data[git][module][file].keys():
                if method not in noNeedMethods:
                    if isPolluter:
                        for victim in victims:
                            additionalData.append([git,vpc_data[git]["sha"],newMod,victim,method,0,vpc_data[git][module]["codes"][victim],methods_data[git][module][file][method]])
                            if len(additionalData)==diff:
                                break
                    elif isBrittle:
                        for brittle in brittles:
                            additionalData.append([git,vpc_data[git]["sha"],newMod,brittle,method,0,vpc_data[git][module]["codes"][brittle],methods_data[git][module][file][method]])
                            if len(additionalData)==diff:
                                break
                    elif isCleaner:
                        for victim in victims:
                            for polluter in vpc_data[git][module]["polluters"][victim]:
                                additionalData.append([git,vpc_data[git]["sha"],newMod,victim,polluter,method,0,vpc_data[git][module]["codes"][victim],vpc_data[git][module]["codes"][polluter],methods_data[git][module][file][method]])
                                if len(additionalData)==diff:
                                    break
                            if len(additionalData)==diff:
                                break
                if len(additionalData)==diff:
                    break
            if len(additionalData)==diff:
                break
    except:
        print("Failed to add extra negatives for :"+",".join(["Git: "+git,"Module: "+module,"diff: "+str(diff),"isPolluter: "+str(isPolluter),"isCleaner: "+str(isCleaner),"isBrittle: "+str(isBrittle)]))
        appendFile("Log.txt","Failed to add extra negatives for :"+",".join(["Git: "+git,"Module: "+module,"diff: "+str(diff),"isPolluter: "+str(isPolluter),"isCleaner: "+str(isCleaner),"isBrittle: "+str(isBrittle)]))
        additionalData=[]
    return additionalData

def processData(csv_data):
    """
    data={
        "git":{
            "sha":"",
            "module":{
                "filePath":{
                    "method":"code"
                }
            }
        }
    }
    """
    data={}
    for row in csv_data:
        git=row[1]
        sha=row[2]
        module=row[3]
        filePath=row[4]
        projectName=getProjName(git)
        if module !="":
            filePath=filePath.replace(projectName+"/"+module+"/src/test/java/","")
        else:
            filePath=filePath.replace(projectName+"/src/test/java/","")
        filePath=filePath.replace(".java","")
        filePath=filePath.replace("/",".")
        method=row[6]
        methodCode=row[7]
        if git not in data.keys():
            data[git]={}
        data[git]["sha"]=sha
        if module =="":
            module="NA"
        if module not in data[git].keys():
            data[git][module]={}
        if filePath not in data[git][module].keys():
            data[git][module][filePath]={}
        data[git][module][filePath][method]=methodCode
        #data[git][module][filePath][method]="code"
    return data

def beautifyCSV(csv_data):
    newData=[]
    for row in csv_data:
        if not row[0].endswith(".git"):
            print("changed : "+str(row))
            row[0]=row[0]+".git"
        row[0]=extract_github_info(row[0])
        if row[2].startswith("."):
            row[2]=row[2][1:]
        if row[2].startswith("/"):
            row[2]=row[2][1:]
        if row[2]=="NA":
            row[2]=""
        if len(row)==10:
            if not row[7].startswith("\""):
                row[7]="\""+row[7]
            if not row[7].endswith("\""):
                row[7]=row[7]+"\""
            if not row[8].startswith("\""):
                row[8]="\""+row[8]
            if not row[8].endswith("\""):
                row[8]=row[8]+"\""
            if not row[9].startswith("\""):
                row[9]="\""+row[9]
            if not row[9].endswith("\""):
                row[9]=row[9]+"\""
            row[7]=preProcessCode(row[7])
            row[8]=preProcessCode(row[8])
            row[9]=preProcessCode(row[9])
        elif len(row)==8:
            if not row[6].startswith("\""):
                row[6]="\""+row[6]
            if not row[6].endswith("\""):
                row[6]=row[6]+"\""
            if not row[7].startswith("\""):
                row[7]="\""+row[7]
            if not row[7].endswith("\""):
                row[7]=row[7]+"\""
            row[6]=preProcessCode(row[6])
            row[7]=preProcessCode(row[7])
        newData.append(row)
    return newData

def generateBSSCombinationsCsv(csv_data, fileName):
    mkdir("output/BSS")
    csv_data = [["project","sha","module","brittle","ss_or_nss","isBSSPair","brittle_code","ss_or_nss_code"]] + csv_data
    fileName = os.path.join("output/BSS",fileName)
    createCSV(fileName,csv_data)

def addAllMethods(data, methods_data):
    for git in data.keys():
        for module in data[git].keys():
            if module != "sha":
                if git in methods_data.keys() and module in methods_data[git].keys():
                    for file in methods_data[git][module].keys():
                        for method in methods_data[git][module][file].keys():
                            data[git][module]["methods"].add(method)
                            data[git][module]["codes"][method]=methods_data[git][module][file][method]
    return data

def extract_github_info(github_url):
    parsed_url = urlparse(github_url)        
    path_parts = parsed_url.path.strip("/").split("/")        
    username = path_parts[0]
    repo = path_parts[1]    
    if repo.endswith('.git'):
        repo = repo[:-4]        
    return username+"/"+repo

def removeTestFromCode(code):
    code=code.replace("@Test","")
    return code

def removeJavaComments(code):
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return code

def printPerProjVPCsvs(data):
    mkdir("output/VP")    
    for git in data.keys():
        newData=[]
        print("generating VP file for "+git)
        sha=data[git]["sha"]
        for module in data[git].keys():
            if module != "sha":
               victims=data[git][module]["victims"]
               methods=data[git][module]["methods"]
               for victim in victims:
                    polluters=data[git][module]["polluters"][victim]
                    for method in methods:
                        if method != victim:
                            if method in polluters:
                                isVPPair=1
                            else:
                                isVPPair=0
                            if module == "NA":
                                newModule=""
                            else:
                                newModule=module
                            #remove @Test
                            vCode=removeTestFromCode(data[git][module]["codes"][victim])
                            mCode=removeTestFromCode(data[git][module]["codes"][method])
                            #remove Comments
                            vCode=removeJavaComments(vCode)
                            mCode=removeJavaComments(mCode)
                            newData.append([extract_github_info(git),sha,newModule,victim,method,isVPPair,vCode,mCode])
        if len(newData)>0:
            generateVPCombinationsCsv(newData,"VP_"+extract_github_info(git).replace("/","_")+".csv")

def printPerProjVCCsvs(data):
    mkdir("output/VC")    
    for git in data.keys():
        newData=[]
        print("generating VC file for "+git)
        sha=data[git]["sha"]
        for module in data[git].keys():
            if module != "sha":
               victims=data[git][module]["victims"]
               methods=data[git][module]["methods"]
               for victim in victims:
                    polluters=data[git][module]["polluters"][victim]
                    for polluter in polluters:
                        if victim in data[git][module]["cleaners"].keys() and polluter in data[git][module]["cleaners"][victim].keys():
                            cleaners=data[git][module]["cleaners"][victim][polluter]
                            for method in methods:
                                if method != victim and method != polluter and victim!= polluter:
                                    if method in cleaners:
                                        isVCPair=1
                                    else:
                                        isVCPair=0
                                    if module=="NA":
                                        newModule=""
                                    else:
                                        newModule=module
                                    #remove @Test
                                    vCode=removeTestFromCode(data[git][module]["codes"][victim])
                                    pCode=removeTestFromCode(data[git][module]["codes"][polluter])
                                    mCode=removeTestFromCode(data[git][module]["codes"][method])
                                    #remove Comments
                                    vCode=removeJavaComments(vCode)
                                    pCode=removeJavaComments(pCode)
                                    mCode=removeJavaComments(mCode)                    
                                    newData.append([extract_github_info(git),sha,newModule,victim,polluter,method,isVCPair,vCode,pCode,mCode])            
        if len(newData)>0:
            generateVCCombinationsCsv(newData,"VC_"+extract_github_info(git).replace("/","_")+".csv")

def printPerProjBSSCsvs(data):
    mkdir("output/BSS")    
    for git in data.keys():
        newData=[]
        print("generating BSS file for "+git)
        sha=data[git]["sha"]
        for module in data[git].keys():
            if module != "sha":
               brittles=data[git][module]["brittles"]
               methods=data[git][module]["methods"]
               for brittle in brittles:
                    statesetters=data[git][module]["statesetters"][brittle]
                    for method in methods:
                        if method != brittle:
                            if method in statesetters:
                                isVPPair=1
                            else:
                                isVPPair=0
                            if module == "NA":
                                newModule=""
                            else:
                                newModule=module
                            #remove @Test
                            bCode=removeTestFromCode(data[git][module]["codes"][brittle])
                            mCode=removeTestFromCode(data[git][module]["codes"][method])
                            #remove Comments
                            bCode=removeJavaComments(bCode)
                            mCode=removeJavaComments(mCode)
                            newData.append([extract_github_info(git),sha,newModule,brittle,method,isVPPair,bCode,mCode])
        if len(newData)>0:
            generateBSSCombinationsCsv(newData,"BSS_"+extract_github_info(git).replace("/","_")+".csv")

def getFileFromMethod(method):
    fileName=method[:method.rfind(".")]
    return fileName

def createFileMethodsData(data,project,module,fileName,methodName,sha,methodsCount):
    if project not in data.keys():
        data[project]={}
        data[project]["sha"]=sha
    if module not in data[project].keys():
        data[project][module]={}
    if fileName not in data[project][module].keys():
        data[project][module][fileName]=set()
    data[project][module][fileName].add(methodName)
    data[project][module]["methodsCount"]=methodsCount
    return data

def getFilesnMethodData(csv_data):
    data={}
    for row in csv_data:
        row=row.split(",")
        project=row[0]
        sha=row[1]
        module="NA" if row[2]=="" else row[2]
        methods=row[3].split("|")
        count=len(methods)
        for method in methods:
            fileName=getFileFromMethod(method)
            data=createFileMethodsData(data,project,module,fileName,method,sha,count)
    return data

def total_permutations(groups):
    num_groups = math.factorial(len(groups))
    if num_groups>1000:
        return 1000
    num_within_group = 1
    for group in groups:
        num_within_group *= math.factorial(len(group))
        if num_within_group>1000:
            return 1000
    return num_groups * num_within_group

def random_permutations(project,module, mCount, filesnMethods, num_permutations=1000):
    max_permutations = total_permutations(filesnMethods)
    num_permutations = min(num_permutations, max_permutations)
    files=filesnMethods.keys()
    unique_perms = set()
    if project=="doanduyhai/Achilles" and module=="integration-test-3_7":
        num_permutations=720
    while len(unique_perms) < num_permutations:
        print("Got "+ str(len(unique_perms))+" unique permutations for project: "+project+", module: "+module)
        order=[]
        tempFilesList = list(files).copy()
        random.shuffle(tempFilesList)
        for fileName in tempFilesList:
            methods = filesnMethods[fileName]
            tempMethodsList=list(methods).copy()
            random.shuffle(tempMethodsList)
            order=order+tempMethodsList
        unique_perms.add(tuple(order))
    return list(unique_perms)[:num_permutations]

def generateRandomOrdersCsvData(csv_object):
    newData=[]
    for project in csv_object.keys():
        sha=csv_object[project]["sha"]
        for module in csv_object[project].keys():
            if module != "sha":                
                print("Generating orders for project: "+project+", module: "+module)
                count =  csv_object[project][module]["methodsCount"]
                del csv_object[project][module]["methodsCount"]
                filesAndMethods=csv_object[project][module]
                ordersList=[]
                orders=random_permutations(project,module,count,filesAndMethods)
                for order in orders:
                    ordersList.append(":".join(order))
                ordersStr="|".join(ordersList)
                newModule="" if module=="NA" else module
                newData.append([project,sha,newModule,ordersStr])
    return newData

def generateRandomOrdersCsv(data,fileName):
    data=[["project","sha","module","orders"]]+data
    fileName = os.path.join("output",fileName)
    createCSV(fileName,data)

def readLargeCsv(csv_file,hasHeaders=False):
    csv_data=readFile(csv_file)
    csv_data = csv_data.split("\n")
    if hasHeaders:
        csv_data.pop(0)
    new_csv_data=[]
    for row in csv_data:
        new_csv_data.append(row.split(","))
    return new_csv_data

def heuristic_one(isPass,oldRank):
    if isPass:
        return oldRank-1
    else:
        return oldRank+1
    
def heuristic_methods_before_victim(oldRank,order,isPass,victim_index):
    noOfMethods=len(order[:victim_index])
    if isPass:
        return oldRank+(1/noOfMethods)
    else:
        return oldRank-(1/noOfMethods)

def count_elements_between(a, index_b, lst):
    index_a = lst.index(a)
    if index_a == index_b:
        return 0
    if index_a < index_b:
        return len(lst[index_a+1:index_b])
    else:
        return len(lst[index_b+1:index_a])
    
def heuristic_distance_to_victim(oldRank,order,isPass,victim_index,method):
    noOfMethodsBetween=count_elements_between(method,victim_index,order)
    if noOfMethodsBetween==0:
        if isPass:
            return oldRank-(1/sys.maxsize)
        else:
            return oldRank+(1/sys.maxsize)
    if isPass:
        return oldRank-(1/noOfMethodsBetween)
    else:
        return oldRank+(1/noOfMethodsBetween)
    

def updateRank(oldRank,order,isPass,victim_index,method):
    #newRank=heuristic_one(isPass,oldRank)
    newRank=heuristic_methods_before_victim(oldRank,order,isPass,victim_index)
    #newRank=heuristic_distance_to_victim(oldRank,order,isPass,victim_index,method)
    return newRank

def sort_dict(dict):
    sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_dict

def procressOrders(ordersStr):
    orders=[]
    ordersStr=ordersStr.split("|")
    for order in ordersStr:
        orders.append(order.split(":"))
    return orders

def procressOrdersPerProj(csv_data):
    data={}
    #project,sha,module,orders
    for row in csv_data:
        if row[0]!="":
            git=row[0]
            sha=row[1]
            module="NA" if row[2]=="" else row[2]
            orders=row[4]
            procressedOrders=procressOrders(orders)
            if git not in data.keys():
                data[git]={}
                data[git]["sha"]=sha
            if module not in data[git].keys():
                data[git][module]=procressedOrders
    return data

def getPassOrFail(order,victim,vpc_data):
    subOrder=order[:order.index(victim)]
    isPolluted=False
    for polluter in vpc_data["polluters"][victim]:
        if polluter in subOrder:
            isPolluted=True
            middleOrder=subOrder[subOrder.index(polluter)+1:]
            if victim in vpc_data["cleaners"].keys() and polluter in vpc_data["cleaners"][victim].keys():
                for cleaner in vpc_data["cleaners"][victim][polluter]:
                    if cleaner in middleOrder:
                        isPolluted=False
                        break
    if isPolluted:
        return False
    else:
        return True

def updateRanks(git,module,victim,polluter,order,isPass,lastRanks):
    newRanks={}
    subOrder=order[:order.index(victim)]
    if polluter in subOrder:
        middleOrder=subOrder[subOrder.index(polluter)+1:]
        for method in middleOrder:
            newRanks[method]=updateRank(lastRanks[method],order,isPass,order.index(victim),method)
    for method in lastRanks.keys():
        if method not in newRanks.keys():
            newRanks[method]=lastRanks[method]
    return newRanks

def getUpdatedRankListPerVictim(git,module,victim,polluter,order,lastRanks,vpc_bss_data):
    vpc_data=vpc_bss_data[git][module]
    isPass=getPassOrFail(order,victim,vpc_data)
    rankList=updateRanks(git,module,victim,polluter,order,isPass,lastRanks)
    if victim in rankList.keys():
        del rankList[victim]
    if polluter in rankList.keys():
        del rankList[polluter]
    return rankList

def getRankedLists(git,module,orders,vpc_bss_data):
    data={}
    victims=vpc_bss_data[git][module]["victims"]
    for victim in victims:
        data[victim]={}
        polluters=vpc_bss_data[git][module]["polluters"][victim]
        for polluter in polluters:
            data[victim][polluter]={}
    for i,order in enumerate(orders):
        print("processing git: "+git+", module: "+module+", i = "+str(i))         
        for victim in victims:
            #print("processing victim: "+victim)
            polluters=vpc_bss_data[git][module]["polluters"][victim]
            for polluter in polluters:
                #print("processing polluter: "+polluter)
                if i==0:
                    data[victim][polluter][str(i-1)]={}
                    for method in order:
                        data[victim][polluter][str(i-1)][method]=0
                    data[victim][polluter][str(i-1)]=sort_dict(data[victim][polluter][str(i-1)])
                data[victim][polluter][str(i)]=getUpdatedRankListPerVictim(git,module,victim,polluter,order,data[victim][polluter][str(i-1)],vpc_bss_data)
                data[victim][polluter][str(i)]=sort_dict(data[victim][polluter][str(i)])
    return data

def generateRanks(ordersPerProj,vpc_bss_data):
    data={}
    for git in ordersPerProj.keys():
        for module in ordersPerProj[git].keys():
            sha=ordersPerProj[git]["sha"]
            if module!="sha":
                orders=ordersPerProj[git][module]
                rankedListsPerVictim=getRankedLists(git,module,orders,vpc_bss_data)
                for victim in rankedListsPerVictim.keys():
                    if git not in data.keys():
                        data[git]={}
                        data[git]["sha"]=sha
                    if module not in data[git].keys():
                        data[git][module]={}
                    data[git][module][victim]=rankedListsPerVictim[victim]
        #writeFile("sample.txt",str(data))
    return data 

def printData(data):
    count=0
    for git in data.keys():
        for module in data[git].keys():
            if module!="sha":
                for victim in data[git][module].keys():
                    for polluter in data[git][module][victim].keys():
                        for order in data[git][module][victim][polluter].keys():
                            if order!="-1":
                                count=count+1
                                print("Count: "+str(count))
                                #print("writing file git: "+git+", module: "+module+", victim: "+victim+", polluter: "+polluter+", i: "+order)
                                location="output/"+git.replace("/","#")+"/"+module.replace("/","#")+"/"+victim.replace("/","#")+"/"+"/"+polluter.replace("/","#")+"/"
                                mkdir(location)
                                writeFile(location+order+".txt",str(data[git][module][victim][polluter][order]))              

def getFilesInDir(location):
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(location) for f in filenames]
    return files

def createXYData(data,methods):
    yP=[]
    yN=[]
    x=[i+1 for i in range(0,len(data.keys()))]
    for i in range(0,len(data.keys())):
        pCount=0
        nCount=0
        for method in methods:
            if data[str(i)][method]>0:
                pCount=pCount+1
            else:
                nCount=nCount+1
        yP.append(pCount)
        yN.append(nCount)
    return x,yP,yN

def plotGraph(x,yP,yN,output,gitLoc,victim,polluter,brittle):
    if victim!="":
        if polluter!="":
            output="output/"+output+"VC/"+gitLoc+victim.replace("/","#")+"/"+polluter.replace("/","#")+"/"
            plt.title("Rank distribution of cleaners between +ve and -ve\nProject/Module: "+gitLoc+"\nVictim: "+victim+"\nPolluter: "+polluter)
            plt.ylabel('#cleaners with +ve and -ve ranks')
            plt.plot(x, yP, label='#cleaners with +ve ranks', color='b')
            plt.plot(x, yN, label='#cleaners with -ve ranks', color='r')
        else:
            output="output/"+output+"VP/"+gitLoc+victim.replace("/","#")+"/"
            plt.title("Rank distribution of polluters between +ve and -ve\nProject/Module: "+gitLoc+"\nVictim: "+victim)
            plt.ylabel('#polluters with +ve and -ve ranks')
            plt.plot(x, yP, label='#polluters with +ve ranks', color='b')
            plt.plot(x, yN, label='#polluters with -ve ranks', color='r')
    elif brittle!="":
        output="output/"+output+"BSS/"+gitLoc+brittle.replace("/","#")+"/"
        plt.title("Rank distribution of statesetters between +ve and -ve\nProject/Module: "+gitLoc+"\nBrittle: "+brittle)
        plt.ylabel('#statesetters with +ve and -ve ranks')
        plt.plot(x, yP, label='#statesetters with +ve ranks', color='b')
        plt.plot(x, yN, label='#statesetters with -ve ranks', color='r')
    mkdir(output)
    #plt.figure(figsize=(8, 6))    
    plt.xlabel('#orders')    
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output+victim+".png", bbox_inches='tight')
    plt.clf()

def generateVPGraphs(vpsLocation,vpc_bss_data,output):
    for git in vpc_bss_data.keys():
        gitLoc=git.replace("/","#")+"/"
        for module in vpc_bss_data[git].keys():
            moduleLoc=module.replace("/","#")+"/"
            if module!="sha":
                victims=vpc_bss_data[git][module]["victims"]
                for victim in victims:
                    victimLoc=victim.replace("/","#")+"/"
                    Loc=vpsLocation+gitLoc+moduleLoc+victimLoc
                    files=getFilesInDir(Loc)
                    data={}
                    for file in files:
                        fileContent=readFile(file)
                        fileContent=fileContent.replace("'","\"")
                        fileContent=json.loads(fileContent)
                        fileContent=sort_dict(fileContent)
                        fileName=file[file.rfind("/")+1:]
                        fileName=fileName.replace(".txt","")
                        data[fileName]=fileContent
                    x,yP,yN=createXYData(data,vpc_bss_data[git][module]["polluters"][victim])
                    print("generating VP graph for "+(gitLoc+moduleLoc)+", "+victim)
                    plotGraph(x,yP,yN,output,gitLoc+moduleLoc,victim,"","")

def generateVCGraphs(vcsLocation,vpc_bss_data,output):
    for git in vpc_bss_data.keys():
        gitLoc=git.replace("/","#")+"/"
        for module in vpc_bss_data[git].keys():
            moduleLoc=module.replace("/","#")+"/"
            if module!="sha":
                victims=vpc_bss_data[git][module]["victims"]
                for victim in victims:
                    victimLoc=victim.replace("/","#")+"/"
                    polluters=vpc_bss_data[git][module]["polluters"][victim]
                    for polluter in polluters:
                        polluterLoc=polluter.replace("/","#")+"/"
                        Loc=vcsLocation+gitLoc+moduleLoc+victimLoc+polluterLoc
                        files=getFilesInDir(Loc)
                        data={}
                        for file in files:
                            fileContent=readFile(file)
                            fileContent=fileContent.replace("'","\"")
                            fileContent=json.loads(fileContent)
                            fileContent=sort_dict(fileContent)
                            fileName=file[file.rfind("/")+1:]
                            fileName=fileName.replace(".txt","")
                            data[fileName]=fileContent
                        try:
                            print("generating VC graph for "+(gitLoc+moduleLoc)+", "+victim+", "+polluter)
                            x,yP,yN=createXYData(data,vpc_bss_data[git][module]["cleaners"][victim][polluter])                            
                            plotGraph(x,yP,yN,output,gitLoc+moduleLoc,victim,polluter,"")
                        except:
                            print("No cleaner...")

def generateBSSGraphs(bssLocation,vpc_bss_data,output):
    for git in vpc_bss_data.keys():
        gitLoc=git.replace("/","#")+"/"
        for module in vpc_bss_data[git].keys():
            moduleLoc=module.replace("/","#")+"/"
            if module!="sha":
                brittles=vpc_bss_data[git][module]["brittles"]
                for brittle in brittles:
                    brittleLoc=brittle.replace("/","#")+"/"
                    Loc=bssLocation+gitLoc+moduleLoc+brittleLoc
                    files=getFilesInDir(Loc)
                    data={}
                    for file in files:
                        fileContent=readFile(file)
                        fileContent=fileContent.replace("'","\"")
                        fileContent=json.loads(fileContent)
                        fileContent=sort_dict(fileContent)
                        fileName=file[file.rfind("/")+1:]
                        fileName=fileName.replace(".txt","")
                        data[fileName]=fileContent
                    x,yP,yN=createXYData(data,vpc_bss_data[git][module]["statesetters"][brittle])
                    print("generating BSS graph for "+(gitLoc+moduleLoc)+", "+brittle)
                    plotGraph(x,yP,yN,output,gitLoc+moduleLoc,"","",brittle)

def generateGraphsPlusOne(filesLocation,vpc_bss_data):
    resultsLocation="Ranking - Plus One/"
    vpsLocation="VP/"
    vcsLocation="VC/"
    bssLocation="BSS/"
    vpsLocation=filesLocation+resultsLocation+vpsLocation
    generateVPGraphs(vpsLocation,vpc_bss_data,resultsLocation)
    vcsLocation=filesLocation+resultsLocation+vcsLocation
    generateVCGraphs(vcsLocation,vpc_bss_data,resultsLocation)
    bssLocation=filesLocation+resultsLocation+bssLocation
    generateBSSGraphs(bssLocation,vpc_bss_data,resultsLocation)

def generateGraphsDistanceToVictim(filesLocation,vpc_bss_data):
    resultsLocation="Ranking - Distance To Victim/"
    vpsLocation="VP/"
    vcsLocation="VC/"
    bssLocation="BSS/"
    vpsLocation=filesLocation+resultsLocation+vpsLocation
    generateVPGraphs(vpsLocation,vpc_bss_data,resultsLocation)
    vcsLocation=filesLocation+resultsLocation+vcsLocation
    generateVCGraphs(vcsLocation,vpc_bss_data,resultsLocation)
    bssLocation=filesLocation+resultsLocation+bssLocation
    generateBSSGraphs(bssLocation,vpc_bss_data,resultsLocation)

def generateGraphsDistanceToVictim(filesLocation,vpc_bss_data):
    resultsLocation="Ranking - Distance To Victim/"
    vpsLocation="VP/"
    vcsLocation="VC/"
    bssLocation="BSS/"
    vpsLocation=filesLocation+resultsLocation+vpsLocation
    generateVPGraphs(vpsLocation,vpc_bss_data,resultsLocation)
    vcsLocation=filesLocation+resultsLocation+vcsLocation
    generateVCGraphs(vcsLocation,vpc_bss_data,resultsLocation)
    bssLocation=filesLocation+resultsLocation+bssLocation
    generateBSSGraphs(bssLocation,vpc_bss_data,resultsLocation)

def generateGraphsNoOfMethods(filesLocation,vpc_bss_data):
    resultsLocation="Ranking - #Methods/"
    vpsLocation="VP/"
    vcsLocation="VC/"
    bssLocation="BSS/"
    vpsLocation=filesLocation+resultsLocation+vpsLocation
    generateVPGraphs(vpsLocation,vpc_bss_data,resultsLocation)
    vcsLocation=filesLocation+resultsLocation+vcsLocation
    generateVCGraphs(vcsLocation,vpc_bss_data,resultsLocation)
    bssLocation=filesLocation+resultsLocation+bssLocation
    generateBSSGraphs(bssLocation,vpc_bss_data,resultsLocation)

def main():
    #change this location if the results are present in different location
    vpc_bss_file="data.csv"
    print("---reading vpc data")
    vpc_bss_csv_data=readCSV(vpc_bss_file,True)
    vpc_bss_data=createData(vpc_bss_csv_data)
    filesLocation="../../../Stat Results/"
    generateGraphsPlusOne(filesLocation,vpc_bss_data)
    generateGraphsDistanceToVictim(filesLocation,vpc_bss_data)
    generateGraphsNoOfMethods(filesLocation,vpc_bss_data)

if __name__ == "__main__":
    mkdir("output")
    main()