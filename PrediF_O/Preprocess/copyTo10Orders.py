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
import shutil

def copyDir(source,destination):
    shutil.copytree(source, destination)

def removeDir(location):
    try:
        shutil.rmtree(location)
    except:
        #print(location+" not found.")
        x=1

def copy_file_with_new_name(source_path, destination_directory, new_name):
    dest_path = destination_directory + new_name
    shutil.copy2(source_path, dest_path)              

def copyFolders():
    destRoot="../Results/"
    removeDir(destRoot)
    for i in range(1,11):
        y=str(i)+"/"
        ranks=["Ranking - #Methods/","Ranking - All/","Ranking - Distance To Victim/","Ranking - Plus One/"]
        for rank in ranks:
            destLoc=destRoot+y+rank            
            copyDir("../"+rank,destLoc)
            types=["VP/","VC/","BSS/"]
            for type in types:
                copy_file_with_new_name("../10Of1000Orders/randomOrders_"+str(i)+".csv",destLoc+type,"randomOrders.csv")

def main():
    copyFolders()

if __name__ == "__main__":
    main()