from bs4 import BeautifulSoup
import os
import sys

def output_xml_results(xml_file):
    with open(xml_file[1]) as fp:
        y=BeautifulSoup(fp, features="xml")
        # print str(y)
        tout = []
        failedconstructor = ""
        tests = set()
        #for f in y.testsuite.findAll("testcase"):
        for f in y.findAll("testcase"):
            s = "unknown"
            if f.find('failure'):
                s = "failure"
            elif f.find('error'):
                s = "error"
            else:
                s = "pass"
            if len(xml_file) == 4 and len(xml_file[3]) != 0 and (f["name"] == f["classname"] or f["name"] == ""):
                t = xml_file[3]
                failedconstructor = str.format("{},{},{},{},{}", t, s, f["time"], xml_file[2], xml_file[1])
                break
            else:
                t = str.format("{}.{}", f["classname"], f["name"])
                if t in tests:
                    t = str.format("{}.{}=DUPLICATE", f["classname"], f["name"])
                tests.add(t)
            tout.append(str.format("{},{},{},{},{}", t, s, f["time"], xml_file[2], xml_file[1]))

        if failedconstructor != "":
            print (failedconstructor)
        else:
            for f in tout:
                print (f)
            
if __name__ == '__main__':
    output_xml_results(sys.argv)
