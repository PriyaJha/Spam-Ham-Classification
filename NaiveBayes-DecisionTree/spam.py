
# coding: utf-8

# In[3]:

#authors: Priyanka Jha and SivaKumar

#This code is the implementation of continous model where training and classification is done 
#based on the total number of occurenences of words in the document. Binary features implementation 
#results are compared to this and recorded in the report "assignment4_part1.pdf". In addition comments 
#are provided in this file for the ligibility of the code.

#Caveat: The program uses the argument model name to build the Naive Nayes and the decision Tree 
#classifier, so it is important that we pass the argument carefully for train and reuse it for test 
#or else error will be thrown as classifier relevant objects will not be built.

#imports
import sys
if sys.version_info.major==2:
    print("warning! use python3")
import sys
import os
import re
from collections import Counter
import string
from math import log
import pickle
import random
import operator

#arguments passed:
# mode, technique, directory is read from the arguments...
mode=sys.argv[1]
technique=sys.argv[2]
ds_dir=sys.argv[3]
dir_spam=ds_dir+'/spam/'
dir_notspam=ds_dir+'/notspam/'
model_file=sys.argv[4]+'.pkl'

test_files=[]
prior=Counter()
spam_word_counter=Counter()
notspam_word_counter=Counter()
total_spam_wordcount = 0
total_notspam_wordcount = 0

# Cleaning the document files...
def data_preprocessing(a_file,folder):
    f = open(folder + a_file, 'r',errors='ignore')
    text= f.read()
    f.close()        
    cleantext=re.findall('[A-Za-z0-9]+',str.lower(text))
    return cleantext

# Training the model by storing the parameters of Bayes net classifier
def init_lists(folder,label):
    global prior    
    file_list = os.listdir(folder)
    for a_file in file_list:
        lines = data_preprocessing(a_file,folder)
        prior[label] +=1      
        for word in lines:
            if label=='spam':                      
                spam_word_counter[word]+=1
            if label=='notspam':
                notspam_word_counter[word]+=1 
                

def build_featureVector(folder,featureList):
    file_list = os.listdir(folder)
    for a_file in file_list:
        lines = data_preprocessing(a_file,folder)
        for word in lines:
            featureList.append(word)
    
    return featureList

def build_dataTable(folder,featureList,label,dataTable):
    file_list = os.listdir(folder)
    for a_file in file_list:
        word_counter=Counter()
        dataRow=[]
        lines = data_preprocessing(a_file,folder)
        for word in lines:
            word_counter[word]+=1
        for feature in featureList:
            if feature in word_counter.keys():
                dataRow.append(word_counter[feature])
            else: dataRow.append(0)
        #The last column is the class column and holds spam or notspam value...
        dataRow.append(label)
        dataTable.append(dataRow)
    
    return dataTable
            

#divideset funtion divides a dataset on Column's value
def  divideset(rows,col,value):
    set1=[]
    set2=[]
    if isinstance(value,int) or isinstance(value,float):
        for row in rows:
            if int(row[col])>=value:
                set1.append(row)
            else:
                set2.append(row)
    return(set1,set2)   

#create individual counts of each possible results
def uniquecounts(rows):
    results={}
    for row in rows:
        r=row[len(row)-1]
        if r not in results: results[r]=0
        results[r]+=1
    return results

#Calculate Entropy
def entropy(rows):
    countRows=len(rows)
    log2=lambda x:log(x)/log(2)
    unqCount=uniquecounts(rows)
    entrpy=0
    for uc in unqCount.keys():
        #print (uc,unqCount[uc])
        p= float(unqCount[uc])/countRows
        entrpy -= p* log2(p)
    return entrpy

class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb
        
def buildtree(rows):
    if len(rows)==0: return decisionnode()
    
    current_score=entropy(rows)

    # Set up some variables to track the best criteria
    best_gain=0.0
    best_criteria=None
    best_sets=None
    
    column_count=len(rows[0])-1 
    #print('column_count',column_count)
    
    for col in range(0,column_count):
        try:
            column_mean_value=0
            row_count=0
            for row in rows:
                row_count+=1
                value=row[col]
                column_mean_value+=value # adds the current column value to the dict.
            column_mean_value=int(column_mean_value/row_count)
            value=column_mean_value
            
            (set1,set2)=divideset(rows,col,value)
            p1=float(len(set1)/len(rows))
            p2=float(len(set2)/len(rows))
            gain=current_score-p1*entropy(set1)-p2*entropy(set2)
            #print(col,gain,best_gain)
                
            if gain>best_gain and len(set1)>0 and len(set2)>0:
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set1,set2)
                
        except:
            pass
    #create the sub-branches..
    print("best_gain",best_gain)
    if best_gain>0:
        trueBranch=buildtree(best_sets[0])
        falseBranch=buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))  

def printtree(tree,indent=''):
    if tree.results!=None:
        print(str(tree.results))
    else:
        print(str(tree.col),':',str(tree.value),'?')
        # Print the branches
        print(indent+'T->', end=" ")
        printtree(tree.tb,indent+'  ')
        print(indent+'F->', end=" ")
        printtree(tree.fb,indent+'  ')

def uniqify(seq):
    keys = {}
    unQ=[]
    for e in seq:
        keys[e] = 1
    for e in keys.keys():
        unQ.append(e)  
    return unQ       

def classifydt(observation,tree):
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
        branch=None
        if isinstance(v,int) or isinstance(v,float):
            if v>=tree.value: branch=tree.tb
            else: branch=tree.fb
        else:
            if v==tree.value: branch=tree.tb
            else: branch=tree.fb
    return classifydt(observation,branch)

# Predict the new documents' class
def classifybayes(folder,label):
    #identifies the class that maximizes posterior
    global test_files
    global count_prior
    global total_spam_wordcount
    global total_notspam_wordcount
    file_list = os.listdir(folder)
    count=0
    countSpam=0
    count_prior=0
    total_spam_wordcount=float(sum(spam_word_counter.values()))
    total_notspam_wordcount=float(sum(notspam_word_counter.values()))
    
    
    for item in prior.items():
        count_prior+=item[1]
    print("count_prior",count_prior)
    for a_file in file_list:
        count+=1
        SpamPosterior=float(log(prior['spam']/count_prior))
        NotSpamPosterior=float(log(prior['notspam']/count_prior))
        lines = data_preprocessing(a_file,folder) 
        #print(lines)
        for word in lines:
            SpamPosterior=SpamPosterior+log(float(spam_word_counter[word]+1)/float(total_spam_wordcount+2))
            
            NotSpamPosterior=NotSpamPosterior+log(float(notspam_word_counter[word]+1)/float(total_notspam_wordcount+2))
         
        if SpamPosterior>NotSpamPosterior:
            countSpam+=1
            test_files.append((a_file,'spam',label))
        else:
            test_files.append((a_file,'notspam',label))
    
    print("count",count)  
    print("countSpam",countSpam)   
    print("percent spam:",countSpam*100/count)


def findTopWords():
    print("Top 10 words...")
    global total_spam_wordcount
    global total_notspam_wordcount
    spam_word_contribution = {}
    notspam_word_contribution = {}
    count_prior=0
    for item in prior.items():
        count_prior+=item[1]
    total_spam_wordcount=float(sum(spam_word_counter.values()))
    total_notspam_wordcount=float(sum(notspam_word_counter.values()))
    for word in spam_word_counter.keys():
        pw = ((float(spam_word_counter[word]+1))+float(notspam_word_counter[word]+1))/float(total_spam_wordcount + total_notspam_wordcount)
        spam_word_contribution[word] = ( float(spam_word_counter[word]+1) / float(total_spam_wordcount+2) * (float(prior['spam'])/float(count_prior)))/ pw
    count = 0
    print("\n10 words most associated with Spam")
    for Key, value in sorted(spam_word_contribution.items(),key=operator.itemgetter(1), reverse= True):
        print (Key,value)
        count=count+1
        if count > 11:
            break
    for word in notspam_word_counter.keys():
        pw = ((float(spam_word_counter[word]+1))+float(notspam_word_counter[word]+1))/float(total_spam_wordcount + total_notspam_wordcount)
        notspam_word_contribution[word] = ( float(notspam_word_counter[word]+1) / float(total_notspam_wordcount+2) * (float(prior['notspam'])/float(count_prior)))/ pw
    count = 0
    print("\n10 words most associated with NotSpam")
    for Key, value in sorted(notspam_word_contribution.items(),key=operator.itemgetter(1), reverse= True):
        print (Key,value)
        count=count+1
        if count > 11:
            break


def print_confusionMatrix():
    tn=0
    fp=0
    fn=0
    tp=0
    for item in test_files:
        if item[2]==item[1]=='notspam':
            tn=tn+1
        if item[2]!=item[1] and item[1]=='spam':
            fp=fp+1
        if item[2]!=item[1] and item[1]=='notspam':
            fn=fn+1   
        if item[2]==item[1]=='spam':
            tp=tp+1
    if technique=="dt":
        classifier="Decision Tree"
    else: classifier="Naive Bayes"
    #Predict Accuracy
    print("\n--------------------------------Percent Accuracy------------------------------\n")
    print("The accuracy of the model is: ",float(tp+tn)*100/len(test_files),"%")  
    
    #Build Confusion Matrix
    print("\n------------------Confusion Matrix for ",classifier," Classifier----------------------\n")
    print("Total = ",len(test_files),"-------------------Predicted NO---------------------Predicted YES")
    print("Actual YES----------------------------TN = ",tn,"---------------------FP = ",fp)
    print("Actual NO-----------------------------FN = ",fn,"-----------------------TP = ",tp)
    
def main(): 
    #global data structures...
    global total_spam_wordcount
    global total_notspam_wordcount
    global test_files
    global spam_word_counter
    global notspam_word_counter
    global prior
    featureList=[]
    dataTable=[] 
    #print(model_file)
    ##Code for Naive Bayes classifier
    if technique=='bayes':
        print("Naive Bayes Classifier")
        if mode=='train':
            #read the train spam and non-spam files and count the words
            print('reading files...\n')
            init_lists(dir_spam,"spam")
            init_lists(dir_notspam,"notspam")
            print(prior)
            
            #Save the model to pickle object..
            print("model to pickle object..")
            output=open("bayes.pkl","wb")
            pickle.dump(spam_word_counter,output)
            pickle.dump(notspam_word_counter,output)
            pickle.dump(prior,output)
            output.close()
            #find top associated words from spam and notspam
            findTopWords()
            print("\nNaive Bayes Training complete.")
            
        if mode=="test":
            #Classifying Data...
            print("classifying Naive Bayes test data...")
            inputfile=open("bayes.pkl","rb")
            spam_word_counter=pickle.load(inputfile)
            notspam_word_counter=pickle.load(inputfile)
            prior=pickle.load(inputfile)
            inputfile.close()

            print("classifying spam folder")
            classifybayes(dir_spam,"spam")
            print("classifying notspam folder")
            classifybayes(dir_notspam,"notspam")
            print_confusionMatrix()

    
    #Code for Decision Tree classifier
    if technique=='dt':
        print("Decision Tree Classifier")
        if mode=='train':
            print('reading files...\n')
            #Get the list of feature from Spam emails...
            featureList=build_featureVector(dir_spam,featureList)
            #Get the list of feature from Not-Spam emails...
            featureList=build_featureVector(dir_notspam,featureList)
            #Get the list of unique items...
            featureList=uniqify(featureList)
            print("Number of features: ",len(featureList))
            print("List of features created.")

            print("creating dataTable for spam...")
            dataTable=build_dataTable(dir_spam,featureList,"spam",dataTable)
            print("creating dataTable for notspam...")
            dataTable=build_dataTable(dir_notspam,featureList,"notspam",dataTable)
            print("dataTable created.")
            random.shuffle(dataTable)
            print("building tree...")
            tree=buildtree(dataTable)
            print("tree was built successfully")
            
            #Save the model to pickle object..
            print("model to pickle object..")
            output=open("DT.pkl","wb")

            pickle.dump(tree,output)
            pickle.dump(featureList,output)
            output.close()
            
            #Printing the Decision Tree...
            print("printing tree...")
            print(featureList[tree.col])
            printtree(tree)
            print("\nDecision Tree training complete.")
            
        if mode=='test':
            print("classifying Decision Tree test data...")
            inputfile=open("DT.pkl","rb")
            tree=pickle.load(inputfile)
            featureList=pickle.load(inputfile)
            inputfile.close()
            
            #Prepare dataTable for classification
            print("creating dataTable for spam...")
            dataTable=build_dataTable(dir_spam,featureList,"spam",dataTable)
            print("creating dataTable for notspam...")
            dataTable=build_dataTable(dir_notspam,featureList,"notspam",dataTable)
            print("dataTable created.")
            
            countTotal=0                       
            for observation in dataTable:
                countTotal+=1
                classValue=-1
                classValue=observation[len(observation)-1]
                classObs=classifydt(observation[0:len(observation)-1],tree)
                print(classObs)
                for key in classObs.keys():
                    test_files.append((countTotal,key,classValue))
                
            print_confusionMatrix() 
                
            
    

if __name__=="__main__":
    main()


# In[ ]:



