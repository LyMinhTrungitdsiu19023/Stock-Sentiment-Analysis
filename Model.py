#IMPORT LIBRARIES
import pandas as pd
import re
from nltk.tokenize import sent_tokenize 
import numpy as np
from time import time
import json
from datetime import datetime

#Load the exceptions of Company's name base on Tickers into a dictionary
with open('ticker_with_Companyname.json', encoding='utf-8') as file:
    dic_companyname = json.load(file)

#Load the exceptions of Company's name without Ticker into a dictionary
with open('Companyname_without_ticker.json', encoding='utf-8') as files:
    ex_dic_companyname = json.load(files)

#Load the Positive and Negative words datasets
with open('positive_word.txt', encoding="utf8") as pos:
    pos = tuple(pos.read().split("\n")) 
with open('negative_word.txt', encoding="utf8") as neg:
    neg = tuple(neg.read().split("\n")) 

#Load the exception tickers that contain the companies have special tickers like: VND, USD, VTV, HCM...
with open('exception_tickers.txt') as txt:
    exception = set(txt.read().split("\n")) 
#load the official tickers in Stock
with open('tickers.txt') as txt:
    all_tickers = set(txt.read().split("\n"))
    
#Function to get keys from value in dictionary
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

#Function to convert string to list
def stringToList(string):
    listRes = list(string.split(","))
    return listRes

#Function to input dataset
def input_data(path):
    df = pd.read_csv(path)
    df = df.drop(labels = ['Ticker'], axis = 1) #if the initial dataset already have Ticker columns -> this cell can be hidden 
    df["Body"] = df["Title"] + '. ' + df["Description"] + '. ' + df["Body"] + "."
    df = df.dropna() 
    df = df.reset_index(drop=True) # drop=True option avoids adding new index column with old index values
    return df 


#Function to find News have Tickers
def process_tickers(df, ticker):
    df["Regex"] = df["Body"].str.contains(r"\b[A-Z][A-Z0-9][A-Z0-9]\b", regex = True) 
    df = df.loc[df["Regex"] == True]
    lst_news = [] 
    lst = []
    sub_lst = []
    for i in df["Body"]:         
        news = set(re.findall(r"\b[A-Z][A-Z0-9][A-Z0-9]\b", i)).intersection(all_tickers)
        news -= exception
        news |= {dic_companyname[key] for key in list(dic_companyname.keys()) if re.search(key, i.lower()) }

        lst_news.append(','.join(news))
        
    df = df.drop(labels='Regex', axis=1)
    df.insert(loc=1, column='Ticker', value=lst_news)
    df = df.loc[df["Ticker"].str.len() != 0] 
    df = df.reset_index(drop = True)
    df = df.loc[df["Ticker"].str.contains(ticker.upper())]
    
    for i in range(len(df)):
        lst = stringToList(df["Ticker"].iloc[i])
        sub_lst.append(lst)
    df["Ticker"] = sub_lst
    return df

#Function clean News
def preprocess(text_in_df):
    lst = []
    for text in text_in_df:
        text = text.replace('\xa0', ' ').replace('*', ' ').replace('·', ' ').replace('...', '…').replace("\'\'", " ").replace("\'", " ").replace('"', '')
        text = re.sub(r'([\w\"\'\)]{2,})(\.)([\w^0-9\"\'\(]+)', r'\1\2 \3', text)
        text = re.sub(r'(\d+)(\.)( )(\d)', r'\1\2\4', text)
        text = re.sub(r' +|=+|>+', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'([\w…\'\"\) ]+)(\n)', r'\1. \2', text)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'([\.?!])(\n)', r'\1 \2', text)
        text = text.replace('. . ', '.')
        text = text.replace('TP. HCM', 'TP.HCM').replace('. com', '.com').replace('. vn', '.vn')
        text = re.sub(r'(…)( [A-ZẢÔẾỦ])', r'\1.\2', text)
        text = re.sub(r'(…)( [a-zảôếủ])', r'\1;\2', text)
        text = text.replace('…,', '…;').replace(',…', '…')
        text = re.sub(r'\.(\d{3})', r',\1', text)
        text = re.sub(r'\b(\d{1, 3})*,(\d{1,2}%?)\b', r'\1.\2', text)
        lst.append(text)
    
    df_preprocessed = pd.DataFrame(lst)
    df_preprocessed.columns = ["Body"]
            
    return df_preprocessed

#Function to tokenize sentence
def sentence_tokenize(df):
    lst = []
    for i in df["Paragraph"]:
        sent_token = sent_tokenize(i)
        lst.append('@ '.join(sent_token))
        
    new_df = pd.DataFrame({'Time' : df["Time"].tolist(),
                   'Ticker': df["Ticker"].tolist(),
                   'Body': df["Body"].tolist(),
                      'URL': df["URL"].tolist(),
                      'Paragraph': lst})
    return new_df

def build_df_preprocessed(df_pre, df):
    df_pre["Time"] = df["Time"].tolist()
    df_pre["Ticker"] = df["Ticker"].tolist()
    df_pre["URL"] = df["URL"].tolist()
    df_pre = df_pre[['Time', 'Ticker', 'Body', 'URL']]
    return df_pre
def spit_dataset(df_pre): 
    dataset_multicker = df_pre.loc[df_pre["Ticker"].str.len() > 1]
    dataset_singleticker = df_pre.loc[df_pre["Ticker"].str.len() == 1] 
    return dataset_singleticker, dataset_multicker
def process_dataset_single_ticker(dataset_singleticker): 
    sent_lst_singleticker = []
    sub_lst = []
    para_lst = []
    paresub_lst = []
    lst = []
    for j in range(len(dataset_singleticker)):
        for i in dataset_singleticker["Ticker"].iloc[j]:
            

            if i in list(ex_dic_companyname.values()):
                para_lst = re.findall('([^\n]*'+ get_keys_from_value(ex_dic_companyname, i)[0] + '[^\n]*)', dataset_singleticker["Body"].iloc[j], re.IGNORECASE)
            else:
                para_lst = re.findall('([^\n]*'+ str(i) + '[^\n]*)', dataset_singleticker["Body"].iloc[j])
            paresub_lst.append('\n'.join(para_lst))

            
    dataset_singleticker.insert(loc=4, column='Paragraph', value=paresub_lst)


    return dataset_singleticker

def process_dataset_multiticker():
    time_lst = []
    ticker_lst = []
    body_lst = []
    url_lst = []
    para_lst = []
    parasub_lst = []
    for j in range(len(dataset_multicker)):
        for i in dataset_multicker["Ticker"].iloc[j]:
            time = dataset_multicker["Time"].iloc[j]
            time_lst.append(time)
            ticker_lst.append([i])
            body = dataset_multicker["Body"].iloc[j]
            body_lst.append(body)
            url = dataset_multicker["URL"].iloc[j]
            url_lst.append(url)

            if i in list(ex_dic_companyname.values()):
                para_lst = re.findall('([^\n]*'+ get_keys_from_value(ex_dic_companyname, i)[0] + '[^\n]*)', dataset_multicker["Body"].iloc[j], re.IGNORECASE)
            else:
                para_lst = re.findall('([^\n]*'+ str(i) + '[^\n]*)', dataset_multicker["Body"].iloc[j])
            parasub_lst.append('\n'.join(para_lst))
     
    new_df = pd.DataFrame({'Time' : time_lst,
                   'Ticker': ticker_lst,
                   'Body': body_lst,
                      'URL': url_lst,
                      'Paragraph': parasub_lst})        
    return new_df

def find_sentences_clauses(df): 
    sen_lst = []
    sen_sub_lst = []
    clause_lst = []
    addlst = []
    para = []
    for j in range(len(df)):
        for i in df["Ticker"].iloc[j]:
            clause_sublst = []
            if i in list(ex_dic_companyname.values()):
                sen_lst = re.findall('([^@]*'+ get_keys_from_value(ex_dic_companyname, i)[0] + '[^@]*)', df["Paragraph"].iloc[j], re.IGNORECASE)
            else:
                sen_lst = re.findall('([^@]*'+ str(i) + '[^@]*)', df["Paragraph"].iloc[j]) 
                
            if sen_lst != []:
                for k in sen_lst:
                    if re.search(r'[;]', k):
                        clause_lst = re.findall('([^;,]*'+ '[^A-Za-z,]*' + str(i) +'[^a-z;]*'+ '[^,…;]*)', k)
                    elif re.search(r'([0-9],)', k):
                        clause_lst = re.findall('([^;]*'+ '[^A-Za-z,]*' + str(i) +'[^a-z]*'+ '[^…;]*)', k)
                    elif re.search(r'([A-Z][A-Z0-9][A-Z0-9],…)', k):
                        clause_lst = re.findall('([^…;]*'+ '[^A-Za-z,]*'+ str(i) +'[^A-Za-z]*'+ '[^,…;]*)' , k)
                        
                    elif re.search(r'([A-Z][A-Z0-9][A-Z0-9]….)', k):
                        clause_lst = re.findall('([^…;]*'+ '[^A-Za-z]*'+ str(i) +'[^A-Za-z]*'+ '[^,…;]*)' , k)
                    else:
                        clause_lst = re.findall('([^,…;]*'+ '[^a-z,]*'+ str(i) +'[^a-z]*'+ '[^,…;]*)' , k)
                        
                    clause_sublst.append(','.join(clause_lst))
                    
                addlst.append('|'.join(clause_sublst))
            else: 
                addlst.append([])
                
        sen_sub_lst.append(''.join(sen_lst))
    
    for text in df["Paragraph"]:
        text = text.replace('@', '')
        para.append(text)
    df["Paragraph"] = para
    df.insert(loc=5, column='Sentences', value=sen_sub_lst) 
    df.insert(loc=6, column='Clauses', value=addlst) 

    return df   

def concat_dataframe(dataset_singleticker, dataset_multicker, ticker):
    final_df = pd.concat([dataset_singleticker, dataset_multicker]) 
    lst = []
    for i in final_df["Ticker"]: 
        lst.append("".join(i))
    final_df["Ticker"] = lst
    final_df = final_df.reset_index(drop = True)
    final_df = final_df.loc[final_df["Paragraph"].str.len() > 50]
    return final_df.loc[final_df["Ticker"] == ticker.upper()]

def polarity_score(text):

    pos_score = sum(0.05 for word in pos if text.lower().find(word) != -1 )
    if re.search(r'(tăng(( gần)|( đến)|( hơn))? \d)|(từ.*(lên))|(\+ ?\d\.?\d?%)|(mua \d)', text):
        pos_score +=0.1
    neg_score = sum(-0.05 for word in neg if text.lower().find(word) != -1)
    if re.search(r'(giảm(( gần)|( đến)|( hơn))? \d)|(từ.*((xuống)|(còn)))|(\- ?\d\.?\d?%)', text):
        pos_score -=0.1
    return pos_score + neg_score

def sentiment(score): 
      if score >= 0.05:
        return 'Positive'
      elif score <= -0.05:
        return 'Negative'
      else:
        return 'Neutral'
    
def label(df, start, end, choose):
    seg_lst = []
    
    for j in range(len(df)):
        if any(word in df["Sentences"].iloc[j] for word in pos) and any(word in df["Sentences"].iloc[j] for word in neg) and len(set(re.findall(r"\b[A-Z][A-Z0-9][A-Z0-9]\b", df["Sentences"].iloc[j]))) > 1:      
            score = polarity_score(df["Clauses"].iloc[j])
        else:
            score = polarity_score(df["Sentences"].iloc[j])
        label = sentiment(score)
        seg_lst.append(label)
    df.insert(loc=7, column='Label', value=seg_lst)
    
    df['Time'] = pd.to_datetime(df['Time'])
    df['Time'] = df['Time'].dt.date
    df = df.drop(labels='Clauses', axis=1)
    df = df.drop(labels='Body', axis=1)
    if choose.lower() == 'yes':
        mask = (df['Time'] >= datetime.strptime(start, '%Y-%m-%d').date()) & (df['Time'] <= datetime.strptime(end, '%Y-%m-%d').date())
        df = df.loc[mask]
        df.sort_values(by = 'Time', inplace = True)
        df = df.reset_index(drop = True)
        return df
    
    if choose.lower() == 'no':
        df.sort_values(by = 'Time', inplace = True)
        df = df.reset_index(drop = True)
        return df



if __name__ == '__main__':
    df = input_data('MarketNews.csv')
    # df = df.head(3000)
    print("Load data Successfully")
    ticker = input("Which Ticker do you want to get sentiment?\n >>> ")
    choose = input("Choose days? Yes or No\n >>> ")
    if choose.lower() == 'yes':
        start = input("Enter Start day (YYYY-MM-DD)\n>>> ")
        end = input("Enter End day\n>>> ")
    else:
        start = 0
        end = 0
    print(">>> Proccessing................. >>>")
    start_time = time() 
    df = process_tickers(df, ticker)
    df_pre = preprocess(df["Body"])
    df_pre = build_df_preprocessed(df_pre, df) 
    dataset_singleticker = spit_dataset(df_pre)[0]
    dataset_multicker = spit_dataset(df_pre)[1]
    dataset_singleticker = process_dataset_single_ticker(dataset_singleticker)

    dataset_multicker = process_dataset_multiticker()
    dataset_singleticker = sentence_tokenize(dataset_singleticker) 
    dataset_multicker = sentence_tokenize(dataset_multicker)

    dataset_singleticker = find_sentences_clauses(dataset_singleticker)
    dataset_multicker = find_sentences_clauses(dataset_multicker)

    final_df = concat_dataframe(dataset_singleticker, dataset_multicker, ticker)
    labeled_df = label(final_df,start, end, choose) 
    print(">>> Successfully >>>")
    labeled_df.to_csv('Ticker labeled.csv',encoding='utf-8-sig')
    print(">>> You can find the result in Ticker labeled.csv file in this folder >>>")

    print("Total processing time: ", time() - start_time)