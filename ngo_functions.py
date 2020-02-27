# base imports
import numpy as np
import pandas as pd
import re
import time
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# NLP imports
import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.collocations import *
from nltk import FreqDist
from nltk import word_tokenize
import string
import re
import gensim
from collections import Counter
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

# recommendation imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# visualization imports
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
warnings.filterwarnings("ignore")


################################################################################################


# scraping imports

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# scraping functions

def get_directory_urls():
    '''
    Returns a list of urls that point to each page of ngos in the UNESA directory - there are 25 per page,
    and 11726 ngos for 470 total urls.
    '''
    url = []

    for i in range(0,11726, 25):
        url.append(f'https://esango.un.org/civilsociety/displayConsultativeStatusSearch.do?method=list&show=25&from=list&col=&order=&searchType=&index={i}')
    return url  


def get_ngo_hrefs(driver, unesa_url_list):
    '''
    Get names of all href links for the NGO sites (that's what 'names' is - it contains the program ID at the end)
    '''
    names = []
    for i in range(2,27):
        page = driver.find_elements_by_xpath(f'//*[@id="pagedResults1"]/form/table/tbody/tr[{i}]/td[1]/a')
        names.append(page[0].get_attribute("href"))
    for url in unesa_url_list[1:]:
        driver.get(url)
        for i in range(2,27):
            page = driver.find_elements_by_xpath(f'//*[@id="pagedResults1"]/form/table/tbody/tr[{i}]/td[1]/a')
            try:
                names.append(page[0].get_attribute("href"))
            except:
                if len(names) < 11747:
                    print(f'something went wrong. there are currently {len(names)} items in the list and there should be 11747')
                else:
                    print('list done')
                return names
    
    driver.quit()
    return names


def get_activites(ngo_hrefs):
    # get list of activities
    activities_list = []
    for item in ngo_hrefs:    
        spot = item.find('ProfileDetails&') + 15
        full_url = item[:spot] + 'tab=3&'+ item[spot:]
        activities_list.append(full_url)
    print (len(activities_list))
    return activities_list


def get_table_contents(driver, url, get_name=False):
    '''
    Returns a dictionary of table contents organized by row, along with "bad" rows and "bad" columns
    '''
    
    driver.get(url)
    full_table = driver.find_element_by_xpath('//*[@id="content"]/div/form') 
    
    # instatiating return dictionary and bad rows/columns lists
    bad_row = []
    bad_col = []
    table_dict = {}
    
    if get_name == True:
        table_dict["Organization's Name:"] = driver.find_element_by_xpath('//*[@id="content"]/div/h2').text
    

    # find list of table rows, stored as WebElement
    # any rows in a different format or missing will be stored in "bad_row" and function returns
    try:
        tr = full_table.find_elements_by_css_selector('tr')
    except:
        bad_row.append(tr)
        return table_dict, bad_row, bad_col
    
    # iterate thru table rows to find and store contents of columns
    # any cols in a different format or missing will be stored in "bad_col"
    for row in tr:
        td = row.find_elements_by_css_selector('td')
        if row != tr[-1] and td != []:
            try:
                cat = td[0].text
                val = td[1].text
                table_dict[cat] = val
            except:
                bad_col.append(td)

    # returns a dictionary, bad rows and bad cols
    return table_dict, bad_row, bad_col


def scrape_activity(driver, activities_list):
    # Run cell to scrape all activities of full names list with tab=3 appended and dump into pickle file

    activities_df = pd.DataFrame()
    problem_row = [] # broken rows/columns
    problem_dict = [] # broken dictionaries
    table2_list = [] # working list of table dictionaries from tab 3 on website for this function
    counter = 0

    for url in activities_list:

        table_dict, bad_row, bad_col = get_table_contents(driver, url, get_name=True)

        # mark as a problematic page 
        if bad_row != [] or bad_col != []:
            problem_row.append(activities_list.index(url))

        # check for empty dictionary and add to list
        if table_dict != {}:
            table2_list.append(table_dict)
        else:
            problem_dict += [url, activities_list.index(url)]

        counter += 1

        # add to df and refresh access every 25 scrapes
        if counter % 10 == 0:
            activity = pd.DataFrame(table2_list)
            activities_df = pd.concat([activities_df,activity], sort=False)    # send to df
  
            # open a file, where you ant to store the data
            file = open('pickling_file', 'wb')

            # dump information to that file
            pickle.dump(activities_df, file)

            # close the file
            file.close()
            
            # go to page that allows access without login in case it times out
            driver.get("https://esango.un.org/civilsociety/withOutLogin.do?method=getOrgsByTypesCode&orgTypeCode=6&orgTypName=Non-governmental%20organization&sessionCheck=false&ngoFlag=")
            table2_list = []
     
    if counter % 100 == 0:
        print(f'Collected {counter} NGOs, {len(activities_list) - counter} NGOs remaining.')
        
    if (len(problem_row) > 0) & (len(problem_dict) > 0):
        print ("broken rows/columns:", problem_row, "broken dictionaries:",problem_dict,
               "working list of table dictionaries from tab 3 on website for this function:",table2_list)

    return activities_df        
        


            
def loadData(pkl_file_name): 
    # for reading also binary mode is important 
    dbfile = open(pkl_file_name, 'rb')      
    db = pickle.load(dbfile) 
    for keys in db: 
        print(keys, '=>', db[keys]) 
    dbfile.close() 


################################################################################################
    
# misc helper functions

def lemmatize_stemming(text):
    '''
    Return lemmatized text
    '''
    word = WordNetLemmatizer().lemmatize(text, pos='v')
    stemmer = SnowballStemmer('english',ignore_stopwords=True)
    return stemmer.stem(word)

def preprocess(text, stopwords_list=stopwords.words('english'), lem=False):
    '''
    Returns text that is not longer than 3 chars or stop words (as defined by stopwords_list and gensim library) 
    '''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopwords_list:
            if lem == False:
                result.append(token)
            else:
                result.append(lemmatize_stemming(token))
    return result

def show_range(num_range, df):
    return df.iloc[num_range[0]:num_range[1]]


def calc_sim_range(sim_scores, percentage):
    '''
    Calculate the top x percent of similarity scores - helper function to the rec system to indicate how many to return 
    in search results
    '''
    top = sim_scores[1][1]
    bottom = top - (percentage * top)
    sim_end = 0
    for i in sim_scores:
        if i[1] >= bottom:
            sim_end +=1
        else:
            break
    return sim_scores[1:sim_end]


def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()
    
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def find_websites(driver, ngo_content):
    no_site = ngo_content[ngo_content['web site'].isna() == True]
    # function to find website
    non_site = []
    found_site = []

    for name in no_site['organizations name']:
        # index retained from initial list
        ngo_idx = no_site.index[no_site['organizations name'] == name][0] 

        #create
        country = ngo_content.iloc[ngo_idx]['country of activity']
        if type(country) == float:  # ignore nans
            continue
        search = [name]
        if type(country) == list:
            if len(country) > 4:
                country = country[:4]
            search += country
        if type(country) != float and type(country) != list:
            search.append(country)
        search = ' '.join(search)
        search += ' '

        driver.get('https://www.google.com/')
        srch = driver.find_element_by_xpath('//*[@id="tsf"]/div[2]/div[1]/div[1]/div/div[2]/input')
        srch.send_keys(search)
        button = driver.find_element_by_xpath('//*[@id="tsf"]/div[2]/div[1]/div[3]/center/input[1]')
        button.click()

        # get first link on the page
        block = driver.find_element_by_xpath('//*[@id="rso"]/div/div/div[1]/div')
        first = block.find_element_by_xpath('//*[@id="rso"]/div/div/div[1]/div/div/div[1]/a/h3')
        link = driver.find_element_by_xpath('//*[@id="rso"]/div/div/div[1]/div/div/div[1]/a').get_attribute('href')

        ending = [link.find('.com'), link.find('.org'), link.find('/en')]
        for end in ending:
            if end != -1:
                ending = end


        full_name = name
        if len(name.split(' ')) > 3:
            name = generate_ngrams(name.lower(), 3)
        else:
            name = name.lower().split()

        # clean text from first link of punctuations ("first")
        first_t = first.text
        for pun in [string.punctuation[i] for i in range(len(string.punctuation))]:
            first_t = first_t.replace(pun, '')



        if any([nam in generate_ngrams(first.text, 3) for nam in name]):
            if 'wiki' in link or 'facebook' in link: # or in_link == False:
                ngo_content.loc[ngo_idx,'missing'] += ', website'
                non_site.append(ngo_idx)
                continue
            else:
                found_site.append(ngo_idx)
                ngo_content.loc[ngo_idx,'web site'] = link

    driver.quit()
    return ngo_content

    
################################################################################################    

# filter functions


def ngo_filter_lang(langs, df):
    '''
    Filters dataframe to return only observations containing the requested list of languages.
    '''
    df_lang = df.copy()

    lang_list = []

    for idx, row in df_lang.iterrows():
        if type(row.languages) == float:
            lang_list.append(0)
            continue
        lan = [x.strip() for x in row.languages]
        inter = len(set(lan).intersection(set(langs)))
        
        if inter >= 1:
            if set(lan) == set(langs):
                inter = len(langs) + 0.5
            lang_list.append(inter)
        else:
            lang_list.append(0)

    df_lang['in_lan_list'] = pd.Series(lang_list)
    return df_lang[df_lang['in_lan_list'] > 0].sort_values('in_lan_list', ascending=False)#.reset_index(drop=False)


def ngo_filter_country(countries, df):
    '''
    Filters dataframe to return only observations containing the requested list of countries. Fill with 'any' for no preference.
    '''
    df_contry = df.copy()
    df_contry['in_ct_list'] = 0

    ct_list = []

    for idx, row in df_contry.iterrows():
        if type(row.country) == float:
            continue
            
        else:
            ct = [x.strip() for x in row.country]
            inter = len(set(ct).intersection(set(countries)))
            if inter > 0:
                if set(ct) == set(countries):
                    inter = len(countries) + 0.5
                df_contry.loc[idx, 'in_ct_list'] = inter


    return df_contry[df_contry['in_ct_list'] > 0].sort_values('in_ct_list', ascending=False)


def ngo_filter_age(age, df):
    '''
    Filters dataframe to return only observations containing the requested ages. Four options:
        'recent' (for NGOs established after 2006),
        'adolescent' (for NGOs established between 1992-2005),
        'mature' (for NGOs established before 1992),
        'any' (for no preference)
    '''
    if age != 'any':
            if age == 'recent':
                return df[(df['yr_est'] < 2020) & (df['yr_est'] > 2005)]

            elif age == 'adolescent':
                return df[(df['yr_est'] < 2005) & (df['yr_est'] > 1992)]

            elif age == 'mature':
                return df[df['yr_est'] < 1992]
        
    return df


def ngo_filter(age, langs, countries, data, only=True):
    '''
    Takes in a dataframe along with age and language list, and returns a dataframe filtered for those aspects.
    Will return ngos established within a 2-year timeframe of selected ngo's age. 
    Will return observations that include at least one of the elements in langs and countries.
    
    lang: must be a list, even if one element
    age: how many years (from 2019) since the ngo was established
    countries: must be a list, even if one element
    
    only: if true, will return observations that fulfill every match. Otherwise, will return a concatenated 
    dataframe of each matching component
    '''
    df = pd.DataFrame()
    if type(langs) == list:
        langs = [l.capitalize() for l in langs]
    
    if only == False:
        df_age = ngo_filter_age(age, data)
        df_lang = ngo_filter_lang(langs, data)
        lang_count = df_lang['in_lan_list']
        df_country = ngo_filter_country(countries, data)
        country_count = df_country['in_ct_list']
        
        df = pd.concat([df_age, df_country, df_lang]).drop_duplicates('name')
        df['num_sim'] = lang_count.add(country_count)
        
        return df.sort_values('num_sim',ascending=False).reset_index()[['num_sim','name','yr_est','languages','fs','mdg','ms',
                                                                     'website','expertise','country', 'index']]
    
    if only == True:
        df = ngo_filter_lang(langs, data)
        df = ngo_filter_country(countries, df)
        df['num_sim'] = df['in_lan_list'].add(df['in_ct_list'])  
        df = ngo_filter_age(age, df)
    
        return df.sort_values('num_sim',ascending=False).reset_index()[['num_sim','name','yr_est','languages','fs','mdg','ms',
                                                                     'website','expertise','country', 'index']]


    
    
    
def clean_df(nf):    
    nf.country = nf.country.apply(lambda x: x if type(x) == float else x.replace('[','').replace(']','').replace("'",'').split(','))
    nf.expertise = nf.expertise.apply(lambda x: x if type(x) == float else x.replace('[','').replace(']','').replace('"','').replace("'",'').replace(":",'').split(','))
    nf.languages = nf.languages.apply(lambda x: x if type(x) == float else x.replace('[','').replace(']','').replace("'",'').split(','))
    nf.languages = nf.languages.apply(lambda x: x if type(x) == float else [lan.strip() for lan in x]) # removes whitespace in each 
    return nf

    
# recommendation function

def ngo_rec(idx, dff):
    '''
    Takes in an org name (or index of that org in the df) and dataframe.
    Then returns top 10 similar orgs based on cosine similarity of vectorized mission statement, funding structure, 
    MDG, area of expertise and country of activity.
    '''
    if type(idx) == str:
        idx = dff[dff.name == idx].index
        

    ngo_name = dff[dff.index == idx]['name'].iloc[0]
    print ('Getting recommendation for:', ngo_name)
    print(f"Summary of {ngo_name}'s Content:")
    # wordcloud with regular words
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").\
    generate(dff[dff.index == idx]['bag_of_description'].iloc[0])

    plt.figure(figsize=(15,15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    
    # tfidf vectorize descriptions in dff
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(dff['bag_of_description'])

    # get dot product of tfidf matrix on the transposition of itself to get the cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    if len(sim_scores) > 1:
        print (sim_scores[:5])
        percent = 0.05
        while len(calc_sim_range(sim_scores, percent)) < 5:
            if percent > .5:
                break
            percent += 0.05
        sim_scores = calc_sim_range(sim_scores, percent)
        ngo_indices = [i[0] for i in sim_scores] 
        ngo_sim_scores = [i[1] for i in sim_scores] 

        rec_df = dff.iloc[ngo_indices][:len(sim_scores)]
        rec_df['sim_score'] = ngo_sim_scores

        rec_df = clean_df(rec_df)
        rec_df = rec_df[['name','country','languages','yr_est','fs','mdg','ms','expertise',
                                           'website', 'sim_score']]
        
        rec_df.columns = ['name', 'country', 'languages spoken', 'year established', 'funding sources', 
                          'millenium development goals', 'mission statement',
                          'areas of expertise', 'website', 'similarity score (how close to seed NGO)']
        
        return rec_df
    
    else:
        print ('There are no dataframes that directly match this NGO. Please try again.')
        return pd.DataFrame
 
 
 
    
    
# full, combined function

def ngo_filter_rec(data):
    # seeking to make a 'quit' function in this
#     answer = input("Enter 'quit' at any time to exit the system.. Press 'ok' and enter to continue. \n")
#     while answer != 'quit':

    answer = input('Please enter desired age range: \n\
    recent (for NGOs established after 2006) \n\
    adolescent (for NGOs established between 1992-2005) \n\
    mature (for NGOs established before 1992) \n\
    any (for no preference) ').lower()
    age = answer

    answer = input('Please enter desired languages, separated only by commas ( or any for no preference): \n')
    langs = answer.capitalize().strip().replace(' ','').split(',')


    answer = input('Please enter desired countries, separated only by commas ( or any for no preference): \n')
    countries = answer.strip().replace(' ','').split(',')

    answer = input("Please indicate whether you would like NGOs that satify all of these attributes ('all'),\
    or at least one of them ('any'): ")

    if answer.lower() == 'all':
        only = True
    elif answer.lower() == 'any':
        only = False
    print(only)

    d_filter = ngo_filter(age, langs, countries, data, only)
    bottom_range = 0
    top_range = 6
    print (d_filter.reset_index()[['name','index']])

    answer = input("Please review the above NGOs and enter the index (the number to the left of the name) of the\
        organization you would like to see recommendations for. To see more, please enter 'more' : ").strip().lower()

    while answer == 'more':
        bottom_range += 5
        top_range += 5
        print(show_range((bottom_range,top_range), d_filter.reset_index())[['name','index']])
        answer = input("Please review the above NGOs and enter the index (the number to the left of the name) of the \
        organization you would like to see information about as well as recommendations for. \
        \To see more, please enter 'more' : ").strip().lower()

    idx = int(answer.strip())
    good = False
    while good == False:
        try:        
            idx = int(idx)
            good = True
        except:
            idx = input("Invalid index, try again: ")
            
    org_name = data.name[idx]
    top_recs = ngo_rec(idx, data)
    top_recs.to_excel(f'top_ngos_{org_name}.xlsx')
    top_recs.reset_index(inplace=True, drop=True)
    
    print (f'Your top recommendations are for NGOs: {top_recs.name[:5]}. Please see the downloaded Excel file titled top_ngos_{org_name}.xlsx for more info.')
    return 
