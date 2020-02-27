import pandas as pd
from fp_functions import *  # contains thorough imports list


# import csv 
ngo_df = pd.read_csv('ngo_df.csv')


# call rec - initiates a call and response to fill out desired aspects of the NGO: age of NGO, language spoken, 
# country of activity and if you'd like to be strict about all features
# allows you to explore NGO names before getting a recommendation, and then shows a word cloud description of the
# chosen NGO's mission statement before saving an Excel file with top recommendations.

ngo_filter_rec(ngo_df)




