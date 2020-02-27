# NGO_Network

NGO Network is a project intended to reduce the labor of NGOs seeking out each other, as well as individuals seeking out NGOs. Unlike other connections, NGO Network allows users to look though a database of international, UN-affiliated NGOs and generate novel recommendations. This algorithm generates a more nuanced sort of relationship, as it sorts by similarity score that determiend by overlap across 12 different features.

The data on nearly 12,000 NGOs were scraped from a public website, at <https://esango.un.org/civilsociety/login.do>. Please reach out to me regarding more information on how I conducted the scrape.


**How you can use NGO Network:**
ngo_script.py initiates a call and response to the user - 
You will be asked to fill out desired aspects of NGOs that are stored in the dataframe including: age of NGO, language spoken, country of activity and if you'd like to be strict about all features being present in the search.

It then allows you to explore NGO names in a scrolling basis before getting a recommendation, and then opens a word cloud description of the chosen NGO's mission statement. You are able to interact with and save this file, and then Excel file with top recommendations will be saved to your computer.


Below is an example of the wordcloud and dataframe produced by the function:


!['Example WordCloud for Gender Equity NGO'](https://github.com/rebecca-hh-rosen/ngo_net/blob/master/gender_equality_wc.png)


!['Example Data Frame for Gender Equity NGO'](https://github.com/rebecca-hh-rosen/ngo_net/blob/master/example_pic.png?raw=true "Optional Title")


Enjoy! 

*Feel free to reach out to me at rebeccahhrosen@gmail.com for any questions, inquiries or comments. If you liked this, please check out my Medium page at <https://medium.com/@rebeccahhrosen>.*
