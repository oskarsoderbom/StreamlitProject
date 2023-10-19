import requests
from bs4 import BeautifulSoup
from datetime import date
import pandas as pd 
import re


def getdata():

    URL = "https://www.allsvenskan.se/matcher"
    html = requests.get(URL)
    soup = BeautifulSoup(html.content, features='lxml')
    p_matches = soup.find_all("a", {"class": re.compile('match-id-')})

    matchp = []
    for match in p_matches:
        teams = match.find("div", {"class": "teams"}).text
        result = match.find("div", {"class": "match-score"}).text
        matchp.append([teams, result])

        #Convert to pandas Dataframe and cleans data
    all_matches = pd.DataFrame(matchp, columns=['teams','result'])


    all_matches['HomeTeam'] = all_matches['teams'].str.split('-').str[0]
    all_matches['AwayTeam'] = all_matches['teams'].str.split('-').str[1]
    all_matches['FTHG'] = all_matches['result'].str.split('-').str[0]
    all_matches['FTAG'] = all_matches['result'].str.split('-').str[1]
    dataframe = all_matches.drop(columns=['teams', 'result'])
    dataframe['HomeTeam'] = dataframe['HomeTeam'].str.replace(' ', '')
    dataframe['AwayTeam'] = dataframe['AwayTeam'].str.replace(' ', '')
    dataframe['FTAG'] = dataframe['FTAG'].str.replace(' ', '')
    dataframe['FTHG'] = dataframe['FTHG'].str.replace(' ', '')

    dataframe[["FTAG", "FTHG"]] = dataframe[["FTAG", "FTHG"]].apply(pd.to_numeric)
    dataout = dataframe.drop_duplicates()

    return dataout
