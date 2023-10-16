import pandas as pd
import numpy as np
from math import factorial
from itertools import product as cart_product
from datetime import date
from time import time
from bs4 import BeautifulSoup
import requests
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Callable
#Definitions used later, moved here for easy access and testing
#Home/Away is only used for calculating a single game
#N is the number of MC-calcs, use 1 when testing. 100k should take approx. 8 mins.
home = 'MalmöFF'
away = 'AIK'
date = date.today()

def getdata(print=False):
      
    datex = date.today()
#Scrape data from allsvenskan.se

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

    matchday = []
    for i in range(1,30):
        for j in range(1,9):
            matchday.append(i)

    dataout['Matchday'] = matchday[:len(dataout)]
    
    if print:

        path = "DataOutput/allsvenskan2022_"
        dataout.to_csv(f'{path}{datex.strftime("%d-%m-%Y")}.csv')

    return dataout

df = getdata()
print("Data created")



def log_likelihood(dataset: pd.DataFrame, alpha: dict, beta: dict, gamma: float) -> float:
    """
    This function calculates the log-likelihood of the observed data given the parameters alpha, beta, and gamma.
    The log-likelihood is a measure of the probability of observing the given data given the parameters.
    In this case, the data is the number of goals scored by the home and away teams in each match, and the parameters
    are the attacking strength (alpha), defensive strength (beta), and home advantage (gamma) of each team.

    The function iterates over each row in the dataset, which represents a match, and for each match, it calculates
    the log-likelihood of the observed number of goals scored by the home and away teams given the parameters.
    The log-likelihoods for all matches are then summed to give the total log-likelihood.

    Parameters:
    dataset (pd.DataFrame): A DataFrame containing the observed data. Each row represents a match and contains
                            the number of goals scored by the home and away teams.
    alpha (dict): A dictionary mapping each team to its attacking strength.
    beta (dict): A dictionary mapping each team to its defensive strength.
    gamma (float): The home advantage parameter.

    Returns:
    float: The log-likelihood of the observed data given the parameters.
    """
    ll = 0
    for index, row in dataset.iterrows():
        home_lambda = alpha[row.HomeTeam] * beta[row.AwayTeam] * gamma
        away_lambda = alpha[row.AwayTeam] * beta[row.HomeTeam]
        ll += row.FTHG * np.log(home_lambda) - home_lambda - np.log(factorial(row.FTHG))
        ll += row.FTAG * np.log(away_lambda) - away_lambda - np.log(factorial(row.FTAG))
    return ll


def optimise_params(dataset: pd.DataFrame, threshold: float = 1e-10) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    This function optimises the parameters alpha, beta, and gamma using the method of maximum likelihood estimation.
    The parameters are optimised iteratively until the change in the parameters is less than a specified threshold.

    The function uses the observed data (the number of goals scored by the home and away teams in each match) to
    estimate the parameters. The parameters are the attacking strength (alpha), defensive strength (beta), and home
    advantage (gamma) of each team.

    The function iterates over each team in the dataset, and for each team, it calculates the new values of alpha and
    beta based on the observed data and the current values of the parameters. The function also calculates the new
    value of gamma based on the observed data and the current values of alpha and beta.

    The function continues to iterate until the change in the parameters is less than the threshold.

    Parameters:
    dataset (pd.DataFrame): A DataFrame containing the observed data. Each row represents a match and contains
                            the number of goals scored by the home and away teams.
    threshold (float): The threshold for the change in the parameters. The function stops iterating when the change
                       in the parameters is less than this threshold.

    Returns:
    Tuple[Dict[str, float], Dict[str, float], float]: A tuple containing the optimised parameters. The first element
                                                      is a dictionary mapping each team to its optimised attacking
                                                      strength (alpha). The second element is a dictionary mapping
                                                      each team to its optimised defensive strength (beta). The third
                                                      element is the optimised home advantage parameter (gamma).
    """
    # Initialize dictionaries for attacking and defensive strengths of teams
    alpha, beta = {}, {}
    # Initialize home advantage parameter
    gamma = 1.0
    # For each unique team in the dataset, set initial attacking and defensive strengths to 1.0
    for team in dataset.HomeTeam.unique():
        alpha[team] = 1.0
        beta[team] = 1.0
        
    # Initialize an array to store the old values of the parameters
    old = np.zeros(len(alpha)+len(beta)+1)
    # Continue iterating until the change in the parameters is less than the threshold
    while np.abs(np.array(list(alpha.values()) + list(beta.values()) + [gamma]) - old).max() > threshold:
        # Store the old values of the parameters
        old = np.array(list(alpha.values()) + list(beta.values()) + [gamma])
        
        # Calculate the denominator for the new value of gamma
        denom = 0
        for index, row in dataset.iterrows():
            denom += alpha[row.HomeTeam]*beta[row.AwayTeam]
        # Update gamma
        gamma = dataset.FTHG.sum()/denom
        
        # For each unique team in the dataset, calculate the new values of alpha and beta
        for team in dataset.HomeTeam.unique():
            # Get the teams that the current team has played against at home and away
            away_teams = dataset[dataset.HomeTeam == team].AwayTeam
            home_teams = dataset[dataset.AwayTeam == team].HomeTeam
            
            # Update alpha for the current team
            alpha[team] = (
                    (dataset[dataset.HomeTeam == team].FTHG.sum() + dataset[dataset.AwayTeam == team].FTAG.sum()) 
                     /(gamma * np.sum([beta[x] for x in away_teams]) + np.sum([beta[x] for x in home_teams]))
                        )
            # Update beta for the current team
            beta[team] = (
                    (dataset[dataset.AwayTeam == team].FTHG.sum() + dataset[dataset.HomeTeam == team].FTAG.sum()) 
                     /(gamma * np.sum([alpha[x] for x in home_teams]) + np.sum([alpha[x] for x in away_teams]))
                        )
    # Return the optimised parameters
    return alpha, beta, gamma



def score_probability(home: str, away: str, hg: int, ag: int, alpha: Dict[str, float], beta: Dict[str, float], gamma: float) -> float:
    """
    This function calculates the probability of a given score in a football match between two teams.
    The calculation is based on the Poisson distribution, which is a common statistical model for the number of goals scored by a team in a football match.
    The Poisson distribution is parameterised by a rate parameter lambda, which is the expected number of goals.
    The rate parameter is calculated as the product of the attacking strength of the team (alpha), the defensive strength of the opposing team (beta), and a home advantage factor (gamma).
    The function calculates the probability of the home team scoring a certain number of goals (hg), and the away team scoring a certain number of goals (ag).
    The final probability is the product of these two probabilities.

    Parameters:
    home (str): The name of the home team.
    away (str): The name of the away team.
    hg (int): The number of goals scored by the home team.
    ag (int): The number of goals scored by the away team.
    alpha (Dict[str, float]): A dictionary mapping each team to its attacking strength.
    beta (Dict[str, float]): A dictionary mapping each team to its defensive strength.
    gamma (float): The home advantage factor.

    Returns:
    float: The probability of the given score.
    """
    home_lambda = alpha[home] * beta[away] * gamma
    away_lambda = alpha[away] * beta[home]
    hg_prob = home_lambda**hg * np.exp(-home_lambda) / factorial(hg)
    ag_prob = away_lambda**ag * np.exp(-away_lambda) / factorial(ag)
    return hg_prob*ag_prob

def predict_game(home: str, away: str, alpha: Dict[str, float], beta: Dict[str, float], gamma: float) -> Tuple[int, int]:
    """
    This function predicts the outcome of a football match between two teams.
    The prediction is based on the Poisson distribution, which is a common statistical model for the number of goals scored by a team in a football match.
    The Poisson distribution is parameterised by a rate parameter lambda, which is the expected number of goals.
    The rate parameter is calculated as the product of the attacking strength of the team (alpha), the defensive strength of the opposing team (beta), and a home advantage factor (gamma).
    The function generates a random number from the Poisson distribution for the home team and the away team, representing the number of goals scored by each team.

    Parameters:
    home (str): The name of the home team.
    away (str): The name of the away team.
    alpha (Dict[str, float]): A dictionary mapping each team to its attacking strength.
    beta (Dict[str, float]): A dictionary mapping each team to its defensive strength.
    gamma (float): The home advantage factor.

    Returns:
    Tuple[int, int]: The predicted number of goals scored by the home team and the away team.
    """
    return np.random.poisson(alpha[home]*beta[away]*gamma), np.random.poisson(alpha[away]*beta[home])

def compute_points(dataset: pd.DataFrame) -> Dict[str, int]:
    """
    This function computes the total points for each team in a football league based on the results of the matches.
    The function iterates over each row in the dataset, which represents a match, and for each match, it updates the points of the home and away teams based on the number of goals scored by each team.
    If the home team scores more goals, it gets 3 points. If the away team scores more goals, it gets 3 points. If the number of goals is equal, both teams get 1 point.
    The function returns a dictionary mapping each team to its total points.

    Parameters:
    dataset (pd.DataFrame): A DataFrame containing the observed data. Each row represents a match and contains
                            the number of goals scored by the home and away teams (FTHG and FTAG respectively), and the names of the teams (HomeTeam and AwayTeam respectively).

    Returns:
    Dict[str, int]: A dictionary mapping each team to its total points.
    """
    points = dict(zip(dataset.HomeTeam.unique(), [0]*len(dataset.HomeTeam.unique())))
    for index, row in dataset.iterrows():
        if row.FTHG > row.FTAG:
            points[row.HomeTeam] += 3
        elif row.FTHG == row.FTAG:
            points[row.HomeTeam] += 1
            points[row.AwayTeam] += 1
        else:
            points[row.AwayTeam] += 3
    return points

def model_optimisation_and_testing(df, home, away):
    """
    This function optimises the model parameters and performs testing.
    """
    # Load data and fit model
    print("Optimising parameters")
    model = optimise_params(df)
    print("Done Optimising parameters")
    ll = log_likelihood(df, *model)

    # Testing the model
    # Assert that MLE has been found
    # This is done by slightly perturbing the parameters and checking that the log-likelihood decreases.
    # This is because the log-likelihood should be maximised at the optimal parameters.
    for team in df.HomeTeam.unique():
        for i in [0,1]:
            model[i][team] += 0.01
            assert(log_likelihood(df, *model) < ll)
            model[i][team] -= 0.02
            assert(log_likelihood(df, *model) < ll)
            model[i][team] += 0.01

    # Assert scores sum to 1
    # This is a sanity check to ensure that the probabilities of all possible scores sum to 1.
    # This is a fundamental property of probability distributions.
    assert(np.isclose(sum([score_probability(home, away, hg, ag, *model)
                for hg, ag in cart_product(range(100), range(100))]), 1))

    return model

# Call the function
model = model_optimisation_and_testing(df, home, away)


def single_game(home_team: str, away_team: str, score_probability: Callable, model: Tuple[Dict[str, float], Dict[str, float], float]) -> None:
    """
    This function simulates a single game between two teams and visualizes the results in a heatmap.
    The function calculates the probability of each possible score up to a maximum number of goals, and then uses these probabilities to calculate the probabilities of the home team winning, the away team winning, and a draw.
    The function then plots a heatmap of the score probabilities and saves it as a PNG file.

    The function uses the Poisson distribution to model the number of goals scored by each team, which is a common approach in football analytics. The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event.

    Parameters:
    home_team (str): The name of the home team.
    away_team (str): The name of the away team.
    score_probability (Callable): A function that calculates the probability of a given score.
    model (Tuple[Dict[str, float], Dict[str, float], float]): The model parameters. The first element is a dictionary mapping each team to its attacking strength (alpha). The second element is a dictionary mapping each team to its defensive strength (beta). The third element is the home advantage parameter (gamma).
    """
    max_goals = 50
    score_probs = np.zeros((max_goals, max_goals))
    for hg, ag in cart_product(range(max_goals), range(max_goals)):
        score_probs[hg, ag] = score_probability(home_team, away_team, hg, ag, *model)
    plt.figure()

# Compute W,L,D probs
    home_win = 0
    away_win = 0
    for hg, ag in cart_product(range(max_goals), range(max_goals)):
        if hg > ag:
            home_win += score_probs[hg, ag]
        elif hg < ag:
            away_win += score_probs[hg, ag]
 
    print(home_team, 'win probablity:', home_win)
    print('Draw probablity:', score_probs.diagonal().sum())
    print(away_team, 'win probablity:', away_win)

# Plot
    ax = sns.heatmap(score_probs[:6, :6], annot=True, fmt='.3f', cbar=False, cmap="YlGnBu")
    ax.set_xlabel(home_team + ' goals')
    ax.set_ylabel(away_team + ' goals')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.savefig('ChartOutput\heatmap.png')
    

single_game(home, away, score_probability, model)



# Compute remaining fixtures
teams = df.HomeTeam.unique()
matches = list(filter(lambda x: x[0] != x[1], cart_product(teams, teams)))
played = list(zip(df.HomeTeam, df.AwayTeam))
to_play = list(filter(lambda x: x not in played, matches)) 



points = compute_points(df)
sorted_teams = sorted(teams, key=lambda x: points[x], reverse=True)
counts = pd.DataFrame(0, columns=['Winner', 'Europe', 'PlayOffs','Relegated'], index=sorted_teams)

print("Starting simulation")
def simulate_all(N: int = 1) -> pd.DataFrame:
    """
    This function simulates all the remaining matches in the league and updates the points table accordingly.
    It also counts the number of times each team finishes in each position (Winner, Europe, PlayOffs, Relegated).
    The simulation is based on the Poisson model, which assumes that the number of goals scored by a team in a match
    follows a Poisson distribution with a rate parameter that is the product of the team's attacking strength,
    the opposing team's defensive strength, and a home advantage factor.

    Parameters:
    N (int): The number of simulations to run. Default is 1.

    Returns:
    pd.DataFrame: A DataFrame with the proportion of times each team finishes in each position.
    """
    for i in range(N):
    
        simulated_points = points.copy()
    
        for home, away in to_play:
            hg, ag = predict_game(home, away, *model)
            if hg > ag:
                simulated_points[home] += 3
            elif hg < ag:
                simulated_points[away] += 3
            else:
                simulated_points[home] += 1
                simulated_points[away] += 1       
   
        ranking = sorted(simulated_points, key=lambda x: simulated_points[x], reverse=True)
      
        counts.Winner[ranking[0]] += 1
        counts.Europe[ranking[1:3]] +=1
        counts.PlayOffs[ranking[-3]] +=1
        counts.Relegated[ranking[-2:]] += 1
    
    return counts/N

start = time()
SIMRUNS = 5_000
simulation = simulate_all(SIMRUNS)
simulationtime = round(time() - start, 2)/60

print(f"{SIMRUNS} simulations took {simulationtime} minutes")

print(simulation)


