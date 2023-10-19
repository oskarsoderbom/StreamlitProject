import streamlit as st

st.set_page_config(
    page_title="Model",
    page_icon="📖",
)

st.markdown(
    """
# Football Match Simulator

The model used in this application is a football match simulator that predicts the outcome of matches based on historical data. 
It uses a statistical approach known as the Poisson distribution, which is a common method for modeling the number of goals scored in a football match.

Here's a step-by-step breakdown of how the model works:

1. Data Collection:
The model first collects historical match data, including the teams that played, the number of goals scored by each team, and the location of the match (home or away). 
This data is scraped from a website and organized into a structured format.

2. Parameter Estimation:
The model then estimates three parameters for each team: attacking strength (alpha), defensive strength (beta), and home advantage (gamma). 
These parameters are estimated using a method called maximum likelihood estimation, which finds the values that make the observed data most likely. 
The parameters are estimated iteratively, with the process continuing until the change in the parameters is less than a specified threshold.

3. Score Probability Calculation:
With the parameters estimated, the model can calculate the probability of any given score in a match between two teams. 
This is done using the Poisson distribution, with the rate parameter (lambda) being the product of the attacking strength of the team, 
the defensive strength of the opposing team, and the home advantage factor. The model calculates the probability of the home team scoring a certain number of goals, 
and the away team scoring a certain number of goals. The final probability of a given score is the product of these two probabilities.

4. Match Outcome Prediction:
The model predicts the outcome of a match by generating a random number from the Poisson distribution for the home team and the away team, 
representing the number of goals scored by each team. The team with the higher number of goals is predicted to win the match.

5. League Simulation:
The model can simulate the remaining matches in the league and update the league table accordingly. 
It does this by predicting the outcome of each remaining match and updating the points for each team based on these predictions. 
The model also counts the number of times each team finishes in each position (Winner, Europe, PlayOffs, Relegated) over multiple simulations.

6. Result Visualization:
The results of the simulation can be visualized in various ways. 
For example, the model can generate a heatmap showing the probability of each possible score in a match between two teams. 
It can also generate a league table showing the predicted final standings of the teams.

In summary, this model provides a statistical approach to predicting football match outcomes and simulating league standings. 
It uses historical data and the Poisson distribution to make its predictions, and it provides a variety of ways to visualize the results.

"""
)