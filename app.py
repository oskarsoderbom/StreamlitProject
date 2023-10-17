import streamlit as st
import pandas as pd
import main
from simulation_functions import single_game, score_probability, model_optimisation_and_testing
from data_functions import getdata

def run_simulation(N: int = 5_000):
    # Call the main function from your main.py file

    simulation = main.main(N)
    return simulation

st.title('Simulation Dashboard')

N = st.number_input('Enter number of simulations', min_value=1, value=5000, step=1)
if st.button('Start Simulation of remaining fixtures'):
    # Run the simulation when the button is clicked
    simulation = run_simulation(N)

    # Display the simulation output in a table
    st.table(pd.DataFrame(simulation))

df = getdata()
home_teams = df['HomeTeam'].unique().tolist()
away_teams = df['AwayTeam'].unique().tolist()
home = st.selectbox('Select Home Team', home_teams, index=home_teams.index('AIK'))
away = st.selectbox('Select Away Team', away_teams, index=away_teams.index('MalmöFF'))

if st.button('Simulate Single Game'):
    # Run the single game simulation when the button is clicked
        # Assuming these are your model parameters
    
    model = model_optimisation_and_testing(df, home, away)
    fig, homewin, awaywin = single_game(home, away, score_probability, model)
    st.write(f'Probability of {home} winning: {homewin*100:.1f}%')
    st.write(f'Probability of {away} winning: {awaywin*100:.1f}%')

    # Display the probabilities and heatmap
    st.pyplot(fig)