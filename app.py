import streamlit as st
import pandas as pd
import main
from simulation_functions import single_game, score_probability, model_optimisation_and_testing
from data_functions import getdata

def run_simulation():
    # Call the main function from your main.py file
    simulation = main.main()
    return simulation

st.title('Simulation Dashboard')

if st.button('Start Simulation'):
    # Run the simulation when the button is clicked
    simulation = run_simulation()

    # Display the simulation output in a table
    st.table(pd.DataFrame(simulation))



st.title('Simulation Dashboard')

if st.button('Simulate Single Game'):
    # Run the single game simulation when the button is clicked
        # Assuming these are your model parameters
    df = getdata()
    home = 'AIK'
    away = 'MalmöFF'
    model = model_optimisation_and_testing(df, home, away)
    fig, homewin, awaywin = single_game(home, away, score_probability, model)
    st.write(f'Probability of {home} winning: {homewin*100:.1f}%')
    st.write(f'Probability of {away} winning: {awaywin*100:.1f}%')

    # Display the probabilities and heatmap
    st.pyplot(fig)