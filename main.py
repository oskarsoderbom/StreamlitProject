from simulation_functions import model_optimisation_and_testing, single_game, score_probability, simulate_all
from data_functions import getdata
from time import time

def main():

    df = getdata()
    home = 'AIK'
    away = 'MalmöFF'

    # Call the function
    model = model_optimisation_and_testing(df, home, away)

    single_game(home, away, score_probability, model)

    start = time()
    SIMRUNS = 5_000
    simulation = simulate_all(N= SIMRUNS, dataframe=df, model=model)
    simulationtime = round(time() - start, 2)/60

    print(f"{SIMRUNS} simulations took {simulationtime} minutes")

    print(simulation)
    return


if __name__ == '__main__':
    main()