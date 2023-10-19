from functions.simulation_functions import model_optimisation_and_testing, single_game, score_probability, simulate_all
from functions.data_functions import getdata
from time import time

def main(runs: int = 5_000):

    df = getdata()
    home = 'AIK'
    away = 'MalmöFF'

    # Call the function
    model = model_optimisation_and_testing(df, home, away)

    single_game(home, away, score_probability, model)

    start = time()
    SIMRUNS = runs
    simulation = simulate_all(N= SIMRUNS, dataframe=df, model=model)
    simulationtime = round(time() - start, 2)/60

    print(f"{SIMRUNS} simulations took {simulationtime} minutes")

    
    return simulation


if __name__ == '__main__':
    main()