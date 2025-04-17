import pandas as pd
from semopy import Model
from semopy.inspector import inspect


def run_sem(data: pd.DataFrame, model_desc: str):
    """
    Runs SEM using semopy.
    Returns the fitted Model and a dictionary of results.
    """
    model = Model(model_desc)
    model.fit(data)
    estimates = inspect(model)
    fit_stats = model.calc_stats()
    return model, estimates, fit_stats
