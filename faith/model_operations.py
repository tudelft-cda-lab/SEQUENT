from flexfringe import FlexFringe
import os
import pandas as pd

def train(file: str, output_file: str, output_format: str, model_ini_file: str):
    """
    Make a call to the fit function from the python wrapper of FlexFringe.
    """
    check_ff_binary()
    ff = FlexFringe(
        flexfringe_path= os.environ.get('FLEXFRINGE_PATH') + '/flexfringe',
        ini=model_ini_file,
        outputfile=output_file,
        output = output_format
    )

    ff.fit(file, output_file=output_file, output_format=output_format)


def predict(file: str, apta_file: str, predict_ini_file:str) -> pd.DataFrame:
    """
    Make a call to the predict function from the python wrapper of FlexFringe.
    """
    check_ff_binary()
    ff = FlexFringe(
        flexfringe_path= os.environ['FLEXFRINGE_PATH'] + '/flexfringe',
        ini=predict_ini_file,
    )

    return ff.predict(file, apta_file)

def check_ff_binary():
    """
    Check whether the FlexFringe binary is available. The FlexFringe binary is expected 
    to be available in the FLEXFRINGE_PATH environment variable.
    """
    if not os.path.exists(os.environ['FLEXFRINGE_PATH'] + '/flexfringe'):
        raise ValueError('The flexfringe binary could not be found. Please make sure that the FLEXFRINGE_PATH environment variable is set correctly.')