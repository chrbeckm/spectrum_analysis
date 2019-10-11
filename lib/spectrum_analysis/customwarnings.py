import warnings
from warnings import warn

class ParameterWarning(UserWarning):
    pass

def custom_formatwarning(message, category, filename, lineno, line=None):
    return formatwarning_orig(message, category, filename, lineno, line='') #don't show line in warning

formatwarning_orig = warnings.formatwarning
warnings.formatwarning = custom_formatwarning #change format of warning
