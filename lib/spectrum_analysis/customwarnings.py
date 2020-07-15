"""Customize warnings."""
import warnings


class ParameterWarning(UserWarning):
    """User defined Warning class."""


def custom_formatwarning(message, category, filename, lineno, line=None):
    """Format warnings."""
    # don't show line in warning
    return formatwarning_orig(message, category, filename, lineno, line='')


formatwarning_orig = warnings.formatwarning
warnings.formatwarning = custom_formatwarning  # change format of warning
