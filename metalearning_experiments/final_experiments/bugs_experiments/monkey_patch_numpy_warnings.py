
import numpy as np
import warnings

if not hasattr(np, "warnings"):
    class FakeWarnings:
        @staticmethod
        def catch_warnings():
            return warnings.catch_warnings()

        @staticmethod
        def filterwarnings(action, message="", category=Warning, module="", lineno=0, append=False):
            return warnings.filterwarnings(action, message, category, module, lineno, append=append)

    np.warnings = FakeWarnings
