import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import importlib

# 1. INDISPENSABLE : Dire à Python de chercher dans le dossier parent (racine)
sys.path.append(os.path.abspath(".."))

# 2. Maintenant l'import fonctionne immédiatement
from utils_files import utils_credit
from utils_files import utils_ai4i
from utils_files import utils_partie3
from utils_files import utils_partie5

print("Tous les modules ont été importés avec succès !")