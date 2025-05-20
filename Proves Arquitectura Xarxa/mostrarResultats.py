import numpy as np
import os

# Funció per llegir un fitxer i extreure els resultats
def read_results_file(file_path):
    """
    Llegeix un fitxer de resultats línia per línia i converteix cada línia en un diccionari de Python.

    :param file_path: Ruta al fitxer de resultats.
    :return: Llista de diccionaris amb els resultats.
    """
    results = []
    with open(file_path, "r") as file:
        for line in file:
            # Converteix la línia de text en un diccionari
            result = eval(line.strip())
            results.append(result)
        return results

# Llegeix tots els fitxers i desa els resultats combinats en un nou fitxer
def combine_results_to_file(start_step, end_step, output_file, file_prefix="results_MCI_step_"):
    """
    Combina els resultats de múltiples fitxers en un únic fitxer de sortida.

    :param start_step: Valor inicial del pas.
    :param end_step: Valor final del pas.
    :param output_file: Nom del fitxer de sortida combinat.
    :param file_prefix: Prefix dels fitxers d'entrada.
    :return: None.
    """
    all_results = []
    for step in range(start_step, end_step + 1):
        file_name = f"{file_prefix}{step}.txt"
        if os.path.exists(file_name):
            print(f"Llegint el fitxer {file_name}")
            results = read_results_file(file_name)
            all_results.extend(results)
        else:
            print(f"Fitxer {file_name} no trobat, saltant.")

    # Desa els resultats combinats en el fitxer nou
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(str(result) + "\n")

# Funció per trobar els 5 millors resultats en funció de 'corr_flat'
def find_top5_results(filepath):
    """
    Troba els 5 millors resultats dins d’un fitxer segons el valor de 'corr_flat'.

    :param filepath: Ruta del fitxer amb els resultats combinats.
    :return: None.
    """
    # Llegeix el fitxer i processa cada línia
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Substitueix 'np.float64(...)' per només el valor numèric
    cleaned_lines = [line.replace('np.float64(', '').replace(')', '').strip() for line in lines]

    # Converteix cada línia en un diccionari
    results = [eval(line, {"np": np}) for line in cleaned_lines]

    # Ordena per 'corr_flat' i selecciona els 5 millors
    top_5 = sorted(results, key=lambda x: x['corr_flat'], reverse=True)[:5]

    # Guarda els millors resultats en un fitxer nou
    with open('millorsResultats_Steps_MCI.txt', 'w') as f:
        for result in top_5:
            f.write(str(result) + '\n')

# Executa les funcions
combine_results_to_file(start_step=70, end_step=100, output_file="resultatsTotalsSteps_MCI.txt")
find_top5_results(filepath="resultatsTotalsSteps_MCI.txt")
