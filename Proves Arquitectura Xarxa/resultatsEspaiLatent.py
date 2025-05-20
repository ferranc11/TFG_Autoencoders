import os
import glob

# Carpeta on es troben els fitxers de resultats
resultats_folder = "Espai latent"

# Llistes de grups i latents
grups = ["AD", "HC", "MCI"]
latent_dims = [5, 8, 10, 12, 15]

# Diccionari per emmagatzemar els resultats separats per grup i espai latent
grouped_results = {grup: {latent: [] for latent in latent_dims} for grup in grups}

# Obté la llista de fitxers .txt dins la carpeta
fitxers = glob.glob(os.path.join(resultats_folder, "*.txt"))

# Processa cada fitxer de resultats
for fitxer in fitxers:
    for grup in grups:
        if f"results_{grup}" in fitxer:  # Verifica si el fitxer pertany a aquest grup
            with open(fitxer, "r") as f:
                contingut = f.read()

            # Divideix en blocs basats en separadors "--------------------------"
            blocs = contingut.split("--------------------------")

            for bloc in blocs:
                if "latent_dim" in bloc:
                    lines = bloc.strip().split("\n")
                    latent_dim = None
                    resultat = []

                    for line in lines:
                        resultat.append(line)
                        if "latent_dim:" in line:
                            latent_dim = int(line.split(":")[1].strip())

                    if latent_dim in latent_dims:
                        grouped_results[grup][latent_dim].append("\n".join(resultat) + "\n--------------------------\n")

# Escriu cada conjunt de resultats en un fitxer separat per grup i latència
for grup in grups:
    for latent_dim in latent_dims:
        output_filename = f"{grup}_latent_dim_{latent_dim}.txt"
        with open(output_filename, "w") as f:
            f.writelines(grouped_results[grup][latent_dim])

        print(f"Fitxer creat: {output_filename}")
