import csv

def convert_txt_to_csv(input_txt_file, output_csv_file):
    """
    Converteix un fitxer .txt amb els resultats (arquitectura, pèrdua de validació i correlació) en un fitxer .csv.

    :param input_txt_file: Ruta del fitxer de text d’entrada amb els resultats dels experiments.
    :param output_csv_file: Ruta del fitxer .csv de sortida on es guardaran els resultats estructurats.
    :return: None.
    """
    with open(input_txt_file, 'r') as txt_file, open(output_csv_file, 'w', newline='') as csv_file:
        # Defineix els noms de les columnes (sense latent_dim)
        fieldnames = ["architecture", "val_loss", "corr_flat"]
        writer = csv.writer(csv_file)

        # Escriu la capçalera
        writer.writerow(fieldnames)

        # Inicialitza les variables abans de començar el bucle
        architecture = None
        val_loss = None
        corr_flat = None

        for line in txt_file:
            line = line.strip()

            if not line or line == "--------------------------":
                # Si la línia està buida o és una separació, ignora-la
                continue

            if line.startswith("Arquitectura:"):
                architecture = line.split(":")[1].strip()

            elif line.startswith("val_Loss:"):
                val_loss = line.split(":")[1].strip().replace('.', ',')

            elif line.startswith("corr_flat:"):
                corr_flat = line.split(":")[1].strip().replace('.', ',')

                # Quan tenim totes les dades, escrivim una línia al CSV
                if architecture and val_loss and corr_flat:
                    writer.writerow([architecture, val_loss, corr_flat])

                    # Reiniciem les variables per a la següent entrada
                    architecture = None
                    val_loss = None
                    corr_flat = None

# Converteix els resultats a CSV
convert_txt_to_csv(input_txt_file="MCI_latent_dim_15.txt", output_csv_file="MCI_latent_dim_15.csv")
