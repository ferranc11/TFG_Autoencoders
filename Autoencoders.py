import random
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time # importar el mòdul time per mesurar el temps
import p_values as p_values

from scipy import stats
from skimage.metrics import structural_similarity as ssim
from torch import optim
# from baseDataLoader import DataLoader
# import Schaefer2018
from ADNI_B import ADNI_B

# ============================================================================
# %% Funció per carregar i pre-processar les dades
def load_and_preprocess_data(SchaeferSize=400,split_ratio=0.9,group=None):
    """
    Carrega i pre-processa les dades dels datasets ADNI3 (AD, HC, MCI) amb la parcel·lació Schaefer 400.

    :param SchaeferSize: Nombre de parcel·les.
    :param split_ratio: Proporció de dades pel conjunt d'entrenament.
    :param group: Grup del dataset (AD, HC, MCI).
    :return: Conjunt d'entrenament, conjunt de validació i el nombre de mostres temporals per subjecte.
    """

    # Inicialitza el carregador de dades ADNI_B
    data_loader = ADNI_B(ADNI_version='matching', SchaeferSize=SchaeferSize)

    data = data_loader.get_fullGroup_data(group)

    # Converteix el conjunt de dades a una matriu numpy
    timeseries = np.array([data[(i, group)]['timeseries'].T for i in range(len(data))])

    # Obté el nombre de mostres temporals per subjecte
    t_sub = timeseries[0].shape[0]
    print(f"Nombre de mostres temporals per subjecte ({group}): {t_sub}")


    # Divideix les dades en entrenament i validació
    n_subjects = len(timeseries)
    split_idx = int(n_subjects * split_ratio) # 90% entrenament i 10% validació

    # Combina les dades dins els conjunts d'entrenament i validació
    training_set = np.concatenate(timeseries[:split_idx])
    validation_set = np.concatenate(timeseries[split_idx:])

    # Normalitza les dades
    training_set = stats.zscore(training_set, axis=1)
    validation_set = stats.zscore(validation_set, axis=1)

    return training_set, validation_set, t_sub, timeseries


# %% Llavor aleatòria i configuració del dispositiu
def set_seed(seed=None, seed_torch=True):
    """
    Configura una llavor aleatòria.

    :param  (int): Si `None`, es genera una llavor aleatòria.
    :param seed_torch (bool): Si és `True`, configura la llavor pels tensors de PyTorch.
    :return: int -> La llavor generada.
    """
    if seed is None:
        seed = np.random.randint(0, 2 ** 31-1)  # Genera una llavor aleatòria en un rang segur
    random.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Llavor aleatòria configurada: {seed}')

    return seed


def seed_worker(worker_id):
    """
    Configura la llavor pels subprocessos del 'DataLoader'.

    :param worker_id (int): ID del subprocess.
    :return: None
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_device():
    """
    Configura el dispositiu (CPU o GPU) per a l'entrenament i mostra un missatge d'advertència si GPU no està disponible.

    :return: str -> El dispositiu configurat ('cuda' o 'cpu').
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("ADVERTÈNCIA: Per millorar el rendiment, habilita la GPU si és possible.")
    else:
        print("GPU habilitada per a aquest projecte.")

    return device

# ============================================================================
def create_dataloaders(training_data, validation_data, batch_size, num_workers=0):
    """
    Crea 'DataLoader' pels conjunts d'entrenament i validació.

    :param training_data (array): Conjunt d'entrenament.
    :param validation_data (array): Conjunt de validació.
    :param batch_size (int): Mida del batch pel carregador.
    :param num_workers (int): Nombre de subprocessos per carregar les dades. Per defecte és 0.
    :return: tuple -> '(train_loader, val_loader)' DataLoaders configurats.
    """
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)

    val_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)

    return train_loader, val_loader

# ============================================================================
# %% Definició de la xarxa neuronal

class AutoencoderNet(nn.Module):
    """
    Autoencoder amb arquitectura lineal i Batch Normalization.
    """

    def __init__(self, latent_dim):
        """
        Inicialitza les capes de la xarxa.

        :param latent_dim (int): Dimensió de l'espai latent.
        """
        super(AutoencoderNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=400, out_features=271),
            nn.BatchNorm1d(num_features=271),
            nn.ReLU(),
            nn.Linear(in_features=271, out_features=142),
            nn.BatchNorm1d(num_features=142),
            nn.ReLU(),
            # nn.Linear(in_features=144, out_features=16),
            # nn.BatchNorm1d(num_features=16),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=32),
            # nn.BatchNorm1d(num_features=32),
            # nn.ReLU(),
            # nn.Linear(in_features=32, out_features=16),
            # nn.BatchNorm1d(num_features=35),
            # nn.ReLU(),
            nn.Linear(in_features=142, out_features=latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=142),
            nn.BatchNorm1d(num_features=142),
            nn.ReLU(),
            nn.Linear(in_features=142, out_features=271),
            nn.BatchNorm1d(num_features=271),
            # nn.ReLU(),
            # nn.Linear(in_features=144, out_features=272),
            # nn.BatchNorm1d(num_features=272),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=400)
            # nn.BatchNorm1d(num_features=128),
            # nn.ReLU(),
            # nn.Linear(in_features=128, out_features=256),
            # nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=271, out_features=400)
        )

    def forward(self, x):
        """
        Executa la propagació endavant completa de l'autoencoder.

        :param x (torch.Tensor): Entrada de la xarxa.
        :return: torch.Tensor -> sortida reconstruïda.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def encode(self,x):
        """
        Codifica l'entrada a l'espai latent.

        :param x (torch.Tensor): Entrada de la xarxa.
        :return: torch.Tensor -> Representació en l'espai latent.
        """
        return self.encoder(x)

    def decode(self, x):
        """
        Decodifica des de l'espai latent a l'espai original.

        :param x (torch.Tensor): Representació latent.
        :return: torch.Tensor -> Reconstrucció de l'entrada.
        """
        return self.decoder(x)


# ============================================================================
# %% Funció d'entrenament
def train(
    model, device, train_loader, validation_loader, epochs, patience, learning_rate, lambd
):
    """
    Entrena el model d'autoencoder amb optimització de pèrdua i parada anticipada.

    :param model (nn.Module): Model d'autoencoder.
    :param device (str): Dispositiu a utilitzar ('cuda' o 'cpu').
    :param train_loader (DataLoader): Loader del conjunt d'entrenament.
    :param validation_loader (DataLoader): Loader del conjunt de validació.
    :param epochs (int): Nombre d'èpoques d'entrenament
    :param patience (int): Nombre d'èpoques sense millora abans de parar.
    :param learnin_rate (float): Taxa d'aprenentatge per a l'optimitzador.
    :param lambd (float): Pes per a la regularització.
    :return: tuple -> Pèrdua d'entrenament, pèrdua de validació, millor model i època òptima.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambd)

    train_losses, val_losses = [], []
    best_loss = float('inf')
    wait, best_epoch = 0, 0
    best_model = None

    print("Iniciant entrenament...")

    for epoch in range(epochs):
        # Entrenament
        model.train()
        train_loss_epoch = 0.0

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device, dtype=torch.float)
            output = model(data)

            loss = criterion(output, data) # Pèrdua principal
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)

        # Validació
        model.eval()
        val_loss_epoch = 0.0

        with torch.no_grad():
            for data in validation_loader:
                data = data.to(device, dtype=torch.float)
                output = model(data)

                loss = criterion(output, data) # Pèrdua principal
                val_loss_epoch += loss.item()

        val_loss_epoch /= len(validation_loader)
        val_losses.append(val_loss_epoch)

        # Comprovació de millora
        if val_loss_epoch < best_loss:
            best_loss = val_loss_epoch
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            wait = 0
            print (f"[{epoch + 1}/{epochs}] Millora de pèrdua de validació: {val_loss_epoch:.4f}")
        else:
            wait += 1
            if wait > patience:
                print (f"Parada anticipada després de {epoch + 1} èpoques. Millor època: {best_epoch + 1}.")
                break

    print ("Entrenament completat.")

    return np.array(train_losses), np.array(val_losses), best_model, best_epoch

# ============================================================================
# ============================================================================

def process(group):
    # ============================================================================
    model_filename = f"model_{group}.pth"

    # Carrega i prepara les dades
    training_set, validation_set, t_sub, timeseries = load_and_preprocess_data(SchaeferSize=400, split_ratio=0.9, group=group)

    # Crea els DataLoaders
    batch_size = 1080
    train_loader, val_loader = create_dataloaders(training_set, validation_set, batch_size=batch_size)

    # ============================================================================
    # %% Hiperparàmetres i resultats finals

    # Definició dels hiperparàmetres
    learning_rate = 0.0001
    epochs = 500
    patience = 50
    SEED = 5
    latent_dim = 15
    lambd = 0.0

    # Configuració de llavor i dispositiu
    set_seed(seed=SEED)
    device = set_device()

    # Inicialització del model
    model = AutoencoderNet(latent_dim).to(device)

    if os.path.exists(model_filename):
        # Si el model ja existeix, carregar-lo
        print(f"Carregant model entrenat des de {model_filename}...")
        model.load_state_dict(torch.load(model_filename, map_location=device))
        best_model = model
    else:
        # Si el model no existeix, entrenar-lo
        print("No s'ha trobat un model entrenat. Iniciant entrenament...")
        train_losses, val_losses, best_model, best_epoch = train(
            model, device, train_loader, val_loader, epochs=epochs, patience=patience, learning_rate=learning_rate,
            lambd=lambd
        )

        # Guardar el model entrenat
        torch.save(best_model.state_dict(), model_filename)
        print(f"Model entrenat guardat a {model_filename}")

    # Cronometrar el temps d'entrenament
    start_time = time.time()  # Comença a comptar

    # # Entrenament del model
    # train_losses, val_losses, best_model, best_epoch = train(
    #     model, device, train_loader, val_loader, epochs=epochs, patience=patience, learning_rate=learning_rate, lambd=lambd
    # )
    #
    # end_time = time.time()  # Atura el comptador
    #
    # # Temps total d'entrenament
    # training_time = end_time - start_time
    # print(f"Temps d'entrenament complet: {training_time:.2f} segons")
    #
    # print (f"Entrenament completat. Millor època: {best_epoch + 1}.")

    # Extracció de representació latent
    latent_train = best_model.encoder(
        torch.Tensor(training_set).to(device, dtype=torch.float)
    ).detach().cpu()

    latent_train = np.array(latent_train) # Convertir a numpy array
    latent_per_subject = np.array([
        latent_train[t_sub * i:t_sub * (i+1), :]
        for i in range(len(latent_train) // t_sub)
    ])

    # Exportació de la representació latent
    output_file = f'tseries_ADNI3_{group}_matching_sch400_QC.txt'
    np.savetxt(output_file, latent_train)
    print(f"Representació latent exportada a {output_file}.")

    # Generació de sortida reconstruïda
    output_train = best_model(
        torch.Tensor(training_set).to(device, dtype=torch.float)
    ).detach().cpu().numpy()

    return output_train, best_model, training_set, timeseries

# ============================================================================
def plot(output_train, training_set, t_sub, group):
    # Visualització de la reconstrucció
    # plt.figure(figsize=(10, 15))
    # plt.suptitle("Latent Decoded vs Original Signals - " f'{group}')
    # for i in range(10):
    #     plt.subplot(10, 1, i + 1)
    #     plt.plot(output_train[:t_sub, i], label='Reconstructed', alpha=0.7)
    #     plt.plot(training_set[:t_sub, i], label='Original', alpha=0.7)
    #     plt.legend()
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    #
    # # Anàlisi de correlacions
    # plt.figure(figsize=(8, 8))
    # plt.title("Correlació del conjunt original")
    # plt.imshow(np.corrcoef(training_set[:t_sub, :].T), cmap='viridis')
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure(figsize=(8, 8))
    # plt.title("Correlació del conjunt reconstruït")
    # plt.imshow(np.corrcoef(output_train[:t_sub, :].T), cmap='viridis')
    # plt.colorbar()
    # plt.show()

    # Comparació de correlacions
    original_corr = np.corrcoef(training_set[:t_sub, :])
    reconstructed_corr = np.corrcoef(output_train[:t_sub, :])

    # SSIM i correlació de matrius aplanades
    ssim_value = ssim(original_corr, reconstructed_corr, data_range=original_corr.max() - original_corr.min())
    corr_flat = np.corrcoef(original_corr.flatten(), reconstructed_corr.flatten()) [0, 1]

    print(f"SSIM entre matrius de correlació: {ssim_value:.4f}.")
    print(f"Coeficient de correlació entre matrius de correlació aplanades: {corr_flat:.4f}.")


# ============================================================================
# ============================================================================
def combine_encoders_decoders(encoder, decoder, input, device):
    """
    Combina l'encoder d'un model amb el decoder d'un altre i reconstrueix les dades.

    :param encoder (AutoencoderNet): Model de l'encoder.
    :param decoder (AutoencoderNet): Model del decoder.
    :param input (np.array): Dades d'entrada a l'encoder.
    :param device (str): Dispositiu d'execució.
    :return: np.array -> Dades reconstruïdes amb encoder i decoder de models diferents.
    """
    encoder.eval()
    decoder.eval()
    input = np.concatenate(input)

    # Convertim les dades a tensor de PyTorch
    input_tensor = torch.Tensor(input).to(device, dtype=torch.float)

    # Passar les dades per l'encoder i després pel decoder
    with torch.no_grad():
        latent = encoder.encode(input_tensor)
        reconstructed_output = decoder.decode(latent)

    return reconstructed_output.cpu().numpy()

# ============================================================================
def calculate_fc(data_set, reshape):
    """

    :param data_set:
    :return:
    """
    # Càlcul FC
    FC_subject = []

    if reshape == 'S':
        t_sub = 197
        num_regions = 400
        num_subjects = data_set.shape[0] // t_sub
        data_set = data_set.reshape(num_subjects, t_sub, num_regions)

    else:
        num_subjects = len(data_set)


    for i in range(num_subjects):
        FC_corr = np.corrcoef(data_set[i].T)
        FC_subject.append(FC_corr)

    return FC_subject

# ============================================================================
def compute_similarity(fc_reconstructed, fc_normal):
    """
    Calcula la similitud entre totes les matrius FC del conjunt reconstruït i les d'un altre grup.

    :param fc_reconstructed: Llista de matrius FC del grup reconstruït.
    :param fc_normal: Llista de matrius FC d'un grup real.
    :return: Llista amb els coeficients de correlació de cada comparació.
    """
    similarities = []

    for reconstructed in fc_reconstructed:
        for normal in fc_normal:
            # Aplanem les matrius per calcular la correlació entre vectors
            fc1_flat = reconstructed.flatten()
            fc2_flat = normal.flatten()
            corr = np.corrcoef(fc1_flat, fc2_flat)[0, 1]
            similarities.append(corr)

    return similarities
# ============================================================================
# ============================================================================
def run():
    t_sub = 197
    # Entrenar models per cada grup
    output_AD, model_AD, training_AD, data_AD = process('AD')
    output_HC, model_HC, training_HC, data_HC = process('HC')
    output_MCI, model_MCI, training_MCI, data_MCI = process('MCI')

    # Mostrar resultats per cada grup
    plot(output_AD, training_AD, t_sub, 'AD')
    plot(output_HC, training_HC, t_sub, 'HC')
    plot(output_MCI, training_MCI, t_sub, 'MCI')

    # Combinar encoders i decoders de grups diferents
    print("Combinant encoders i decoders...")
    device = set_device()

    # Carregar els models entrenats
    model_AD.load_state_dict(torch.load('model_AD.pth', weights_only=True))
    model_HC.load_state_dict(torch.load('model_HC.pth', weights_only=True))
    model_MCI.load_state_dict(torch.load('model_MCI.pth', weights_only=True))

    # Combinació d'autoencoders
    reconstructed_HC_MCI = combine_encoders_decoders(model_HC, model_MCI, data_HC, device)
    reconstructed_MCI_AD = combine_encoders_decoders(model_MCI, model_AD, data_MCI, device)
    reconstructed_HC_AD = combine_encoders_decoders(model_HC, model_AD, data_HC, device)

    # Calcular les FC de cada grup
    FC_AD = calculate_fc(data_AD, 'N')
    FC_HC = calculate_fc(data_HC, 'N')
    FC_MCI = calculate_fc(data_MCI, 'N')

    # Calcular les FC recombinades
    FC_HC_MCI_SIM = calculate_fc(reconstructed_HC_MCI, 'S')
    FC_MCI_AD_SIM = calculate_fc(reconstructed_MCI_AD, 'S')
    FC_HC_AD_SIM = calculate_fc(reconstructed_HC_AD, 'S')

    # Calcular la similitud entre tots els parells possibles
    similarity_HC_to_MCI_vs_AD = compute_similarity(FC_HC_MCI_SIM, FC_AD)
    similarity_HC_to_MCI_vs_HC = compute_similarity(FC_HC_MCI_SIM, FC_HC)
    similarity_HC_to_MCI_vs_MCI = compute_similarity(FC_HC_MCI_SIM, FC_MCI)

    similarity_MCI_to_AD_vs_AD = compute_similarity(FC_MCI_AD_SIM, FC_AD)
    similarity_MCI_to_AD_vs_HC = compute_similarity(FC_MCI_AD_SIM, FC_HC)
    similarity_MCI_to_AD_vs_MCI = compute_similarity(FC_MCI_AD_SIM, FC_MCI)

    similarity_HC_to_AD_vs_AD = compute_similarity(FC_HC_AD_SIM, FC_AD)
    similarity_HC_to_AD_vs_HC = compute_similarity(FC_HC_AD_SIM, FC_HC)
    similarity_HC_to_AD_vs_MCI = compute_similarity(FC_HC_AD_SIM, FC_MCI)

# ============================================================================
    # Comparar p-valor
# ============================================================================
    # Definir les dades per a cada comparació
    res_HC_MCI = {
        'HC': similarity_HC_to_MCI_vs_HC,
        'MCI': similarity_HC_to_MCI_vs_MCI,
        'AD': similarity_HC_to_MCI_vs_AD
    }

    res_MCI_AD = {
        'HC': similarity_MCI_to_AD_vs_HC,
        'MCI': similarity_MCI_to_AD_vs_MCI,
        'AD': similarity_MCI_to_AD_vs_AD
    }

    res_HC_AD = {
        'HC': similarity_HC_to_AD_vs_HC,
        'MCI': similarity_HC_to_AD_vs_MCI,
        'AD': similarity_HC_to_AD_vs_AD
    }

    # Definir etiquetes per als gràfics
    labels = ['HC', 'MCI', 'AD']

    # Generar el primer gràfic
    p_values.plotComparisonAcrossLabels2(
        res_HC_MCI, columnLables=labels,
        graphLabel='Similitud HC → MCI amb els grups reals'
    )

    # Generar el segon gràfic
    p_values.plotComparisonAcrossLabels2(
        res_MCI_AD, columnLables=labels,
        graphLabel='Similitud MCI → AD amb els grups reals'
    )

    # Generar el tercer gràfic
    p_values.plotComparisonAcrossLabels2(
        res_HC_AD, columnLables=labels,
        graphLabel='Similitud HC → AD amb els grups reals'
    )

    print("FET")

if __name__ == '__main__':
    run()