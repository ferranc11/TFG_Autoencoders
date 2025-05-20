import random
import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time # importar el mòdul time per mesurar el temps
from scipy import stats
from scipy.stats import norm
from torch import optim

# from baseDataLoader import DataLoader
# import Schaefer2018
from ADNI_B import ADNI_B

# ============================================================================
# %% Funció per carregar i pre-processar les dades
def load_and_preprocess_data(SchaeferSize=400,split_ratio=0.8,group=None):
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

    return training_set, validation_set, t_sub


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
            nn.Linear(in_features=400, out_features=264),
            nn.BatchNorm1d(num_features=264),
            nn.ReLU(),
            nn.Linear(in_features=264, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            # nn.Linear(in_features=110, out_features=13),
            # nn.BatchNorm1d(num_features=13),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=32),
            # nn.BatchNorm1d(num_features=32),
            # nn.ReLU(),
            # nn.Linear(in_features=32, out_features=16),
            # nn.BatchNorm1d(num_features=35),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=264),
            nn.BatchNorm1d(num_features=264),
            # nn.ReLU(),
            # nn.Linear(in_features=255, out_features=271),
            # nn.BatchNorm1d(num_features=271),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=400)
            # nn.BatchNorm1d(num_features=128),
            # nn.ReLU(),
            # nn.Linear(in_features=128, out_features=256),
            # nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=264, out_features=400)
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
    """
    Carrega les dades de l'autoencoder, genera la representació latent i retorna la reconstrucció obtinguda.

    :param group (str): Conjunt de dades a processar.
    :return: tuple -> Sortida reconstruïda, dades d'entrenament, sèrie temporal i pèrdua de validació.
    """

    # Carrega i prepara les dades
    training_set, validation_set, t_sub = load_and_preprocess_data(SchaeferSize=400, split_ratio=0.9, group=group)

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
    latent_dim = 12
    lambd = 0.0

    # Configuració de llavor i dispositiu
    # set_seed(seed=SEED)
    device = set_device()

    # Inicialització del model
    model = AutoencoderNet(latent_dim).to(device)

    # Cronometrar el temps d'entrenament
    start_time = time.time()  # Comença a comptar

    # Entrenament del model
    train_losses, val_losses, best_model, best_epoch = train(
        model, device, train_loader, val_loader, epochs=epochs, patience=patience, learning_rate=learning_rate, lambd=lambd
    )

    end_time = time.time()  # Atura el comptador

    # Temps total d'entrenament
    training_time = end_time - start_time
    print(f"Temps d'entrenament complet: {training_time:.2f} segons")

    print (f"Entrenament completat. Millor època: {best_epoch + 1}.")

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

    return output_train, training_set, t_sub, val_losses


def plot_histogram(corr_flat_list, label):
    """
    Ploteja un histograma dels coeficients de correlació finals després de 30 iteracions, amb un ajust Gaussià.

    :param corr_flat_list (list): Llista de coeficients de correlació finals.
    :param label (str): Nom per a la llegenda de la configuració.
    """
    # Paràmetres de color i configuració
    color = 'green'

    # Ajust Gaussià als coeficients de correlació
    mu, sigma = norm.fit(corr_flat_list)

    # Generació de valors per a la corba Gaussiana ajustada
    x = np.linspace(min(corr_flat_list), max(corr_flat_list), 100)
    pdf = norm.pdf(x, mu, sigma)

    # Ploteig de l'histograma de correlacions
    plt.hist(corr_flat_list, bins=10, alpha=0.5, color=color, label=f"Config {label}", density=True)

    # Ploteig de la corba Gaussiana ajustada
    plt.plot(x, pdf, color=color, linestyle='dashed', linewidth=2, label=f"Fit {label} (μ={mu:.4f}, σ={sigma:.4f})")

    # Etiquetes i títol
    plt.xlabel("Coeficient de correlació (flat)")
    plt.ylabel("Densitat")
    plt.legend()
    plt.title(f"Distribució del coeficient de correlació (flat) amb ajust Gaussià ({label})")
    plt.grid(True)
    plt.show()


def run_experiments():
    """
    Executa múltiples reconstruccions per cada grup (AD, HC, MCI) i en mostra la distribució de correlacions.
    """
    groups = ['AD', 'HC', 'MCI']

    for group in groups:
        corr_flat_list = []

        for i in range(30):
            print(f"\nExecució {i + 1}/30 per al grup {group}...\n")
            output_train, training_set, t_sub, corr_flat = process(group)
            corr_flat_list.append(corr_flat)

        # Aplanar la llista de llistes en una llista plana
        corr_flat_list = np.concatenate(corr_flat_list).tolist()

        plot_histogram(corr_flat_list, group)

if __name__ == '__main__':
    run_experiments()
