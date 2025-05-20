import random
import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time # importar el mòdul time per mesurar el temps
from scipy import stats
from skimage.metrics import structural_similarity as ssim
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
            nn.Linear(in_features=400, out_features=64),
            # nn.BatchNorm1d(num_features=256),
            # nn.ReLU(),
            # nn.Linear(in_features=256, out_features=128),
            # nn.BatchNorm1d(num_features=128),
            # nn.ReLU(),
            # nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=16),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=400)
            # nn.BatchNorm1d(num_features=128),
            # nn.ReLU(),
            # nn.Linear(in_features=128, out_features=256),
            # nn.BatchNorm1d(num_features=256),
            # nn.ReLU(),
            # nn.Linear(in_features=256, out_features=400)
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
# Funció per a les proves sistemàtiques per a un sol grup
def systematic_experiments_for_group(
   group,
   step_size,
   latent_dims=[5, 8, 10, 12, 15],  # Llista d'espais latents a provar
   epochs=500,
   patience=50,
   learning_rate=0.0001,
   lambd=0.0,
   output_file="results.txt"
):
    """
    Executa diversos entrenaments per a un grup determinat, provant diferents dimensions de l'espai latent.

    :param group: Conjunt de dades a processar.
    :param step_size: Diferència de neurones entre capes.
    :param latent_dims: Llista de dimensions de l'espai latent a provar.
    :param epochs: Nombre d’èpoques.
    :param patience: Nombre d’èpoques sense millora abans d’aplicar early stopping.
    :param learning_rate: Rati d’aprenentatge de l’optimitzador.
    :param lambd: Valor del terme de regularització.
    :param output_file: Fitxer on s’emmagatzemen els resultats dels entrenaments.
    :return: None.
    """
    training_set, validation_set, t_sub = load_and_preprocess_data(SchaeferSize=400, split_ratio=0.9, group=group)
    batch_size = 1080
    train_loader, val_loader = create_dataloaders(training_set, validation_set, batch_size=batch_size)

    results = []

    for latent_dim in latent_dims:
        dims = []
        current_dim = 400
        while current_dim > latent_dim:
            dims.append(current_dim)
            current_dim -= step_size
        dims.append(latent_dim)  # Última capa a la dimensió latent

        print(f"Arquitectura: {dims}")

        # Defineix el model amb una arquitectura flexible
        class DynamicAutoencoder(nn.Module):
            """
            Autoencoder amb arquitectura lineal dinàmica en funció de la llista de dimensions.
            """
            def __init__(self, layers):
                """
                Inicialitza dinàmicament les capes de l'encoder i el decoder.

                :param layers: Llista d’enters que defineixen la mida de cada capa del model.
                """
                super(DynamicAutoencoder, self).__init__()
                encoder_layers = []
                for in_dim, out_dim in zip(layers[:-1], layers[1:]):
                    encoder_layers.append(nn.Linear(in_dim, out_dim))
                    encoder_layers.append(nn.BatchNorm1d(out_dim))
                    encoder_layers.append(nn.ReLU())
                self.encoder = nn.Sequential(*encoder_layers)

                decoder_layers = []
                for in_dim, out_dim in zip(layers[::-1][:-1], layers[::-1][1:]):
                    decoder_layers.append(nn.Linear(in_dim, out_dim))
                    decoder_layers.append(nn.BatchNorm1d(out_dim))
                    decoder_layers.append(nn.ReLU())
                decoder_layers.pop()  # Treure última ReLU
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                """
                Executa la propagació endavant de l'AE.

                :param x: Entrada de la xarxa.
                :return: Sortida reconstruïda.
                """
                return self.decoder(self.encoder(x))

        device = set_device()
        model = DynamicAutoencoder(dims).to(device)

        # Entrenament del model
        train_losses, val_losses, best_model, best_epoch = train(
            model, device, train_loader, val_loader, epochs=epochs, patience=patience, learning_rate=learning_rate, lambd=lambd
        )

        # Avaluació
        output_train = best_model(torch.Tensor(training_set).to(device, dtype=torch.float)).detach().cpu().numpy()
        original_corr = np.corrcoef(training_set[:t_sub, :])
        reconstructed_corr = np.corrcoef(output_train[:t_sub, :])
        corr_flat = np.corrcoef(original_corr.flatten(), reconstructed_corr.flatten())[0, 1]
        ssim_value = ssim(original_corr, reconstructed_corr, data_range=original_corr.max() - original_corr.min())

        result_str = (
            f"latent_dim: {latent_dim}\n"
            f"Arquitectura: {dims}\n"
            f"val_Loss: {min(val_losses):.4f}\n"
            f"corr_flat: {corr_flat:.4f}\n"
            f"ssim: {ssim_value:.4f}\n"
            "--------------------------\n"
        )
        results.append(result_str)

        print(f"Resultat: latent_dim={latent_dim}, Arquitectura={dims}, val_loss={min(val_losses):.4f}, corr={corr_flat:.4f}, ssim={ssim_value:.4f}")

    # Desa els resultats en un fitxer de text
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(results)

    print(f"Totes les proves completades per {group}. Resultats desats a {output_file}.")


# Funció per provar tots els grups amb passos incrementals
def systematic_experiments_all_groups():
    """
    Executa experiments sistemàtics per a tots els grups, provant diferents valors de pas (step_size)
    en la construcció de l'arquitectura de l'autoencoder.
    """
    groups = ["AD", "HC", "MCI"]
    initial_step = 120
    max_step = 180  # Increment fins a un límit
    for step_size in range(initial_step, max_step + 1):
        print(f"Iniciant proves amb pas {step_size}.")
        for group in groups:
            output_file = f"results_{group}_step_{step_size}.txt"
            print(f"Iniciant proves per al grup {group} amb pas {step_size}.")
            systematic_experiments_for_group(
                group=group,
                step_size=step_size,
                latent_dims=[5, 8, 10, 12, 15],
                epochs=500,
                patience=50,
                learning_rate=0.001,
                lambd=0.00,
                output_file=output_file
            )

if __name__ == '__main__':
    systematic_experiments_all_groups()
