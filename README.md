# Autoencoders

Aquest repositori de GitHub recull el codi desenvolupat pel meu treball de fi de grau, titulat: **Simulació de l’evolució de l’Alzheimer mitjançant xarxes neuronals**, fet per Ferran Conchillo.

## Objectiu del projecte

L’objectiu del treball és analitzar i simular l’evolució de l’Alzheimer a través de xarxes neuronals basades en *autoencoders* i les seves variants, generant transformacions entre grups clínics (HC, MCI, AD) per simular possibles progressions de la malaltia.

## Requisits per a l'execució

Per replicar correctament aquest projecte cal:

- Col·locar els tres fitxers de dades a la mateixa carpeta on es troben els arxius .py del projecte per assegurar el correcte accés a les dades.

- Tenir instal·lades totes les llibreries especificades en la memòria del treball i llistades a l’inici de cada fitxer .py.

- Recomanable utilitzar un entorn virtual per mantenir un entorn de treball net.

## Fitxers externs (aportats pel tutor)

Els següents fitxers han estat extrets íntegrament del repositori de GitHub del meu tutor. S'encarreguen de la inicialització, càrrega i parcel·lació de les dades utilitzades en el projecte:

- **ADNI_B.py**

- **Schaefer2018.py**

- **baseDataLoader.py**

- **parcellation.py**

Aquest script s'utilitza per visualitzar i analitzar els resultats de les simulacions mitjançant gràfics de p-valors:

- **p_values.py**

## Fitxers propis

- **Autoencoders.py**: Entrenament i simulació mitjançant *autoencoders* simples. Conté les reconstruccions inicials, la distribució de l'espai latent i les simulacions generant els tres *deep fakes* neuronals entre grups.

- **VAE.py**: Versió amb *variational autoencoders*. Conté les reconstruccions inicials, la distribució de l'espai latent i les simulacions generant els tres *deep fakes* neuronals entre grups.

- **CAE.py**: Implementació amb *convolutional autoencoders*. Conté les reconstruccions inicials, la distribució de l'espai latent i les simulacions generant els tres *deep fakes* neuronals entre grups.
