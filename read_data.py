# -*- coding: utf-8 -*-
"""CD-2: Leitura padronizada dos dados (UCS Oil and Gas Well)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path="data/data.csv", test_size=0.25, random_state=42):
    """Le o arquivo CSV local, separa treino/teste e devolve o dicionario padronizado."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path.resolve()}")

    # O arquivo local ja tem o cabecalho na primeira linha (sem skiprows)
    X = pd.read_csv(path, header=0)
    X.dropna(inplace=True)

    target_names = ['UCS (Mpa)']
    variable_names = X.columns.drop(target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X[variable_names], X[target_names],
        test_size=test_size, random_state=random_state,
    )

    n_samples, n_features = X_train.shape

    dataset = {
        'task'          : 'regression',
        'name'          : 'UCS Oil and Gas Well',
        'feature_names' : np.array(variable_names),
        'target_names'  : target_names,
        'n_samples'     : n_samples,
        'n_features'    : n_features,
        'X_train'       : X_train.values,
        'y_train'       : y_train.values.T,
        'X_test'        : X_test.values,
        'y_test'        : y_test.values.T,
        'targets'       : target_names,
        'descriptions'  : 'None',
        'reference'     : "https://www.sciencedirect.com/science/article/pii/S1875510021000238",
    }

    return dataset


if __name__ == "__main__":
    import seaborn as sns
    import pylab as pl

    dataset = read_data("data/data.csv")
    target_names = dataset['target_names']
    variable_names = dataset['feature_names']

    X_train = dataset['X_train']
    y_train = dataset['y_train'].T

    y = pd.DataFrame(y_train, columns=target_names)
    for c in target_names:
        sns.kdeplot(x=c, data=y)
        pl.show()

    X = pd.DataFrame(X_train, columns=variable_names)
    df = pd.concat([X, y], axis=1)

    pl.figure(figsize=(9, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # np.bool foi removido do NumPy
    heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    pl.show()

    c = corr[target_names].T
    pl.figure(figsize=(9, 2))
    sns.heatmap(c, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    pl.show()

    c = corr[target_names].drop(target_names, axis=0).T
    pl.figure(figsize=(7, 2))
    sns.heatmap(c, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    pl.show()

    for c in df.columns:
        n = df[c].unique().shape[0]
        print(n, c)
        if n < 8:
            sns.countplot(x=c, data=df)
        else:
            sns.catplot(y=c, data=df, kind='box', aspect=0.6, notch=1)
        pl.show()