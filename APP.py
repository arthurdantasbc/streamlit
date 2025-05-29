# %%
import streamlit as st
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

st.sidebar.title("Menu")

# ==== INTERFACE PRINCIPAL ====
st.header("Modelo")
dataset_opcao = st.selectbox("**Escolha um dataset**", [" - ", "CWRU", "JNU", "EEG"])

st.divider()

# ==== FEATURES A SEREM EXTRAÍDAS ====
st.write("**Escolha as features a serem extraídas**")
features = [
    "Média", "Variância", "Desvio-padrão", "RMS", "Kurtosis",
    "Peak to peak", "Max Amplitude", "Min Amplitude", "Skewness",
    "CrestFactor", "Mediana", "Energia", "Entropia"
]
selected_features = [f for f in features if st.checkbox(f)]

st.divider()

# ==== MÉTODO DE CODIFICAÇÃO ====
encoding_method = st.selectbox("**Escolha um método de codificação**", [
    " - ", "Angle encoding", "Amplitude encoding",
    "ZFeaturemap", "XFeaturemap", "YFeaturemap", "ZZFeaturemap"
])

st.divider()

# ==== PQC: ROTAÇÕES DE EULER ====
st.write("**PQC: escolha a quantidade de rotações de Euler**")
rot1 = st.checkbox("1")
rot2 = st.checkbox("2")
rot3 = st.checkbox("3")

if rot1 and not rot2 and not rot3:
    eixos = [st.selectbox("**Escolha o eixo da rotação**", [" - ", "X", "Y", "Z"])]
elif rot2 and not rot1 and not rot3:
    eixos = [
        st.selectbox("**Escolha o eixo da primeira rotação**", [" - ", "X", "Y", "Z"]),
        st.selectbox("**Escolha o eixo da segunda rotação**", [" - ", "X", "Y", "Z"])
    ]
elif rot3 and not rot1 and not rot2:
    eixos = [
        st.selectbox("**Escolha o eixo da primeira rotação**", [" - ", "X", "Y", "Z"]),
        st.selectbox("**Escolha o eixo da segunda rotação**", [" - ", "X", "Y", "Z"]),
        st.selectbox("**Escolha o eixo da terceira rotação**", [" - ", "X", "Y", "Z"])
    ]
else:
    eixos = []

st.divider()

# ==== ESCOLHA DA PORTA DE EMARANHAMENTO ====
if encoding_method in ["Angle encoding", "Amplitude encoding", "ZFeaturemap", "XFeaturemap", "YFeaturemap", "ZZFeaturemap"]:
    st.selectbox("**PQC: escolha a porta de emaranhamento**", [" - ", "CZ", "iSWAP", "Real Amplitudes", "QCNN"])
    st.divider()


# ==== HIPERPARÂMETROS ====
paciencia = st.number_input('Insira o valor da paciência:', min_value=0, max_value=400, value=0, step=1)
epocas = st.number_input('Insira o número de épocas:', min_value=1, max_value=500, value=1, step=1)

st.divider()


import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew

# ==== FUNÇÕES DE EXTRAÇÃO DE FEATURES ====

def extrair_features_amostra(amostra):
    features_calculadas = {}
    features_calculadas["Média"] = np.mean(amostra)
    features_calculadas["Variância"] = np.var(amostra)
    features_calculadas["Desvio-padrão"] = np.std(amostra)
    features_calculadas["RMS"] = np.sqrt(np.mean(np.square(amostra)))
    features_calculadas["Kurtosis"] = kurtosis(amostra)
    features_calculadas["Peak to peak"] = np.ptp(amostra)
    features_calculadas["Max Amplitude"] = np.max(amostra)
    features_calculadas["Min Amplitude"] = np.min(amostra)
    features_calculadas["Skewness"] = skew(amostra)
    features_calculadas["CrestFactor"] = np.max(np.abs(amostra)) / (np.sqrt(np.mean(np.square(amostra))) + 1e-10)
    features_calculadas["Mediana"] = np.median(amostra)
    features_calculadas["Energia"] = np.sum(amostra ** 2)
    prob, _ = np.histogram(amostra, bins=30, density=True)
    prob = prob[prob > 0]
    features_calculadas["Entropia"] = -np.sum(prob * np.log(prob))
    return features_calculadas

def extrair_features_dataset(dataset_bruto, selected_features):
    lista_dicts = []
    for amostra in dataset_bruto:
        f = extrair_features_amostra(amostra)
        f_sel = {key: f[key] for key in selected_features}
        lista_dicts.append(f_sel)
    return pd.DataFrame(lista_dicts)

# ==== FUNÇÃO PARA CARREGAR DADOS BRUTOS (simulação) ====

def carregar_dados_brutos(nome):
    # Para usar bases reais, substitua essa parte pelo carregamento real.
    if nome == "CWRU":
        # Simulando sinais: 100 amostras, cada com 1024 pontos
        return np.random.randn(100, 1024), np.random.randint(0, 2, 100)
    elif nome == "JNU":
        return np.random.randn(80, 1024), np.random.randint(0, 3, 80)
    elif nome == "EEG":
        return np.random.randn(120, 256), np.random.randint(0, 2, 120)
    else:
        return None, None

def selecionar_features(X, features, selecionadas):
    indices = [features.index(f) for f in selecionadas]
    return X[:, indices]

# --- CIRCUITOS DE ENCODING ---

def angle_encoding(x, wires, eixos):
    # Aplica rotações baseadas nos eixos fornecidos
    for i, wire in enumerate(wires):
        for eixo in eixos:
            if eixo == "X":
                qml.RX(x[i], wires=wire)
            elif eixo == "Y":
                qml.RY(x[i], wires=wire)
            elif eixo == "Z":
                qml.RZ(x[i], wires=wire)

def amplitude_encoding(x, wires):
    qml.AmplitudeEmbedding(features=x, wires=wires, normalize=True)

def z_featuremap(x, wires):
    for i, wire in enumerate(wires):
        qml.RZ(x[i], wires=wire)

def x_featuremap(x, wires):
    for i, wire in enumerate(wires):
        qml.RX(x[i], wires=wire)

def y_featuremap(x, wires):
    for i, wire in enumerate(wires):
        qml.RY(x[i], wires=wire)

def zz_featuremap(x, wires):
    # Exemplo simples: aplicar RZ e depois CNOT para emaranhamento
    for i, wire in enumerate(wires):
        qml.RZ(x[i], wires=wire)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

# --- CRIAÇÃO DO CIRCUITO PQC ---

def criar_circuito(encoding_method, eixos, entanglement_gate, n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        # Encoding
        if encoding_method == "Angle encoding":
            angle_encoding(x, wires=range(n_qubits), eixos=eixos)
        elif encoding_method == "Amplitude encoding":
            amplitude_encoding(x, wires=range(n_qubits))
        elif encoding_method == "ZFeaturemap":
            z_featuremap(x, wires=range(n_qubits))
        elif encoding_method == "XFeaturemap":
            x_featuremap(x, wires=range(n_qubits))
        elif encoding_method == "YFeaturemap":
            y_featuremap(x, wires=range(n_qubits))
        elif encoding_method == "ZZFeaturemap":
            zz_featuremap(x, wires=range(n_qubits))
        else:
            pass  # Nenhuma codificação
        
        # Camada parametrizada - camada simples com RX, RY, RZ com pesos
        for i in range(n_qubits):
            qml.RX(weights[i, 0], wires=i)
            qml.RY(weights[i, 1], wires=i)
            qml.RZ(weights[i, 2], wires=i)

        # Emaranhamento (exemplo simples)
        if entanglement_gate == "CZ":
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i+1])
        elif entanglement_gate == "iSWAP":
            for i in range(n_qubits - 1):
                qml.ISWAP(wires=[i, i+1])
        elif entanglement_gate == "Real Amplitudes":
            qml.templates.layers.RealAmplitudes(weights, wires=range(n_qubits))
        elif entanglement_gate == "QCNN":
            # Coloque aqui o seu template QCNN se quiser
            pass
        
        # Medição
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit

# --- EXECUÇÃO DO MODELO ---

def executar_teste():
    st.success("Execução iniciada!")

    if dataset_opcao == " - ":
        st.error("Por favor, selecione um dataset.")
        return
    if len(selected_features) == 0:
        st.error("Por favor, selecione ao menos uma feature.")
        return
    if encoding_method == " - ":
        st.error("Por favor, selecione um método de codificação.")
        return
    if len(eixos) == 0 and encoding_method == "Angle encoding":
        st.error("Por favor, selecione os eixos das rotações para Angle encoding.")
        return

    # Carrega dados
    X, y = carregar_dados_brutos(dataset_opcao)
    if X is None or y is None:
        st.error("Erro ao carregar o dataset.")
        return
    
    # Seleciona features
    X_sel = selecionar_features(X, features, selected_features)
    
    # Pré-processa (normalização)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    # Define número de qubits como o número de features selecionadas
    n_qubits = X_scaled.shape[1]
    
    # Pesos aleatórios para camada parametrizada (exemplo)
    weights = np.random.uniform(low=0, high=2 * np.pi, size=(n_qubits, 3), requires_grad=True)

    # Porta de emaranhamento escolhida (default CZ)
    entanglement_gate = "CZ"
    # Aqui você pode melhorar para pegar do selectbox e armazenar numa variável, exemplo:
    # entanglement_gate = st.session_state.get("entanglement_gate", "CZ")

    # Cria circuito
    circuit = criar_circuito(encoding_method, eixos, entanglement_gate, n_qubits)

    # Executa circuito para todo dataset (exemplo: só execução direta, sem treino)
    resultados = np.array([circuit(x, weights) for x in X_scaled])

    # Como exemplo simples, usa saída quântica para treino SVM
    X_features = resultados
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"**Acurácia do modelo: {acc:.8f}**")




if st.button("Executar modelo"):
    executar_teste()



# %%
