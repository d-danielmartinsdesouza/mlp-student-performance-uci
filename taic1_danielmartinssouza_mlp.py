# Nome: Daniel Martins de Souza - RA: pg407589
# Trabalho de Tópicos Avançados em Inteligência Computacional I

import time # Importado para medir o tempo de execução
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # Torna dados de entrada em 0 e 1
from sklearn.model_selection import train_test_split # Divide dados em treino e teste (sem vícios)

# Função de ativação: Sigmoidal - Código produzido em aula
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

print("Carregando a base de dados...")
# Lendo a base de dados de matemática
dataset_mat = pd.read_csv('student-mat.csv', sep=';')

# Cria uma cópia para não alterar os dados originais da base de dados
df = dataset_mat.copy()

# Remove as notas G1 e G2 para tornar a previsão um desafio mais útil 
# (recomendação da base, mencionado no resumo do artigo: https://archive.ics.uci.edu/dataset/320/student+performance)
df = df.drop(['G1', 'G2'], axis=1) # axis 1 indica que está removendo uma coluna

# Separa a nota final (G3) do restante das características, definindo uma variável para acessá-la
y = df['G3'].values

# Ajuste para uso da função sigmóide:
# Nota final >= 10 vira 1 (Aprovado), Nota < 10 vira 0 (Reprovado)
y_binario = np.where(y >= 10, 1, 0)
y_binario = y_binario.reshape(-1, 1) # Ajuste para uma matriz coluna

# Remove a coluna G3 das características de entrada
X_df = df.drop(['G3'], axis=1)

# Converte todas as colunas de texto em colunas numéricas (0 e 1) com pandas
X_df = pd.get_dummies(X_df, drop_first=True, dtype=int)
scaler = MinMaxScaler()
X_normalizado = scaler.fit_transform(X_df)

# Separa em dados de Treinamento (80%) e Teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split( # train_test_split é da biblioteca importada
    X_normalizado, y_binario, test_size=0.2, random_state=42
)

# --- FUNÇÃO DE TREINAMENTO E TESTE ---
# Envolvemos o seu código original de treino numa função para podermos chamá-la várias vezes no menu
def executar_rede(quantidade_oculta, taxaAprendizagem, momento, epocas):
    quantidade_entradas = X_treino.shape[1]
    quantidade_saidas = 1 # Apenas 0 (Reprovado) ou 1 (Aprovado)
    
    # Inicializa com Pesos Aleatórios (conforme feito em sala)
    np.random.seed(42) # Mantém pesos aleatórios iguais em todos os testes
    pesos0 = 2 * np.random.random((quantidade_entradas, quantidade_oculta)) - 1
    pesos1 = 2 * np.random.random((quantidade_oculta, quantidade_saidas)) - 1

    # Define as Entradas e Saídas do Treinamento
    camadaEntrada = X_treino
    saidas = y_treino

    print("\nIniciando o treinamento da rede MLP...")
    tempo_inicio = time.time() # Começa a cronometrar

    # Loop de Treinamento (Backpropagation)
    for j in range(epocas):
        # Passo à frente (Forward Propagation)
        somaSinapse0 = np.dot(camadaEntrada, pesos0)
        camadaOculta = sigmoid(somaSinapse0)

        somaSinapse1 = np.dot(camadaOculta, pesos1)
        camadaSaida = sigmoid(somaSinapse1)

        # Cálculo do Erro Absoluto Médio
        erroCamadaSaida = saidas - camadaSaida
        mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))

        # Passo para trás (Backward Propagation)
        derivadaSaida = sigmoidDerivada(camadaSaida)
        deltaSaida = erroCamadaSaida * derivadaSaida

        pesos1Transposta = pesos1.T
        deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
        deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

        camadaOcultaTransposta = camadaOculta.T
        pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
        pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

        camadaEntradaTransposta = camadaEntrada.T
        pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
        pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)

    tempo_fim = time.time() # Termina de cronometrar
    tempo_decorrido = tempo_fim - tempo_inicio

    acuracia_treino = (1.00 - mediaAbsoluta) * 100

    # Troca dos dados de treino para os dados de teste
    camadaEntradaTeste = X_teste
    somaSinapse0_teste = np.dot(camadaEntradaTeste, pesos0)
    camadaOculta_teste = sigmoid(somaSinapse0_teste)

    somaSinapse1_teste = np.dot(camadaOculta_teste, pesos1)
    camadaSaida_teste = sigmoid(somaSinapse1_teste)

    previsoes = np.where(camadaSaida_teste >= 0.5, 1, 0)

    # Calcula o sucesso (Generalização)
    acertos = np.sum(previsoes == y_teste)
    acuracia_teste = (acertos / len(y_teste)) * 100

    return mediaAbsoluta, acuracia_treino, acuracia_teste, tempo_decorrido

# MENU de testes
def menu():

    # Parâmetros Iniciais
    ocultos = 15
    eta = 0.3
    momento_alfa = 1.0
    epocas = 10000

    teste_anterior = None

    while True:
        print("\n" + "="*50)
        print("   PAINEL DE CONTROLE - REDE NEURAL MLP")
        print("="*50)
        print(" CONFIGURAÇÃO ATUAL:")
        print(f"  [1] Neurônios Ocultos  : {ocultos}")
        print(f"  [2] Taxa Aprend. (eta) : {eta:.2f}")
        print(f"  [3] Momento            : {momento_alfa:.2f}")
        print(f"  [4] Épocas de Treino   : {epocas}")
        print("-" * 50)
        print(" OPÇÕES:")
        print("  [R] Rodar o Treinamento")
        print("  [S] Sair")
        print("="*50)

        opcao = input("Escolha uma opção (1-4 para alterar, R para Rodar, S para Sair): ").strip().upper()

        if opcao == '1':
            acao = input("Deseja (A)umentar ou (D)iminuir os neurônios? ").strip().upper()
            if acao == 'A': ocultos += 5
            elif acao == 'D' and ocultos > 5: ocultos -= 5
            
        elif opcao == '2':
            acao = input("Deseja (A)umentar ou (D)iminuir a Taxa de Aprendizagem (0.1)? ").strip().upper()
            if acao == 'A': eta = round(eta + 0.1, 2)
            elif acao == 'D' and eta > 0.1: eta = round(eta - 0.1, 2)

        elif opcao == '3':
            acao = input("Deseja (A)umentar ou (D)iminuir o Momento (0.1)? ").strip().upper()
            if acao == 'A' and momento_alfa < 1.0: momento_alfa = round(momento_alfa + 0.1, 2)
            elif acao == 'D' and momento_alfa > 0.0: momento_alfa = round(momento_alfa - 0.1, 2)

        elif opcao == '4':
            acao = input("Deseja (A)umentar ou (D)iminuir as Épocas (em 1000)? ").strip().upper()
            if acao == 'A': epocas += 1000
            elif acao == 'D' and epocas > 1000: epocas -= 1000

        elif opcao == 'R':
            # Executa a rede com os parâmetros atuais
            erro, ac_treino, ac_teste, tempo = executar_rede(ocultos, eta, momento_alfa, epocas)
            
            print("\n--- RESULTADOS DA EXECUÇÃO ---")
            print(f"Tempo de Execução     : {tempo:.3f} segundos")
            print(f"Erro Final (EAM)      : {erro:.4f}")
            print(f"Acurácia (Treino)     : {ac_treino:.2f}%")
            print(f"Acurácia (Teste Real) : {ac_teste:.2f}%")

            # Comparação com o teste anterior
            if teste_anterior is not None:
                print("\n--- COMPARAÇÃO COM O TESTE ANTERIOR ---")
                
                # Tempo
                dif_tempo = tempo - teste_anterior['tempo']
                if dif_tempo < 0: print(f"Ficou MAIS RÁPIDA em {abs(dif_tempo):.2f} segundos.")
                else: print(f"Ficou MAIS LENTA em {dif_tempo:.2f} segundos.")

                # Aprendizado
                dif_treino = ac_treino - teste_anterior['treino']
                if dif_treino > 0: print(f"APRENDEU MAIS nos dados de treino (+{dif_treino:.2f}%).")
                elif dif_treino < 0: print(f"APRENDEU MENOS nos dados de treino ({dif_treino:.2f}%).")
                else: print("Aprendizado Manteve-se IGUAL.")

                # Generalização
                dif_teste = ac_teste - teste_anterior['teste']
                if dif_teste > 0: print(f"ACERTOU MAIS para dados novos (+{dif_teste:.2f}%).")
                elif dif_teste < 0: print(f"ERROU MAIS na generalização de dados novos ({dif_teste:.2f}%).")
                else: print("Generalização Manteve-se IGUAL.")
            
            # Salva o resultado atual
            teste_anterior = {'tempo': tempo, 'treino': ac_treino, 'teste': ac_teste}
            input("\nPressione ENTER para voltar ao menu...")

        elif opcao == 'S':
            print("\nEncerrando o programa!")
            break
        else:
            print("Opção inválida. Tente novamente.")

# Inicia o programa rodando o menu
if __name__ == "__main__":
    menu()