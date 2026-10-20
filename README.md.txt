========================================================================
INSTRUÇÕES DE EXECUÇÃO DO PROJETO - REDE NEURAL MLP
========================================================================
Disciplina : Tópicos Avançados em Inteligência Computacional I
Professor  : Dr. Rodrigo Calvo (UEM / DIN)
Aluno      : Daniel Martins de Souza (RA: pg407589)
Arquivo    : taic1_danielmartinssouza_mlp.py
Dataset    : student-mat.csv (Student Performance - UCI)
========================================================================

1. PRÉ-REQUISITOS E AMBIENTE
------------------------------------------------------------------------
- Linguagem Python 3.8 ou superior instalada.
- Editor de código recomendado: Visual Studio Code (VS Code).

Bibliotecas necessárias (execute no terminal caso ainda não possua):
  pip install numpy pandas scikit-learn


2. ESTRUTURA DOS ARQUIVOS
------------------------------------------------------------------------
Certifique-se de manter os arquivos na MESMA PASTA do projeto:
  ├── taic1_danielmartinssouza_mlp.py   (Código-fonte em Python)
  ├── student-mat.csv                   (Base de dados)
  └── readme.txt                        (Guia de execução)


3. OPERAÇÃO DO MENU E PARÂMETROS DE ENTRADA
------------------------------------------------------------------------
Ao ser iniciado, o script realiza o pré-processamento dos dados e abre
o menu interativo no terminal.

Parâmetros Iniciais Padrão (Baseline):
  • Neurônios na Camada Oculta : 15
  • Taxa de Aprendizagem (eta) : 0.30
  • Constante de Momento       : 1.00 (ou 0.90)
  • Épocas de Treinamento      : 10000

Opções de Entrada no Terminal:
  [1] Alterar Neurônios Ocultos  : Permite (A)umentar (+5) ou (D)iminuir (-5).
  [2] Alterar Taxa de Aprend.    : Permite (A)umentar (+0.1) ou (D)iminuir (-0.1).
  [3] Alterar Termo de Momento   : Permite (A)umentar (+0.1) ou (D)iminuir (-0.1).
  [4] Alterar Épocas de Treino   : Permite (A)umentar (+1000) ou (D)iminuir (-1000).
  [R] Rodar o Treinamento        : Executa a MLP com os parâmetros atuais e exibe
                                   a comparação com o teste anterior.
  [S] Sair                       : Encerra a execução do programa.


5. MÉTRICAS EXIBIDAS APÓS A EXECUÇÃO [R]
------------------------------------------------------------------------
• Tempo de Execução (em segundos)
• Erro Absoluto Médio Final (EAM)
• Acurácia de Treino (%)
• Acurácia de Teste / Generalização (%)
• Quadro comparativo indicando se a alteração tornou o treino mais 
  rápido/lento, com mais/menos aprendizado e melhor/pior generalização.
========================================================================