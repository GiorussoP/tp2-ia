# Trabalho Prático II: Aprendizado de máquina
Esse repositório compara testa e compara aprendizado supervisionado (k-Nearest Neighbors) com aprendizado não-supervisionado (k-Means), em um dataset da NBA. O objetivo dos algoritmos é prever se a carreira do jogador dura pelo menos 5 anos, dados os atributos do dataset. 

[Link para o Relatório](https://www.overleaf.com/read/vczywfqfwdnp#6cef53)

---

## Como Executar

### 1. Pré-requisitos
Certifique-se de ter o **Python 3** instalado em sua máquina. Será necessário também o gerenciador de pacotes `pip`.

### 2. Clonar o Repositório
Abra o terminal e clone o projeto:
```bash
git clone https://github.com/GiorussoP/tp2-ia.git
cd tp2-ia
```

### 3. Instalar as Dependências

Para garantir que todas as bibliotecas necessárias para a manipulação de dados, algoritmos do Scikit-Learn e geração de gráficos estejam disponíveis, instale-as executando:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 4. Executar os Experimentos

Com as dependências instaladas, basta rodar o script principal que executará os testes do $k\text{NN}$, do $k\text{--Means}$, fará as comparações, imprimirá os resultados e salvará os gráficos na pasta `images/`:

```bash
python3 main.py
```
Caso não funcione, tente  apenas `python main.py`.
