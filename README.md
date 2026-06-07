# Classificacao de Objetos Astronomicos com Deep Learning

Este repositorio reune o codigo, o pipeline experimental e os resultados de um projeto de nivel superior voltado a classificacao automatica de objetos astronomicos em imagens. O sistema compara diferentes arquiteturas de redes neurais profundas para distinguir tres classes principais: `galaxy`, `quasar` e `star`.

O projeto foi estruturado como um fluxo completo de pesquisa aplicada: preparacao do conjunto de dados, treinamento de modelos, avaliacao quantitativa, agregacao de multiplas execucoes, geracao de matrizes de confusao, analise por ensembles e interpretabilidade visual com Grad-CAM.

## Objetivo

O objetivo central e investigar o desempenho de arquiteturas modernas de visao computacional na tarefa de classificacao de imagens astronomicas, avaliando tanto modelos convolucionais quanto modelos baseados em Transformers e arquiteturas recentes.

O projeto permite:

- treinar modelos individualmente ou em lote;
- comparar desempenho entre arquiteturas;
- executar multiplas rodadas com sementes diferentes;
- gerar metricas de classificacao;
- salvar historico de treinamento e graficos;
- produzir mapas de calor para interpretabilidade;
- avaliar ensembles por voto majoritario e media de probabilidades.

## Classes do Problema

O conjunto de dados esta organizado no padrao esperado pelo `torchvision.datasets.ImageFolder`, em que cada classe corresponde a uma pasta:

```text
data/
  train/
    galaxy/
    quasar/
    star/
  test/
    galaxy/
    quasar/
    star/
```

Tambem existe a pasta `data/images/`, utilizada como base de imagens antes da separacao em conjuntos de treino e teste.

## Arquiteturas Avaliadas

As arquiteturas sao centralizadas em `models/model_factory.py` e podem ser selecionadas pela linha de comando:

| Argumento | Arquitetura |
| --- | --- |
| `efficientnet` | EfficientNet |
| `resnet` | ResNet |
| `convnext` | ConvNeXt |
| `vit` | Vision Transformer |
| `vmamba` | Vision Mamba |
| `cnn_shallow` | CNN convolucional rasa |

O comando `--model all` executa o pipeline para todos os modelos registrados.

## Estrutura do Projeto

```text
.
+-- main.py                         # Ponto de entrada do treinamento, avaliacao e heatmaps
+-- requirement.txt                 # Dependencias Python do projeto
+-- data/                           # Imagens organizadas por classe
+-- models/                         # Implementacoes das arquiteturas avaliadas
+-- training/                       # Rotina de treinamento
+-- evaluation/                     # Avaliacao, ensembles e Grad-CAM
+-- shared/                         # Constantes, metricas e pipelines compartilhados
+-- scripts/                        # Scripts auxiliares
+-- utils/                          # Gerenciamento de resultados
+-- logs/                           # Registros de execucao
+-- Resultados/                     # Modelos, metricas, graficos e relatorios gerados
```

## Dependencias

O projeto utiliza Python com PyTorch e bibliotecas de apoio para visao computacional, metricas e visualizacao.

Principais dependencias:

- `torch`, `torchvision`, `torchaudio`
- `numpy`
- `matplotlib`
- `opencv-python`
- `pillow`
- `tqdm`
- `scikit-learn`
- `timm`
- `pandas`
- `h5py`
- `grad-cam`

Instalacao recomendada:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
```

Em Linux ou macOS, a ativacao do ambiente virtual muda para:

```bash
source .venv/bin/activate
```

## Preparacao do Dataset

O script `scripts/prepare_dataset.py` centraliza a preparacao do dataset compartilhado. Ele pode apenas materializar dados ja existentes ou realizar o download antes do preprocessamento.

Preparar a partir de dados locais:

```bash
python scripts/prepare_dataset.py
```

Baixar dados e preparar o dataset:

```bash
python scripts/prepare_dataset.py --download
```

Opcionalmente, e possivel informar uma raiz alternativa para o dataset:

```bash
python scripts/prepare_dataset.py --data-root caminho/para/dataset
```

As configuracoes padrao do conjunto astronomico ficam em `shared/dataset_defaults.py`, incluindo as classes `star`, `galaxy` e `quasar`.

## Execucao do Treinamento

Treinar um modelo especifico:

```bash
python main.py --model resnet
```

Executar todos os modelos:

```bash
python main.py --model all
```

Executar cinco rodadas por modelo:

```bash
python main.py --model all --runs 5
```

Executar um treinamento reduzido para teste rapido:

```bash
python main.py --model efficientnet --epochs 2 --max-train-images 200 --max-validation-images 60 --max-test-images 60
```

Parametros relevantes:

| Parametro | Descricao | Padrao |
| --- | --- | --- |
| `--mode` | Modo de execucao: `train` ou `heatmap` | `train` |
| `--model` | Modelo a treinar ou `all` | obrigatorio |
| `--train-dir` | Diretorio de treino | `data/train` |
| `--test-dir` | Diretorio de teste | `data/test` |
| `--batch-size` | Tamanho do lote | `16` |
| `--epochs` | Numero maximo de epocas | `200` |
| `--learning-rate` | Taxa de aprendizado | `1e-4` |
| `--img-size` | Resolucao de entrada | `224` |
| `--runs` | Numero de repeticoes experimentais | `1` |
| `--seed` | Semente inicial | `42` |
| `--heatmaps-per-model` | Quantidade de heatmaps por modelo apos o treino | `0` |

## Treinamento e Validacao

O carregamento dos dados e feito pela classe `DataModule`, que:

- redimensiona as imagens para o tamanho configurado;
- converte as imagens para tensores;
- divide o conjunto de treino em treino e validacao;
- cria `DataLoader` para treino, validacao e teste;
- permite limitar a quantidade de imagens por etapa para testes controlados.

O treinamento usa:

- funcao de perda `CrossEntropyLoss`;
- otimizador `Adam`;
- scheduler `ReduceLROnPlateau`;
- early stopping baseado em `val_loss`;
- armazenamento do melhor estado do modelo.

## Avaliacao e Resultados

Cada execucao salva seus artefatos em `Resultados/<modelo>/<timestamp>/`, com subpastas para:

```text
model/        # Pesos do modelo treinado
metrics/      # Metricas quantitativas
predictions/  # Predicoes geradas
gradcam/      # Mapas de calor
metadata/     # Informacoes da execucao
plots/        # Curvas e matrizes de confusao
```

Ao executar um modelo individual, o resumo e salvo como:

```text
Resultados/summary_<modelo>.json
```

Ao executar todos os modelos, o resumo consolidado e salvo como:

```text
Resultados/summary_all.json
```

O arquivo `Resultados/summary_all.json` registra execucoes repetidas e metricas agregadas. Na consolidacao atual, os melhores resultados medios de F1 foram:

| Modelo ou Ensemble | Rodadas | Acuracia media | F1 medio |
| --- | ---: | ---: | ---: |
| Ensemble `mean_probability` | 5 | 0,8898 | 0,8892 |
| `resnet50` | 5 | 0,8753 | 0,8742 |
| Ensemble `majority_vote` | 5 | 0,8732 | 0,8719 |
| `efficientnet_b0` | 5 | 0,8659 | 0,8651 |
| `convnext_tiny` | 5 | 0,8483 | 0,8470 |
| `vit_b_16` | 5 | 0,8227 | 0,8210 |
| `vmamba` | 5 | 0,7640 | 0,7300 |
| `cnn_shallow` | 5 | 0,7101 | 0,7003 |

## Interpretabilidade com Grad-CAM

O projeto inclui suporte a mapas de calor por Grad-CAM para auxiliar a analise qualitativa das regioes da imagem utilizadas pelos modelos durante a classificacao.

Gerar heatmaps durante uma execucao completa:

```bash
python main.py --model all --runs 5 --heatmaps-per-model 10
```

Executar o modo dedicado de heatmap:

```bash
python main.py --mode heatmap --model resnet --weights caminho/para/model.pth
```

Os resultados sao armazenados nas pastas `gradcam/` dentro de cada diretorio de execucao.

## Relatorios

A pasta `Resultados/` contem arquivos consolidados e relatorios gerados a partir dos experimentos, incluindo:

- `summary_all.json`, com metricas agregadas por modelo e ensemble;
- relatorios HTML preenchidos;
- curvas de perda e acuracia;
- matrizes de confusao;
- predicoes e metadados de execucao.

Esses artefatos sao importantes para documentar a metodologia experimental e sustentar a comparacao entre arquiteturas no texto academico.

## Reprodutibilidade

Para aumentar a reprodutibilidade, o pipeline define sementes globais para Python e PyTorch. Em execucoes multiplas, a semente e incrementada a cada rodada:

```text
seed_da_rodada = seed_inicial + indice_da_rodada - 1
```

Recomendacoes para experimentos academicos:

- registrar a versao do Python e das bibliotecas;
- manter os mesmos splits de treino, validacao e teste;
- executar mais de uma rodada por arquitetura;
- reportar media e desvio padrao das metricas;
- comparar modelos sob os mesmos hiperparametros sempre que possivel;
- preservar os arquivos gerados em `Resultados/`.

## Observacoes Academicas

Este projeto tem escopo compativel com um Trabalho de Conclusao de Curso ou projeto de pesquisa aplicada, pois cobre o ciclo completo de experimentacao em aprendizado profundo:

- definicao do problema;
- organizacao e preprocessamento dos dados;
- implementacao de modelos;
- treinamento supervisionado;
- avaliacao quantitativa;
- comparacao experimental;
- interpretabilidade;
- consolidacao de resultados.

Por se tratar de um projeto experimental, os valores de desempenho devem ser interpretados considerando o tamanho do conjunto de dados, o balanceamento entre classes, a estrategia de particionamento e as condicoes de hardware usadas durante o treinamento.
