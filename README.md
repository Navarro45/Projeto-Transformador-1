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

O conjunto de dados ativo do projeto fica em `dataset/`. O diretorio `legacy/data/` preserva a primeira versao do dataset apenas para historico e nao deve ser usado como fonte dos experimentos atuais.

A camada SDSS DR20 local esta organizada por classe principal:

```text
dataset/
  images/
    galaxy/
    quasar/
    star/
  spectra/
    galaxy/
    quasar/
    star/
  metadata/
    spectra_metadata.csv
  reports/
    subclass_report.md
    processing_subclasses_report.md
  gaia_sdss/
  star_objects/
  star_supervised/
```

Os fluxos baseados em imagens continuam esperando o padrao `torchvision.datasets.ImageFolder` quando forem treinados. Quando os splits de imagem forem materializados, eles devem ficar sob `dataset/train`, `dataset/val` e `dataset/test`, mantendo as pastas `galaxy`, `quasar` e `star` em cada split.

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
+-- dataset/                        # Fonte ativa: imagens, espectros, metadados e camadas derivadas
+-- legacy/data/                    # Dataset inicial preservado apenas como historico
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

A fonte ativa dos experimentos atuais e `dataset/`. A pasta `legacy/data/` nao deve ser usada para novas execucoes, relatorios ou metricas.

O script `scripts/download_sdss_dr20_dataset.py` materializa o dataset SDSS DR20 local a partir do CSV de entrada, salvando imagens, FITS espectrais, plots, metadados e relatorios em `dataset/`.

Executar o download/preparacao completa a partir do CSV:

```bash
python scripts/download_sdss_dr20_dataset.py --csv MyResult_2026824.csv
```

O script `scripts/prepare_dataset.py` permanece disponivel para materializar splits `train/val/test` quando necessario para os modelos de imagem. Ele usa `dataset/` como raiz compartilhada.

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

### Relatorio Das Subclasses Em Processamento

Para listar todas as subclasses preservadas e tambem as subclasses normalizadas usadas pelos modelos supervisionados de estrelas:

```bash
python scripts/generate_processing_subclass_report.py
```

Saidas principais:

```text
dataset/reports/processing_subclasses_report.md
dataset/reports/processing_sdss_raw_subclasses.csv
dataset/reports/processing_star_normalized_subclasses.csv
dataset/reports/processing_star_raw_to_normalized_subclasses.csv
dataset/reports/processing_star_parent_class_counts.csv
dataset/reports/processing_subclasses_summary.json
```

O relatorio separa `SDSS raw`, que representa a subclasse original preservada no dataset local, de `STAR normalizado`, que representa o alvo efetivamente usado pelos modelos supervisionados de subclasses estelares. Subclasses com `count <= 5` sao marcadas como raras, mas nao sao removidas automaticamente.

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
| `--train-dir` | Diretorio de treino ImageFolder; usar a camada ativa em `dataset/train` quando os splits forem materializados | `dataset/train` |
| `--test-dir` | Diretorio de teste ImageFolder; usar a camada ativa em `dataset/test` quando os splits forem materializados | `dataset/test` |
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

## Descoberta Espectral Nao Supervisionada

O projeto tambem possui um modo independente para descobrir agrupamentos dentro da classe `star` usando espectros reais SDSS/SEGUE acessados via MAST. Esse fluxo nao treina modelos de imagem e nao altera `ModelFactory`, `DataModule`, `Trainer`, `Evaluator` ou Grad-CAM.

A fonte cientifica usada como referencia para acesso aos espectros SDSS/SEGUE e o notebook oficial do MAST:

```text
https://spacetelescope.github.io/mast_notebooks/notebooks/SDSS/SDSS_SEGUE_spectra/sdss_stellar_spectra.html
```

### Requisito de Metadados

O modo espectral exige um CSV externo confiavel que associe cada imagem ao objeto SDSS correto. O arquivo deve conter `image_path` ou a combinacao `split,class_name,image_filename`, e pelo menos um identificador astronomico:

```text
obs_id
```

ou:

```text
ra,dec
```

ou:

```text
plate,mjd,fiber
```

O pipeline nao infere `obs_id`, RA/DEC, `objID`, plate, MJD ou fiber a partir do nome do arquivo. Linhas sem associacao rastreavel sao rejeitadas e registradas em `spectral/metadata/rejected_metadata.csv` ou `spectral/metadata/failures.csv`.

### Execucao

Teste pequeno:

```bash
python main.py --mode spectral-unsupervised --spectral-metadata-csv caminho/metadata.csv --max-spectral-samples 10
```

Execucao completa:

```bash
python main.py --mode spectral-unsupervised --spectral-metadata-csv caminho/metadata.csv
```

Parametros principais:

| Parametro | Descricao | Padrao |
| --- | --- | --- |
| `--spectral-metadata-csv` | CSV externo com associacao imagem-SDSS | obrigatorio |
| `--max-spectral-samples` | Limite de estrelas processadas; `-1` usa todas | `-1` |
| `--spectral-length` | Tamanho final do vetor espectral | `1024` |
| `--latent-dim` | Dimensao latente do Autoencoder | `32` |
| `--ae-hidden-dim` | Dimensao oculta do Autoencoder | `256` |
| `--ae-epochs` | Epocas maximas do Autoencoder | `100` |
| `--ae-lr` | Taxa de aprendizado do Autoencoder | `1e-3` |
| `--ae-batch-size` | Tamanho do lote espectral | `64` |
| `--hdbscan-min-cluster-size` | Tamanho minimo de cluster HDBSCAN | `10` |
| `--hdbscan-min-samples` | Minimo de amostras HDBSCAN | `5` |
| `--spectral-cache-dir` | Cache persistente de FITS baixados; usar `dataset/spectral_cache` nos experimentos atuais | `dataset/spectral_cache` |
| `--enable-umap` | Gera projecao UMAP quando `umap-learn` estiver instalado | desativado |

### Fluxo Metodologico

O fluxo executa:

```text
metadata confiavel
  -> resolver SDSS
  -> MAST/astroquery
  -> FITS COADD
  -> loglam, flux
  -> wavelength = 10 ** loglam
  -> limpeza e normalizacao
  -> grade comum
  -> Autoencoder
  -> embeddings
  -> HDBSCAN
  -> clusters espectrais
```

O campo `SUBCLASS`, quando presente no CSV ou no FITS, e preservado somente para analise posterior. Ele nao e usado no treinamento do Autoencoder, PCA, UMAP ou HDBSCAN.

### Resultados Espectrais

Cada execucao salva os artefatos em:

```text
Resultados/
  spectral_unsupervised/
    <timestamp>/
      spectral/
        spectra/
        embeddings/
        clusters/
        plots/
        models/
        metadata/
```

Arquivos principais:

```text
spectral/spectra/spectra.npy
spectral/embeddings/embeddings.npy
spectral/embeddings/pca_2d.npy
spectral/clusters/cluster_labels.npy
spectral/metadata/spectral_metadata.csv
spectral/models/autoencoder.pth
spectral/models/preprocessing.json
spectral/models/hdbscan.pkl
spectral/cache_summary.json
```

Os clusters `-1` representam ruido/outliers do HDBSCAN. Os IDs numericos dos clusters sao arbitrarios e nao significam diretamente tipos `O`, `B`, `A`, `F`, `G`, `K` ou `M`.

## Relatorios

A pasta `Resultados/` contem arquivos consolidados e relatorios gerados a partir dos experimentos, incluindo:

- `summary_all.json`, com metricas agregadas por modelo e ensemble;
- relatorios HTML preenchidos;
- curvas de perda e acuracia;
- matrizes de confusao;
- predicoes e metadados de execucao.

Esses artefatos sao importantes para documentar a metodologia experimental e sustentar a comparacao entre arquiteturas no texto academico.

## Integracao Gaia DR3 x SDSS Local

O projeto tambem possui uma camada nova para cruzar os objetos SDSS ja existentes localmente com o catalogo Gaia DR3. Essa etapa nao baixa novamente imagens, espectros, arquivos FITS ou catalogos SDSS; ela reutiliza os dados ja materializados em `dataset/`.

Fonte local usada por padrao:

```text
dataset/metadata/spectra_metadata.csv
dataset/manifests/download_manifest.csv
```

O manifesto e usado para recuperar RA/DEC quando o indice espectral nao traz essas colunas diretamente. Se algum identificador Gaia ja existir no dataset local, ele e detectado e registrado, mas o fluxo padrao consulta apenas objetos ainda nao cacheados. Por padrao, a integracao Gaia consulta somente objetos `STAR`, preservando `GALAXY` e `QSO` no dataset SDSS local sem usa-los nessa etapa.

Executar a integracao Gaia:

```bash
python scripts/gaia_sdss_integration.py
```

Teste pequeno:

```bash
python scripts/gaia_sdss_integration.py --limit 20
```

Reconsultar apenas erros anteriores do cache:

```bash
python scripts/gaia_sdss_integration.py --retry-errors
```

Diagnosticar conexao com consultas individuais:

```bash
python scripts/gaia_sdss_integration.py --limit 20 --batch-size 1 --no-batch --retry-errors
```

Expandir para outras classes, quando necessario:

```bash
python scripts/gaia_sdss_integration.py --classes GALAXY QSO STAR
```

Gerar novamente apenas relatorios, sem consultar Gaia:

```bash
python scripts/gaia_sdss_integration.py --generate-reports-only
```

Parametros principais:

| Parametro | Descricao | Padrao |
| --- | --- | --- |
| `--input` | Dataset local consolidado ja existente | `dataset/metadata/spectra_metadata.csv` |
| `--manifest` | Manifesto com RA/DEC e caminhos locais | `dataset/manifests/download_manifest.csv` |
| `--output` | Diretorio da camada Gaia | `dataset/gaia_sdss` |
| `--match-radius-arcsec` | Raio de busca posicional no Gaia, em arcsec | `1.0` |
| `--ambiguity-delta-arcsec` | Diferenca maxima entre primeiro e segundo candidato para marcar ambiguidade | `0.2` |
| `--classes` | Classes SDSS usadas na consulta Gaia | `STAR` |
| `--batch-size` | Tamanho dos lotes de consulta TAP | `50` |
| `--retry-errors` | Reconsulta apenas objetos selecionados com `match_status=ERROR` no cache | desativado |
| `--force` | Reconsulta todos os objetos selecionados, preservando cache de outras classes | desativado |
| `--no-batch` | Usa consultas individuais, util para diagnostico de conexao | desativado |
| `--generate-reports-only` | Regera metricas, graficos e relatorios do dataset consolidado | desativado |

Saidas geradas:

```text
dataset/gaia_sdss/cache/gaia_matches.csv
dataset/gaia_sdss/gaia_sdss_comparison.csv
dataset/gaia_sdss/reports/gaia_sdss_report.md
dataset/gaia_sdss/reports/*.csv
dataset/gaia_sdss/reports/summary.json
dataset/gaia_sdss/plots/*.png
```

A classe `gaia_derived_class` e derivada de `teff_gspphot` quando Gaia fornece temperatura efetiva. Ela deve ser interpretada como uma derivacao fotometrica/astrofisica, nao como uma subclassificacao espectral observacional. O relatorio tambem registra status de match, distancias angulares, candidatos ambiguos, cobertura dos parametros Gaia e possiveis efeitos de movimento proprio elevado.

### Classe Evolutiva/Fisica Das Estrelas

Para a classe `STAR`, o dataset consolidado tambem cria anotacoes derivadas para apoiar comparacoes mais interpretaveis que a classe romana tradicional. Os campos principais sao:

```text
spectral_subclass_raw
spectral_class_major
roman_luminosity_class_raw
stellar_evolution_class
stellar_evolution_method
stellar_evolution_status
absolute_g_mag
parallax_quality_flag
```

`stellar_evolution_class` pode assumir valores como `WHITE_DWARF`, `MAIN_SEQUENCE_DWARF`, `RED_GIANT`, `GIANT`, `SUBGIANT` e `UNKNOWN`. Essa classe e derivada de dados Gaia quando disponiveis, usando principalmente `teff_gspphot`, `bp_rp`, `phot_g_mean_mag`, `parallax`, `parallax_error` e `logg_gspphot`. Se o Gaia falhar ou a paralaxe estiver ausente/invalida, a estrela permanece no dataset com `stellar_evolution_class=UNKNOWN`.

Gerar a estrutura por objeto sem duplicar imagens e plots:

```bash
python scripts/build_star_object_dataset.py
```

Gerar a estrutura por objeto copiando imagem RGB e plot espectral:

```bash
python scripts/build_star_object_dataset.py --copy-assets
```

Saida por objeto:

```text
dataset/star_objects/STAR_000001/
  RGB/metadata.json
  Spectrogram/metadata.json
  Photometric/photometric.json
  Physical/physical.json
  Target/target.json
  object.json
```

Por padrao, essa estrutura armazena caminhos para os arquivos ja existentes no projeto. Nenhum FITS, imagem ou espectro SDSS e baixado novamente.

## Hierarchical Random Forest Para Subclasses Estelares

O projeto tambem possui um fluxo supervisionado CPU-first para classificar subclasses estelares usando os espectros SDSS locais de `dataset/star_objects`. O Random Forest e implementado com scikit-learn, mas usa a mesma entrada `main.py` e salva resultados em `Resultados/<modelo>/<timestamp>/`, como os demais experimentos.

Treinar o modelo hierarquico:

```bash
python main.py --model hierarchical_random_forest
```

Treinar o baseline flat:

```bash
python main.py --model flat_random_forest
```

Teste rapido:

```bash
python main.py --model hierarchical_random_forest --max-spectral-samples 500 --rf-n-estimators 100
```

O alvo final e a subclasse espectral normalizada. A subclasse SDSS bruta continua preservada nos relatorios para auditoria. Todas as subclasses normalizadas sao mantidas, inclusive raras; o arquivo `metadata/rare_classes.csv` mostra quais classes possuem poucos exemplos.

O modelo hierarquico usa a estrutura:

```text
STAR
  -> spectral_parent_label
    -> spectral_leaf_label
```

As features sao vetores de fluxo espectral gerados dos FITS locais com `astropy` e `SpectralPreprocessor`. Gaia, imagens, metadados derivados do target e campos fisicos nao entram como features neste primeiro experimento, para reduzir risco de data leakage.

Saidas principais:

```text
Resultados/hierarchical_random_forest/<timestamp>/
  model/model.joblib
  model/flat_baseline.joblib
  metrics/metrics.json
  metrics/flat_vs_hierarchical.csv
  predictions/test_predictions.csv
  predictions/leaf_probabilities.npy
  predictions/parent_probabilities.npy
  metadata/hierarchy.json
  metadata/split_manifest.csv
  metadata/rare_classes.csv
  reports/hierarchical_random_forest_report.md
```

A hierarchical loss e usada para avaliacao e comparacao, nao como funcao de otimizacao por gradiente. Cada `RandomForestClassifier` continua sendo treinado com seu criterio interno (`gini`, `entropy` ou `log_loss`). A estrutura hierarquica entra por classificadores locais, probabilidades condicionais e metricas baseadas no caminho da arvore.

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
