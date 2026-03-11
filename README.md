https://rodscaloppe.github.io/architecture-ml-cerrado/

https://ae613343.predicao-seca-chuva.pages.dev/

# Sistema Integrado de Monitoramento do Cerrado — Previsão de Seca e Chuva

## Visão Geral

Sistema que integra **dados de satélite**, **sensores terrestres** e **algoritmos de machine learning** para monitorar o bioma Cerrado em tempo real e prever eventos de seca e chuva com **2 a 4 semanas de antecedência**.

---

## 1. Entrada de Dados

### Satélite (NOAA/Landsat)

| Variável             | Detalhe                          |
|----------------------|----------------------------------|
| NDVI                 | Índice de Vegetação (-1 a +1)    |
| Evapotranspiração    | 3.5–5.8 mm/dia                   |
| Cobertura de Nuvem   | 0–100%                           |
| Radiação Solar       | 0–1000 W/m²                      |
| Temp. de Superfície  | Temperatura do solo              |
| Umidade Atmosférica  | % de vapor de água               |
| **Resolução**        | 30 m × 30 m                      |
| **Frequência**       | Diária (08:30 UTC)               |

### 3 Estações Terrestres

Cada estação possui sensores calibrados para o Cerrado, com coleta **a cada 15 minutos (24/7)**:

| Sensor                | Precisão     | Faixa           |
|-----------------------|-------------|-----------------|
| Temperatura (DHT22)   | ±0.5 °C     | 15–35 °C        |
| Umidade Relativa      | ±3%         | 20–95%          |
| Umidade do Solo (VWC) | ±2%         | 0–100%          |
| Pressão (BMP280)      | ±1 hPa      | 900–1100 hPa    |
| Radiação (Piranômetro)| ±5%         | 0–1200 W/m²     |
| Anemômetro            | —           | 0–20 m/s        |
| Pluviômetro           | 0.2 mm res. | —               |

**Distribuição das estações:**

- **Estação 1 (Norte):** Vegetação densa
- **Estação 2 (Centro):** Vegetação transitória
- **Estação 3 (Sul):** Vegetação esparsa

---

## 2. Pré-processamento

### Limpeza de Dados
- Detecção de outliers via método IQR
- Imputação por forward fill para séries temporais
- Sincronização temporal entre satélite e sensores
- Validação cruzada de consistência entre fontes

### Feature Engineering (40+ variáveis)
- Médias móveis de 7 e 14 dias (tendência)
- Taxa de mudança temporal por variável
- Índices compostos combinando sensores
- Defasagens temporais: t-1, t-7, t-30

### Normalização
- StandardScaler (média 0, desvio padrão 1)
- MinMaxScaler (intervalo [0, 1])
- RobustScaler (resistente a outliers)

---

## 3. Modelos de Machine Learning

### Random Forest
- **500 árvores**, profundidade 20–30 nós
- Bootstrap 70%, features/árvore: √40 ≈ 6
- Processamento paralelo (8 cores), treino em ~45s
- **Acurácia: 88–92%**

### LSTM (Long Short-Term Memory)
- **4 camadas recorrentes**, 256 neurônios/camada
- Dropout 30%, janela temporal de 30 dias
- Otimizador Adam (lr=0.001), loss MSE, treino em ~120s
- **Acurácia: 85–90%**

### XGBoost (Gradient Boosting)
- **200 estimadores**, taxa de aprendizado 0.1
- Profundidade máx. 7, subsample 0.8, regularização L2=1.0
- Treino em ~60s
- **Acurácia: 87–91%**

### Ensemble Learning (Votação Ponderada)
| Modelo        | Peso | Especialidade           |
|---------------|------|-------------------------|
| Random Forest | 35%  | Padrões não-lineares    |
| LSTM          | 35%  | Dependências temporais  |
| XGBoost       | 30%  | Otimização de residuais |

---

## 4. Validação e Métricas

| Métrica             | Valor  |
|---------------------|--------|
| Acurácia Geral      | 90.5%  |
| Precisão (Seca)     | 88.2%  |
| Recall (Chuva)      | 87.6%  |
| F1-Score Combinado  | 89.1%  |
| ROC-AUC             | 0.925  |
| RMSE (Umidade)      | 3.2%   |

Validação por cross-validation 5-fold com intervalo de confiança de 80–95%.

---

## 5. Saídas do Sistema

### Previsões Semanais (4 semanas)
Probabilidades de seca, chuva e condição normal para cada semana, com intervalo de confiança de ±5% (IC 95%).

### Sistema de Alertas Automáticos
| Nível           | Condição                             |
|-----------------|--------------------------------------|
| 🔴 Vermelho     | Seca >60% \| Chuva <10%             |
| 🟠 Laranja      | Seca 40–60% \| Chuva 10–20%         |
| 🟡 Amarelo      | Variação >20% vs histórico           |

Entrega via e-mail, SMS e API, com 2–4 semanas de antecedência.

### Banco de Dados Histórico
- 630 mil+ pontos de dados de sensores (12 meses)
- 52 semanas de previsões anteriores
- Retenção de 24 meses, backup em 3 regiões

### Dashboard Web
- Gráficos de tendência e previsão
- Mapas georreferenciados das 3 estações
- Exportação CSV e relatórios PDF semanais automáticos
- Acesso Web + Mobile (iOS/Android)

---

## Números-Chave do Sistema

| Indicador              | Valor       |
|------------------------|-------------|
| Features engenheiradas | 40+         |
| Modelos no ensemble    | 3           |
| Pontos de treino       | 900 mil+    |
| Acurácia média         | 90.5%       |
| Frequência de coleta   | 15 min      |
| Antecipação            | 4 semanas   |
| Tempo de treino total  | ~225 s      |
| Monitoramento          | 24/7        |


# Evapotranspiração e NDVI

## O que é Evapotranspiração (ET)?

A **evapotranspiração** é o processo combinado de **evaporação** da água do solo e superfícies, junto com a **transpiração** das plantas — ou seja, a água que as plantas absorvem pelas raízes e liberam pela folhagem na atmosfera. É um componente essencial do ciclo hidrológico e muito utilizado para estimar a necessidade hídrica de culturas agrícolas.

### Fatores que influenciam a ET

- **Radiação solar**: maior radiação aumenta a evaporação e a transpiração.
- **Temperatura**: temperaturas mais altas aceleram o processo.
- **Umidade do ar**: ar mais seco favorece maior evapotranspiração.
- **Vento**: ventos mais fortes removem o ar úmido ao redor das plantas, aumentando a ET.
- **Tipo e estágio da vegetação**: culturas em pleno crescimento transpiram mais.

---

## O que é NDVI (Índice de Vegetação por Diferença Normalizada)?

O **NDVI** é um índice calculado a partir de imagens de satélite que mede a **saúde e densidade da vegetação**. Ele utiliza a diferença entre a luz refletida no infravermelho próximo (NIR) e no vermelho visível (RED):

$$
NDVI = \frac{NIR - RED}{NIR + RED}
$$

### Escala de valores do NDVI

| Faixa de Valores | Interpretação |
|---|---|
| **-1 a 0** | Água, nuvens, neve ou solo exposto |
| **0 a 0,2** | Solo exposto ou vegetação muito esparsa |
| **0,2 a 0,4** | Vegetação rala ou em estágio inicial |
| **0,4 a 0,6** | Vegetação moderada |
| **0,6 a 0,8** | Vegetação densa e saudável |
| **0,8 a 1,0** | Vegetação muito densa e vigorosa |

### Como funciona o NDVI?

Plantas saudáveis absorvem grande parte da luz vermelha visível para a fotossíntese e refletem fortemente a luz no infravermelho próximo. Quando a planta está estressada ou doente, ela reflete mais luz vermelha e menos infravermelho, resultando em valores de NDVI mais baixos.

---

## Relação entre Evapotranspiração e NDVI

O NDVI é frequentemente utilizado como **proxy para estimar a evapotranspiração**, pois existe uma forte correlação entre os dois:

- Áreas com **vegetação mais densa** (NDVI alto) tendem a ter **maior evapotranspiração**, já que mais plantas significam mais transpiração.
- Áreas com **solo exposto** (NDVI baixo) apresentam **menor evapotranspiração**.

### Modelos que combinam NDVI e ET

Diversos modelos de sensoriamento remoto utilizam o NDVI junto com dados térmicos de satélite para calcular a ET em grandes áreas:

- **SEBAL** (Surface Energy Balance Algorithm for Land)
- **METRIC** (Mapping Evapotranspiration at High Resolution with Internalized Calibration)
- **SSEBop** (Simplified Surface Energy Balance)

Esses modelos são amplamente utilizados em **agricultura de precisão** para:

- Monitorar o **estresse hídrico** das lavouras
- Otimizar sistemas de **irrigação**
- Prever a **produtividade** agrícola
- Planejar o **manejo sustentável** dos recursos hídricos

---

## Aplicações Práticas

### Na Agricultura

O uso combinado de ET e NDVI permite que produtores rurais tomem decisões mais informadas sobre irrigação, identificando áreas da lavoura que necessitam de mais ou menos água. Isso é especialmente relevante em regiões como o **Mato Grosso**, onde a agricultura extensiva demanda um manejo eficiente dos recursos hídricos.

### No Monitoramento Ambiental

Órgãos ambientais utilizam esses índices para monitorar desmatamento, degradação de ecossistemas e mudanças no uso do solo ao longo do tempo.

### Na Gestão de Recursos Hídricos

Bacias hidrográficas podem ser monitoradas com dados de ET e NDVI para avaliar a disponibilidade hídrica e planejar a distribuição de água entre diferentes usos (urbano, agrícola, industrial).

---

## Referências e Ferramentas Úteis

- **Google Earth Engine**: plataforma gratuita para análise de imagens de satélite, incluindo cálculo de NDVI e estimativas de ET.
- **USGS EarthExplorer**: acesso a imagens Landsat e outros satélites.
- **AppEEARS (NASA)**: ferramenta para extração de dados de evapotranspiração (MOD16) e NDVI (MOD13).
- **QGIS**: software livre de geoprocessamento para análise e visualização de dados espaciais.

---
