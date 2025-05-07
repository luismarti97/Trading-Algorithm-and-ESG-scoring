# Trading-Algorithm-and-ESG-scoring
In this project, I will explain step by step how to create a reliable ESG scoring based on external information, using NLP and LSTM, with the objective of using the data in a trading algorithm.
**Abstract:** This Master's Thesis presents the design, implementation, and evaluation of an innovative system for the investment and rating of companies based on ESG (Environmental, Social, and Governance) criteria. The system utilizes Natural Language Processing (NLP) techniques to analyze large volumes of financial news, identify relevant events, and assess the associated sentiment. This information serves as input for Long Short-Term Memory (LSTM) recurrent neural network models, which predict daily dynamic ESG scores for each company. Finally, these scores are integrated into a signal-based investment algorithm, aiming to validate the hypothesis that the use of unstructured ESG information enhances investment decision-making. The analysis focuses on 30 companies from the S&P 500 index, distributed across six key sectors.
**Keywords:** ESG, Quantitative Investment, Natural Language Processing (NLP), Sentiment Analysis, LSTM Networks, Investment Signals, Financial Markets.
## Table of Contents
1. [INTRODUCTION](#1-introduction)
2. [THEORETICAL FRAMEWORK](#2-theoretical-framework)
    * [a. The Integration of ESG Criteria in Financial Markets](#a-the-integration-of-esg-criteria-in-financial-markets)
    * [b. Limitations of Traditional ESG Ratings](#b-limitations-of-traditional-esg-ratings)
    * [c. The Treatment of Unstructured Data as an ESG Alternative](#c-the-treatment-of-unstructured-data-as-an-esg-alternative)
    * [d. Natural Language Processing for ESG Rating](#d-natural-language-processing-nlp-for-esg-rating)
    * [e. Prediction of ESG Scores with LSTM Models](#e-prediction-of-esg-scores-with-lstm-models)
    * [f. Investment Strategies Based on ESG Signals](#f-investment-strategies-based-on-esg-signals)
3. [DATA EXTRACTION](#3-data-extraction)
    * [a. Company and Sector Selection](#a-company-and-sector-selection)
    * [b. Sources Used and Justification](#b-sources-used-and-justification)
    * [c. Validation and Homogenization Process](#c-validation-and-homogenization-process)
4. [FIRST NLP MODEL](#4-first-nlp-model)
    * [a. News Tagging](#a-news-tagging)
        * [i. Description of the Automated Tagging Process](#i-description-of-the-automated-tagging-process)
    * [b. Model Architecture and Training](#b-model-architecture-and-training)
        * [i. Training Dataset](#i-training-dataset)
        * [ii. Preprocessing and Tokenization](#ii-preprocessing-and-tokenization)
        * [iii. Model Architecture](#iii-model-architecture)
        * [iv. Training Procedure](#iv-training-procedure)
        * [v. Results and Evaluation](#v-results-and-evaluation)
        * [vi. Conclusion](#vi-conclusion)
    * [c. Inference on the Complete Dataset](#c-inference-on-the-complete-dataset)
        * [i. Corpus Preparation for Inference](#i-corpus-preparation-for-inference)
        * [ii. Procedure](#ii-procedure)
        * [iii. Validation and Quality Control](#iii-validation-and-quality-control)
5. [SECOND NLP MODEL](#5-second-nlp-model)
    * [a. Model Architecture and Training](#a-model-architecture-and-training-1)
        * [i. Training Dataset](#i-training-dataset-1)
        * [ii. Training Procedure](#ii-training-procedure-1)
        * [iii. Results and Evaluation](#iii-results-and-evaluation-1)
    * [b. Inference on the Complete Dataset](#b-inference-on-the-complete-dataset-1)
6. [THIRD NLP MODEL](#6-third-nlp-model)
    * [a. Model Architecture and Training](#a-model-architecture-and-training-2)
        * [i. Training and Application of a Proprietary Sentiment Analysis Model](#i-training-and-application-of-a-proprietary-sentiment-analysis-model)
        * [ii. Model Architecture](#ii-model-architecture-1)
        * [iii. Results](#iii-results-1)
7. [DATA PREPARATION](#7-data-preparation)
8. [MEDIA REPUTATION](#8-media-reputation)
    * [a. Theoretical Basis](#a-theoretical-basis)
    * [b. Development](#b-development)
9. [DATA EXPLORATION](#9-data-exploration)
    * [a. Data Preparation](#a-data-preparation-1)
    * [b. Distribution of News by Company and ESG Category](#b-distribution-of-news-by-company-and-esg-category)
    * [c. Clustering Analysis 2](#c-clustering-analysis-2)
10. [LSTM](#10-lstm)
    * [a. Embedding Creation](#a-embedding-creation)
        * [i. Data Preparation](#i-data-preparation-2)
        * [ii. Embedding Generation](#ii-embedding-generation-1)
        * [iii. Aggregation and Dimensional Reduction](#iii-aggregation-and-dimensional-reduction)
        * [iv. Embedding Enrichment](#iv-embedding-enrichment)
    * [b. Window Generation](#b-window-generation)
        * [i. Theoretical Basis](#i-theoretical-basis-1)
        * [ii. Window Generation with Padding](#ii-window-generation-with-padding)
        * [iii. Statistics and Validations](#iii-statistics-and-validations)
        * [iv. Results Export](#iv-results-export)
    * [c. Window Association](#c-window-association)
        * [i. Theoretical Basis](#i-theoretical-basis-2)
        * [ii. Data Preparation](#ii-data-preparation-3)
        * [iii. Tolerant Association Algorithm](#iii-tolerant-association-algorithm)
        * [iv. Validation and Export](#iv-validation-and-export-1)
    * [d. Training](#d-training)
        * [i. Data Loading and Preparation](#i-data-loading-and-preparation)
        * [ii. LSTM Architecture Design](#ii-lstm-architecture-design)
        * [iii. Training Configuration](#iii-training-configuration)
        * [iv. Evaluation](#iv-evaluation-1)
11. [CREATION OF A COMPLETE DATASET](#11-creation-of-a-complete-dataset)
    * [a. Theoretical Basis](#a-theoretical-basis-2)
    * [b. Data Loading](#b-data-loading)
    * [c. Generation of New 7-Day Windows](#c-generation-of-new-7-day-windows)
    * [d. Training of a New LSTM Model](#d-training-of-a-new-lstm-model)
    * [e. Filtering of Companies and Categories](#e-filtering-of-companies-and-categories)
    * [f. Interpolation of Official Scores and Predictions](#f-interpolation-of-official-scores-and-predictions)
    * [g. Final Generated Dataset](#g-final-generated-dataset)
12. [DEVELOPMENT OF AN INVESTMENT ALGORITHM](#12-development-of-an-investment-algorithm)
    * [a. Data Loading and Preparation](#a-data-loading-and-preparation-1)
    * [b. Modular Design of the Trading System](#b-modular-design-of-the-trading-system)
    * [c. Investment Strategy by ESG Categories](#c-investment-strategy-by-esg-categories)
    * [d. Operational Backtest](#d-operational-backtest)
    * [e. Results Analysis](#e-results-analysis)
    * [f. Comparison of the ESG Strategy Against the Benchmark](#f-comparison-of-the-esg-strategy-against-the-benchmark)
    * [g. Interpretation and Conclusions](#g-interpretation-and-conclusions)
    * [h. Lines of Improvement and Exposure Optimization](#h-lines-of-improvement-and-exposure-optimization)
    * [i. Summary](#i-summary)
13. [STREAMLIT AND GOOGLE CLOUD](#13-streamlit-and-google-cloud)
    * [a. ESG Dashboard](#a-esg-dashboard)
        * [i. Objectives](#i-objectives-2)
        * [ii. General Project Architecture](#ii-general-project-architecture)
        * [iii. Application Development](#iii-application-development)
        * [iv. Justification of the Cloud Architecture](#iv-justification-of-the-cloud-architecture)
        * [v. Deployment](#v-deployment)
    * [b. ESG Strategy Simulator and Signal Generator](#b-esg-strategy-simulator-and-signal-generator)
        * [i. Objective 3](#i-objective-3-1)
        * [ii. Development](#ii-development-1)
    * [c. Training on Vertex AI](#c-training-on-vertex-ai)
        * [i. Scalability](#i-scalability-1)
        * [ii. Deployment](#ii-deployment-2)
        * [iii. Conclusions](#iii-conclusions-2)
14. [DISCARDED APPROACHES](#14-discarded-approaches)
    * [a. Use of Company Sustainability Reports](#a-use-of-company-sustainability-reports)
    * [b. Constant Information to Complement the LSTM Network Beyond the Cluster](#b-constant-information-to-complement-the-lstm-network-beyond-the-cluster)
15. [BIBLIOGRAPHY](#15-bibliography)

## 1. INTRODUCTION
In recent years, environmental factors and social responsibility have gained significant importance in financial markets, driven by regulatory evolution, social norms, and the active role of institutional investors. The integration of ESG (Environmental, Social, and Governance) criteria into investment analysis processes has been supported by studies demonstrating their correlation with long-term company growth.

However, measuring ESG performance remains a challenge, marked by opaque methodologies, heterogeneous criteria, and a lack of standardization in traditional ratings, in addition to their low update frequency.

In this context, the processing of large volumes of unstructured information, such as financial news, emerges as a promising alternative, offering a more reactive and sensitive view of changes in corporate behavior. The application of Natural Language Processing (NLP) techniques facilitates the exploitation of this data, enabling the classification of texts, extraction of relevant information, and accurate sentiment assessment. Complementarily, Long Short-Term Memory (LSTM) recurrent neural network models have proven effective in time series prediction.

This thesis focuses on the intersection of sustainability, quantitative finance, and artificial intelligence, with the primary objective of designing, implementing, and evaluating an investment algorithm based on ESG signals generated from news, as well as the issuance of dynamic ESG ratings. The proposed system includes:

1.  An ESG classification and sentiment analysis module, using NLP models to identify relevant news and classify it according to its dimension (E, S, or G) and polarity.
2.  An LSTM model that predicts a daily dynamic ESG score per company, capturing its temporal evolution based on the processed news.
3.  A signal-based investment algorithm that uses these scores as input to generate buy or sell decisions, incorporating realistic execution aspects.

The analysis is conducted on a sample of 30 companies from the S&P 500 index, equally distributed across six key economic sectors, aiming to ensure broad and diversified coverage of the US market.

This project aims to contribute to both academia and professional practice, proposing a replicable and scalable approach for the integration of ESG criteria into automated investment processes.

## 2. THEORETICAL FRAMEWORK
This section delves into the fundamental concepts underpinning this work:

### a. The Integration of ESG Criteria in Financial Markets
This describes the increasing importance of ESG factors in financial markets, driven by regulation, demands from institutional investors, and evidence of their relationship with long-term profitability. It mentions the growth of assets managed under ESG criteria and the development of specialized financial products and rating agencies.

### b. Limitations of Traditional ESG Ratings
This analyzes the main criticisms of conventional ESG ratings, including the lack of methodological standardization among providers, low internal and external coherence, the limited transparency of methodologies, and the infrequent update cycles.

### c. The Treatment of Unstructured Data as an ESG Alternative
This presents the use of unstructured data, such as news, as an alternative and more dynamic source of information for ESG evaluation, capable of capturing relevant events with greater immediacy and offering diverse perspectives. It mentions the technical challenges associated with its processing, such as named entity recognition and sentiment analysis.

### d. Natural Language Processing (NLP) for ESG Rating
This introduces the field of NLP and its application in the ESG context, highlighting the advancement brought by pre-trained language models based on Transformer architectures. Key tasks such as topic classification and sentiment analysis are mentioned, along with how NLP allows for the transformation of textual information into quantifiable variables for predictive models.

### e. Prediction of ESG Scores with LSTM Models
This explains the use of Long Short-Term Memory (LSTM) recurrent neural network models to model the temporal evolution of ESG performance. The architecture of LSTMs and their ability to learn long-term dependencies are described, making them suitable for generating daily dynamic ESG scores from classified news.

### f. Investment Strategies Based on ESG Signals
This defines the concept of signals in quantitative investment systems and how ESG signals, constructed from dynamic scores generated by LSTM, can be used for buy or sell decisions. Examples of signal activation rules and the possibility of integrating complementary signals are mentioned.

## 3. DATA EXTRACTION
This section details the process of collecting the ESG news dataset used in this work, covering a 5-year period (March 2020 - March 2025).

### a. Company and Sector Selection
This describes the selection of 30 companies from the S&P 500 index, equally distributed across six key economic sectors: Healthcare, Technology, Energy, Financials, Communication Services, and Consumer Discretionary. The choice of this set is justified by its representativeness and the aim for sector diversity. The specific companies included in each sector are listed.

### b. Sources Used and Justification
This details the three main sources of information used:

1.  **Quantexa API News (2023–2025):** Highlighting its named entity recognition (NER) capabilities and ESG thematic filters.
2.  **NewsAPI (2020–2022):** Explaining the need to apply a subsequent filtering process using NER techniques (with the Spacy tool and the 'en\_core\_web\_sm' model) to correctly associate news with the selected companies.
