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

## 4. FIRST ROBERTA MODEL
This section details the development and application of the first RoBERTa model, which serves to classify news articles as either related to ESG topics or not.

### a. News Tagging
Notebook: 01_LONGCHAIN.ipynb

One of the primary challenges in working with large volumes of unstructured text is the need to identify and classify information units that are truly relevant to the analysis objectives. In this thesis, this translates to distinguishing between news articles that address issues related to ESG (Environmental, Social, or Governance) criteria and those that do not, despite previous filtering during news extraction. This stage is critical as it acts as an input filter for the system.

To train an NLP model for this task, a sample of labeled news is required. For this purpose, a solution based on large language models (LLMs) was chosen, utilizing the LangChain library as an orchestration framework and the OpenAI API, specifically the GPT-4o model, as the inference engine. Unlike traditional supervised learning-based classifiers, this approach leverages the advanced semantic capabilities of language models without the need for a pre-existing human-labeled dataset.

In the context of this thesis, a news article is considered "ESG-related" if it directly or indirectly addresses any of the three ESG dimensions, even if it does not explicitly mention the acronym "ESG." For example, a news article about fines imposed for polluting discharges or about a change in the board of directors due to corruption scandals is undoubtedly ESG-relevant, even without direct reference to that terminology.

#### i. Description of the Automated Tagging Process
The technical implementation of the classification system was carried out using a combination of LangChain and OpenAI, with a sequential processing logic:

1.  **Prompt Definition:** A detailed prompt was designed instructing the model to act as an expert in ESG analysis. The prompt used was:
    ```
    You are an expert in ESG (Environmental, Social, and Governance) analysis.
    Analyze the following financial news article and determine whether it is related to ESG topics.
    Please interpret ESG in a broad and inclusive way:
    - Environmental includes sustainability, climate change, emissions, energy, biodiversity, etc.
    - Social includes diversity, inclusion, employee well-being, human rights, education, community impact, etc.
    - Governance includes ethical behavior, transparency, corporate governance, executive compensation, stakeholder rights, etc.
    Even if the news is not explicitly labeled as ESG, it may still relate to ESG themes based on its content.
    Always respond in English, even if the original article is in another language.
    Return:
    - `esg`: true or false
    Text:
    {input}
    ```

2.  **Model Used:** The chosen model was gpt-4o, a state-of-the-art LLM optimized for fast and contextually accurate responses.

3.  **Classified Sample:** A random sample of 6,000 news articles was extracted from the total dataset. The defined prompt was applied to each news article, and the model returned a structured output with a boolean label (`True` if the news is ESG-related, `False` otherwise).

This approach allowed for the rapid construction of a labeled dataset (`sample_esg_or_not.csv`) that is consistent and based on a solid interpretative foundation.

### b. Model Architecture and Training
Notebook: 02_ROBERTA_ESG_OR_NOT.ipynb

Once the labeled sample dataset was obtained, the next step involved training a natural language processing model. The chosen model for this process was RoBERTa. The selection of this model stems from the need to interpret not only isolated keywords but also the complete semantic context in which expressions are framed. For this reason, a model based on the RoBERTa (Robustly Optimized BERT Approach) architecture, developed by Liu et al. (2019), was chosen as an evolution of the traditional BERT model. RoBERTa is a language model based on Transformers, pre-trained through self-supervised learning on large amounts of textual data.

The choice of RoBERTa was based on several strategic criteria:

* **Ability to handle financial and news texts with complex nuances, irony, or implicit relationships**, typical of business language.
* **Effective transfer learning:** Since the model has already been pre-trained on a large corpus, only fine-tuning on the specific set of ESG news is required.
* **Robustness to textual noise:** Essential for news extracted from different sources with varied writing styles.

Overall, RoBERTa provides a solid and efficient solution to address the binary ESG / non-ESG classification required in this phase of the system.

#### i. Training Dataset
The dataset used to train the RoBERTa model comes from the initial labeling process described in the previous section. From the complete corpus of news extracted between 2020 and 2025, a sample of 6,000 news articles was selected and automatically labeled using a language model, with manual validation performed on a representative proportion. To prepare the texts before being processed by the NLP models, an initial cleaning and structuring process was applied. First, null values in the ‘title’ and ‘content’ columns were replaced with empty strings (''), ensuring that no errors occurred during the concatenation of fields. Subsequently, a new field called ‘input\_text’ was generated, consisting of the concatenation of the news title and the first 300 characters of the main content.

This decision addresses the need to maintain a balance between context and computational efficiency: the title usually captures the essence of the news, while an initial fragment of the content provides additional details without overloading the input with excessively long text.

To avoid class imbalance problems, a resampling process was implemented:

* The majority class (ESG news) was downsampled to match the number of examples in the minority class (non-ESG news).
* This process allows for the training of a balanced model, minimizing bias towards the majority class.

#### ii. Preprocessing and Tokenization
Before introducing the data into the RoBERTa model, a process called tokenization is necessary. This is a key step in any natural language processing (NLP) model because it converts raw text (e.g., a sentence) into a sequence of numbers that the model can interpret. The selected tokenizer is the official one from ‘roberta-base’. The main operations performed are:

* Conversion of text to tokens according to the pre-trained vocabulary. For example, the sentence ‘The environment is critical’ can be transformed into something like: `[0, 1332, 1234, 19, 4567, 2]`
* Automatic padding to adjust all sequences to the same length (256 tokens), as these models require all sequences within a batch to have the same length.
* Truncation in cases where texts exceeded the maximum length allowed by the model.

The result of this process was a set of input tensors (`input_ids`) and attention masks (`attention_mask`). The input tensors represent the sequences, and the attention masks indicate which positions are real content and which are padding, all necessary to properly feed the RoBERTa model.

The image represents an example of sentence classification.

To properly feed an NLP model in PyTorch, it is essential to structure the data in a format compatible with the library's internal tools. PyTorch provides a module called `torch.utils.data.Dataset`, which is a standard base class for creating custom datasets. This allows the data, even if it comes from a tabular file (like a pandas DataFrame), to be transformed into tensors and batches, which are the working units of deep learning models.

For this purpose, a custom class called ‘CustomDataset’ was designed. This class allows for the transformation. In particular, the dataset contains two key columns: `text` (with the news text) and `label` (the target class, encoded as an integer).

The ‘CustomDataset’ class implemented the following main functionalities:

* **Initialization:** Stores the list of texts and their respective labels, along with the tokenizer and the maximum length (`max_len`) allowed for the sequences.
* **`__len__` method:** Returns the total number of samples, a standard requirement for integration with PyTorch DataLoader.
* **`__getitem__` method:** Processes each text by applying `tokenizer.encode_plus`, which performs tokenization and returns the `input_ids` (tokens) and the `attention_mask` necessary for Transformer-based models.

The method returns a dictionary with:

* `input_ids`: tokenized sequence,
* `attention_mask`: attention mask,
* `targets`: real label (converted to a PyTorch tensor).

Subsequently, for the training and evaluation phase, the balanced dataset was divided into three subsets: train (80%), validation (10%), and test (10%).

This division was performed using the `train_test_split` function from Scikit-learn, applying the `stratify` argument to ensure that the proportion of classes remained constant in all subsets, which is fundamental to avoid bias and ensure robust evaluation.

Once the datasets (`train_dataset`, `valid_dataset`, `test_dataset`) were defined, objects called DataLoaders were created, which are fundamental tools in PyTorch because they: divide the dataset into automatic mini-batches during training (`batch_size` parameter), randomize (`shuffle`) the samples in each epoch to improve generalization, and efficiently optimize data loading into memory.

#### iii. Model Architecture
The model designed for ESG news classification is based on a fine-tuning approach of a pre-trained model, specifically `roberta-base`. This approach leverages the linguistic knowledge previously acquired by RoBERTa, quickly adapting it to a new task with relatively little additional data. The general scheme of the model is as follows:

1.  **Loading the base model (RoBerta Model):** `RobertaModel` is the implementation of the RoBERTa model available in the Hugging Face Transformers library. It is based on the Transformer architecture, a deep neural network designed to work with text sequences and capture complex contextual relationships between words. It transforms the tokenized texts (the `input_ids` and `attention_mask`) into high-dimensional dense representations called contextual embeddings. Each token in the input receives an embedding that not only represents its isolated meaning but also its relationship with the other tokens in the sequence.

    The main output used in this project is `pooler_output`, which corresponds to the final representation of the special `[CLS]` token (which is always at the beginning of the sequence). This vector is treated as a summary of the global meaning of the text and is typically used for sequence classification tasks.

2.  **Dropout layer:** The Dropout layer (with a probability of 0.3 in this project) is a very common regularization technique in neural networks. Implemented using `nn.Dropout` in PyTorch, it works by randomly deactivating a percentage of the neurons during each training pass. This prevents the model from memorizing the training data too much (overfitting) and improves its ability to generalize to new data. In this case, by applying Dropout to the `pooler_output`, the part that most influences the final prediction is being regularized.

3.  **Linear projection layer:** This is a fully connected layer (implemented as `nn.Linear` in PyTorch) that takes the output of the RoBERTa model (a high-dimensional vector, e.g., 768 dimensions in `roberta-base`) and projects it to a lower dimension. In this case: 2 neurons, one for each class (ESG / non-ESG).

    This layer is responsible for converting the contextual representation of RoBERTa into a concrete final prediction. It acts as a "bridge" between the part of the model that understands language and the part that makes decisions.

This architecture was chosen due to its simplicity and efficiency, as the model leverages the power of RoBERTa without adding additional complex layers, which keeps the architecture lightweight and efficient. Furthermore, its robustness in using only a linear layer and Dropout on top of the `pooler_output` follows a recommended and well-established practice in the literature for text classification tasks with transformers (see Liu et al., 2019).

#### iv. Training Procedure
The training of the RoBERTa model fine-tuned for ESG classification was carried out through a carefully designed procedure to ensure numerical stability, good convergence, and avoid typical problems such as overfitting. The key components and their function are described below:

1.  **Loss function: Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss):** This function efficiently combines binary cross-entropy (used for binary classification problems) with the sigmoid function, all in one step. By integrating the sigmoid function directly within the loss, numerical instability that can arise if they are applied separately is avoided. It is the standard and recommended choice for binary classification tasks in deep neural networks like Transformers.

2.  **Optimizer: AdamW (Adam with Weight Decay):** AdamW is an improvement over the traditional Adam optimizer. It introduces weight decay in a decoupled manner, which improves the model's ability to generalize and prevents weights from growing too large during training. AdamW is highly recommended for Transformer-based models, as it maintains good stability during optimization and helps prevent overfitting, which is crucial in NLP tasks where models are often very large.

In this project, a systematic optimization of hyperparameters using exhaustive methods such as grid search, random search, or Bayesian algorithms was not carried out. Instead, a trial-and-error approach was adopted, adjusting the values of key hyperparameters (such as batch size and learning rate) based on prior experience and best practices described in the literature for Transformer-based models.

Main hyperparameters:

* **Batch size:** 32 examples per iteration.
* **Number of epochs:** 10 complete passes over the training set.
* **Initial learning rate:** adjusted by scheduler, starting at 1e-5.

Throughout the training process, the loss on the training and validation sets and the accuracy were monitored to evaluate the quality of the predictions. This monitoring allowed for the identification of potential overfitting problems and the application of early stopping (early detection), in this case, after 5 epochs, when the improvement on the validation set stabilized or began to deteriorate.

The complete training was performed in a GPU environment, allowing for reasonable adjustment times even when working with the RoBERTa model.

#### v. Results and Evaluation
After the training process was completed, the performance of the model was evaluated on the validation set. The main results obtained were:

* **Accuracy (global accuracy):** 88.9%
* **Precision (precision in ESG class):** 89%
* **Recall (sensitivity to detect ESG news):** 91%
* **F1-Score:** 88%

In addition, a confusion matrix was constructed that reflects the balance between false positives and false negatives. These results indicate that the model is capable of effectively identifying news related to ESG, maintaining an adequate balance between precision and sensitivity.

The relatively low rate of false negatives (ESG news incorrectly classified as non-ESG) is particularly important, given that the priority of the system is to maximize the detection of events relevant to corporate sustainability.

#### vi. Conclusion
The RoBERTa model fine-tuned through transfer learning has proven to be an effective tool for addressing the first major challenge of the system: the accurate filtering of news related to ESG.

Thanks to its ability to interpret the context of news, overcome the purely keyword-based approach, and quickly adapt to the financial domain, the model provides a reliable foundation on which to build the following analysis modules.

**Model:** esg\_model\_weights.pt

### c. Inference on the Complete Dataset
Notebook: 03\_INFERENCE\_ROBERTA\_1.ipynb

After the completion of the training and validation of the RoBERTa model specialized in the detection of ESG (Environmental, Social, Governance) news, the model was applied to the complete set of news collected within the time frame of March 2020 to March 2025. This phase allowed for the automatic and efficient classification of each news article according to its ESG relevance, serving as an indispensable preliminary step for subsequent thematic and strategic analyses.

#### i. Corpus Preparation for Inference
The set of news articles used for inference was stored in a single file ‘noticias\_totales.csv’. However, since the RoBERTa model requires well-structured and length-controlled textual inputs, it was necessary to perform the same processing that was carried out in the training phase. This step ensures that the predictions are consistent and comparable with the results obtained during validation.

#### ii. Inference Procedure
The model was applied to the complete news corpus using a DataLoader, which allows for the processing of data in batches and optimizes computational efficiency, especially on GPUs. The flow of operations was as follows:

1.  **Prediction of logits:** For each batch, the model generates raw outputs (logits) corresponding to the ESG and non-ESG classes. Logits represent the scores before applying the sigmoid function.
2.  **Conversion to discrete predictions:** Using the `torch.max` function, the class with the highest probability was selected for each news article. This converts the predictions into discrete values: 0 → News not related to ESG and 1 → News related to ESG.
3.  **Storage of results:** The predictions were assigned to a new column `esg_pred` in the original DataFrame.

#### iii. Validation and Quality Control
Although the model had been previously validated on the test set, a basic post-inference verification was performed (total number of news classified as ESG – 400K):

* **Manual review of a sample:** A random subset of several news articles classified as ESG and non-ESG was selected for manual inspection.
* **Thematic coherence:** The news labeled as ESG predominantly included relevant events such as sustainability initiatives,
## 5. SECOND NLP MODEL
Following the identification of news articles generally related to ESG factors using the first RoBERTa model, the next step involves a more granular classification of these news articles into one of the three traditional dimensions of ESG analysis: Environmental (E), Social (S), or Governance (G).

This second classification is essential for subsequently constructing separate indicators, analyzing the evolution of each category.

The task of assigning a news article to a specific ESG category presents several difficulties:

* Many articles may address multiple aspects simultaneously (e.g., diversity policies within an environmental sustainability strategy).
* Traditional keyword-based classification approaches are insufficient, as ESG concepts often appear implicitly or intertwined within the texts.
* Accuracy and consistency in classification are critical to avoid biases in subsequent indicators.

Notebook: 04\_LONGCHAIN:2.ipynb

The method for creating the labeled sample is the same as in the first model: LangChain + OpenAI API (gpt-4o). All news articles previously labeled as ‘esg\_pred == 1’ (i.e., ESG-relevant) in the initial inference process are filtered, forming an initial working set.

Subsequently, a random sample of 8,000 news articles is extracted, aiming for a sample size sufficient to capture thematic diversity and ensure a solid foundation for the training and validation of subsequent models. A structured prompt was developed instructing the model to act as an ESG analyst, providing clear definitions for each category:
Each news article was evaluated individually, resulting in a categorical label (Environmental, Social, Governance) that was added as a new column in the DataFrame.

### a. Model Architecture and Training
Notebook: 05\_ROBERTA\_E\_S\_G.ipynb

For this second classification task, the strategy of using a model based on the RoBERTa (Robustly Optimized BERT Approach) architecture was maintained, for the same reasons that motivated its initial choice.

#### i. Training Dataset
The dataset used in this phase comes from the ‘sample\_e\_s\_g.csv’ file, generated in the thematic labeling phase described in the previous section. Each news article includes: the text field, which contains the title concatenated with the abbreviated body (first 300 characters) of the article, and the `esg_category` field, which contains the assigned label (Environmental, Social, or Governance). To prepare the training dataset, the textual labels are mapped to numerical values:

* Environmental → 0
* Social → 1
* Governance → 2

**Class Balancing:** Given that the initial distribution of categories showed differences, the number of examples per class was equalized (n = 1680 examples per category) using random undersampling (resample) techniques, thus reducing the risk of bias towards the majority class. Once the DataFrame was balanced, it was divided into 3 sets: training (70%), validation (15%), and test (15%). The partitioning was done in a stratified manner to maintain the proportion of classes in each subset.

The architecture of the second model is the same as the first, with the difference in the classification layer, which in this case has 3 neurons, each corresponding to an ESG category.

#### ii. Training Procedure
The training process was carried out using the same configuration as in the first model. During training, the loss on the validation set was monitored at each epoch, applying implicit early stopping criteria if signs of overfitting were detected.

Epoch 1/10 | Train Loss: 0.7331, Train Acc: 0.6465 | Val Loss: 0.3074, Val Acc: 0.8810
Epoch 2/10 | Train Loss: 0.2717, Train Acc: 0.9031 | Val Loss: 0.2848, Val Acc: 0.8876
Epoch 3/10 | Train Loss: 0.1673, Train Acc: 0.9419 | Val Loss: 0.2959, Val Acc: 0.8929
Epoch 4/10 | Train Loss: 0.1053, Train Acc: 0.9671 | Val Loss: 0.3049, Val Acc: 0.9021
Epoch 5/10 | Train Loss: 0.0580, Train Acc: 0.9810 | Val Loss: 0.3674, Val Acc: 0.8981
Early stopping triggered!


#### iii. Results and Evaluation
After training, the model was evaluated on the independent test set. The main results were:

* Global Accuracy: 91%
* Error Distribution:
    * Most frequent errors: confusion between Social and Governance news, given the semantic proximity of certain topics.
    * High accuracy for the Environmental class, likely due to the specificity of environment-related terms.

These results reflect that the model is capable of solidly identifying the predominant ESG dimension in the news, achieving accuracy levels suitable for feeding the subsequent modules of score generation and trading signals.

In addition to the general global accuracy metric (91%), other detailed metrics were calculated for each of the ESG categories (Environmental, Social, Governance) on the test set. This is crucial for evaluating not only the overall performance of the model but also its ability to effectively discriminate between the three classes, especially considering the complex and often overlapping nature of ESG topics.

| Category      | Precision | Recall   | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
| Environmental | 0.89      | 0.97     | 0.93     | 252     |
| Social        | 0.93      | 0.90     | 0.92     | 252     |
| Governance    | 0.93      | 0.89     | 0.91     | 252     |
| Accuracy Global | 0.92      |          |          |         |

### b. Inference on the Complete Dataset
Notebook: 06\_INFERENCE\_ROBERTA\_2.ipynb

Once the RoBERTa model specialized in the thematic classification of ESG news was trained, the model was applied to the complete set of news previously identified as relevant to ESG matters. The process begins with loading the ‘total\_news\_esg\_filtered.csv’ file, which contains all the news labeled as relevant (`esg_pred == 1`). To prepare the inputs for the model, the `text` column was used as input, composed of the news title concatenated with the first 300 characters of the content, ensuring a balance between information richness and the maximum length allowed by the RoBERTa model.

Each news article is labeled with a number (0 for Environmental, 1 for Social, and 2 for Governance), which is subsequently mapped to its corresponding textual label to facilitate the interpretation of the results.

After complete labeling, the following distribution of categories can be observed: Environmental = 110702, Social = 151232, and Governance = 169286.

To ensure homogenization in company identification, small corrections were applied to the names of companies detected in the news, such as:

* "jp morgan" → "jpmorgan"
* "cvs health" → "cvs"
* "the walt disney company" → "disney"

The new dataset, which includes the ESG thematic classification for each news article, is stored in the ‘news\_second\_classified.csv’ file. This dataset forms the fundamental basis for the following phases of the project, allowing for the construction of specific daily scores by ESG dimension and the design of differentiated investment strategies.

In conclusion, this inference phase completes the thematic structuring of the ESG news corpus, providing the system with a precise, scalable, and aligned information flow for the dynamic tracking of sustainability factors at the business level.

## 6. THIRD NLP MODEL
After completing the thematic classification of ESG-relevant news, the development of a third analysis module aimed at evaluating the sentiment conveyed by each news article is addressed. The objective of this phase was to enrich the database with an additional variable that captures the implicit perception conveyed by financial media about companies, a fundamental aspect for incorporating a qualitative analysis dimension into the subsequent modeling of ESG scores.

Sentiment analysis was specifically focused on ESG news, classifying the general tone of each news article into three possible categories:

* **Bullish:** news with an expected positive impact on the company. Examples: announcements of successful sustainability initiatives, awards, improvement of ESG policies.
* **Bearish:** news with an expected negative impact. Examples: regulatory sanctions, environmental problems, governance scandals.
* **Neutral:** news without a significant foreseeable impact.

To maintain methodological consistency with the previous models, the generation of the labeled sample is carried out following the same approach: LangChain + OpenAI API (gpt-4o).

Notebook: 07\_LONGCHAIN\_3.ipynb

Prompt used:

You are an expert financial analyst specialized in ESG-related news.
Analyze the following news article and classify its overall sentiment towards the company's performance as one of:

bullish (positive impact expected)
bearish (negative impact expected)
neutral (no significant impact) Only respond with one of these three words: bullish, bearish, or neutral. Text: {input}

### a. Model Architecture and Training
Notebook: 08\_SENTIMENT\_ANALYSIS.ipynb

#### i. Training and Application of a Proprietary Sentiment Analysis Model
The dataset used for this task is located in the ‘sentiment\_sample\_e\_s\_g.csv’ file, obtained in the previous phase through language model-assisted classification. Each news article in the dataset contains the concatenation of the title and the beginning of the news body, as well as the sentiment label predicted manually as bullish, bearish, or neutral. The first stage of the process involves adequately preparing the dataset. For this:

* Textual sentiment is mapped to numerical values, assigning 0 to bearish news, 1 to neutral news, and 2 to bullish news.
* Observing a slight imbalance between classes, it is necessary to apply a random undersampling (downsampling) strategy to equalize the number of examples per class, setting a sample size of 2,312 examples for bearish and neutral, equal to the number of bullish examples available.
* The dataset is reorganized into a standard format for classification, with two columns: `body` (input text) and `label` (mapped sentiment).

Subsequently, the texts are tokenized using the official RoBERTa tokenizer (`roberta-base`), truncating and padding up to a maximum length of 256 tokens. The balanced dataset is divided into three subsets: 70% for training, 15% for validation, and 15% for test, applying stratified partitioning to preserve the proportions of classes in each subset. This preparation allows for the construction of TensorDataset and DataLoader for each partition, facilitating efficient batch training and evaluation.

#### ii. Model Architecture
To address the multi-class sentiment analysis task on financial news (classification into bearish, neutral, bullish), a custom model called `NewsSentimentAttentionModel` was developed, designed based on the pre-trained RoBERTa and integrated with additional multi-head attention and feed-forward layers that enrich the ability to capture complex patterns in the texts.

The architecture follows a modular scheme composed of several layers and key functional blocks:

* **Base Layer: RoBERTa Encoder:** The model starts by loading ‘RobertaModel’ (configuration 'roberta-base'), which acts as the main encoder and is responsible for converting the tokenized text (input) into a matrix of contextual embeddings of dimension (sequence x 768).
* **Additional Multi-Head Attention:** To further refine and enrich the representation generated by RoBERTa, an `nn.MultiheadAttention` layer was added. This layer implements a self-attention mechanism, which allows the model to weight different parts of the input sequence in relation to itself, improving the ability to identify complex relational patterns within the text (e.g., relating distant subjects and predicates in the sentence or identifying global emotional context). Justification: Although RoBERTa already contains multiple attention layers, adding an additional external layer can act as a kind of refinement, especially useful when working on a specific domain (financial news with emotional nuances).
* **Additional Feed_Forward:** After the attention layer, a feed-forward block is implemented that follows the typical structure of Transformer blocks, composed of: `Linear(768, 768) + ReLU + Dropout(0.1) + Linear(768, 768)`. This improves the model's ability to capture non-linear and hierarchical patterns within the textual representation.
* **Second Normalization + Pooling Block:** Reinforces the model's robustness. Normalization ensures that activations remain within controlled ranges, while pooling (e.g., mean-pooling or max-pooling over the sequence) condenses the information into a fixed vector of 768 dimensions.
* **Final Classification:** After obtaining the final 768-dimensional vector, a sequence of dense layers is applied that ultimately generates 3 output classes: bearish, neutral, bullish.

Input (input_ids, attention_mask)
→ RoBERTa (last_hidden_state)
→ Multi-Head Attention (4 heads)
→ Add & Norm
→ Feed-Forward + Dropout + ReLU
→ Add & Norm
→ Pooling (mean over sequence)
→ Dense Layer (768→128) + ReLU + Dropout
→ Dense Layer (128→3)
→ Output logits (for CrossEntropyLoss)


**Fine-tuning Strategy: Partial Layer Freezing:** Freezing consists of blocking the parameters (weights) of certain layers of the model so that they are not updated during training. This allows parts of the model with already acquired knowledge (in this case, from the general pre-training of RoBERTa) to remain stable, while only the most relevant layers for the specific task are adjusted. In this specific context, all layers except the last 4 of the encoder were frozen: `layer.8`, `layer.9`, `layer.10`, and `layer.11`.

The model training was carried out for a maximum of 10 epochs, using:

* **Loss function:** CrossEntropyLoss, suitable for multi-class classification problems.
* **Optimizer:** AdamW, especially effective in Transformer architectures.
* **Initial learning rate:** 2e-5.
* **Batch size:** 16 examples.
* **Acceleration using GPU when available.**

To prevent overfitting phenomena, an early stopping mechanism based on validation loss is implemented, with a patience of 3 epochs. The model is saved to disk each time the validation loss improves, allowing for the recovery of the best trained version once the process is complete.

#### iii. Results
The model shows solid performance, especially in the Negative and Positive classes, where F1-scores of 0.83 and 0.84 were achieved respectively. This indicates that it is particularly effective in identifying news with a clear negative or positive bias.

The Neutral class, however, presents a lower F1-score (0.68), reflecting a greater difficulty in accurately detecting this type of news. This behavior is expected in sentiment analysis tasks, where neutral news tends to be more ambiguous or less semantically defined, making its classification difficult even for advanced models.

The global accuracy of the model is 78%, which implies that approximately 8 out of 10 news articles are correctly classified into their corresponding sentiment category.

The evaluation of the model has shown that the system is capable of adequately capturing the tone of the news, offering performance comparable to that of commercial sentiment analysis models, but with the advantage of having been trained specifically on a corpus of ESG news, aligned with the project objectives.

Once the model was validated, inference was performed on the complete corpus of relevant news (`news_second_classified.csv`). For this, the trained model is loaded, the entire corpus is tokenized with the same configurations used in training, and inference is performed batch by batch, classifying each news article as bearish, neutral, or bullish.

bearish 149725
neutral 145726
bullish 135769


The result of this phase was the generation of a new sentiment column (`predicted_sentiment_final`) associated with each ESG news article in the global dataset. This column, along with the thematic classification (Environmental, Social, Governance), forms the necessary information base for constructing the dynamic ESG scores that will guide the investment strategies in the subsequent phases of the system.

In conclusion, the development of a proprietary sentiment analysis model adapted to th
