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

## 7. DATA PREPARATION
Notebook: 09\_DATA\_PREPARATION.ipynb

To ensure the quality and consistency of the ESG news dataset prior to the generation of dynamic scores, a process of cleaning and standardizing information was carried out.

The starting point was the ‘df\_finally\_labeled.csv’ file, which already contains the necessary main fields for each news article: company ticker identifier, publication date, ESG thematic classification (Environmental, Social, or Governance), and predicted sentiment (bullish, bearish, or neutral).

First, a structural cleaning of the dataset was performed. Unnecessary columns that do not add value to the subsequent analysis, such as ‘description’, ‘sentiment\_body’, or ‘id’, were removed. This cleaning reduces the file size and simplifies later operations. The next phase focused on ensuring the correct association between tickers and company names. Cases were detected where, due to extraction problems in previous phases, some entries had the ticker but not the company name, or vice versa. To resolve this situation:

* If a news article has the ticker informed but the company absent, the corresponding company name is imputed using a dictionary generated from existing matches in the dataset.
* Reciprocally, if a news article has the company name but the ticker absent, the corresponding ticker is imputed using the inverse mapping.

This process allows for the accurate filling of the vast majority of missing values in these fields, thus ensuring the correct traceability between news and companies.

However, there were still special cases where the ticker appeared as UNKNOWN. To address this issue, a manual correction process based on the company name was designed. The following substitutions were applied:

* "disney" → "DIS"
* "general motors" → "GM"
* "jpmorgan" → "JPM"
* "morgan stanley" → "MS"
* "wells fargo" → "WFC"
* "comcast" → "CMCSA"
* "american express" → "AXP"
* "the home depot" → "HD"

This manual imputation allows for the accurate recovery of the correct tickers in most UNKNOWN cases, avoiding the loss of valuable information for relevant companies in the analysis.

Once the ticker and company issues were resolved, the imputation of the industry sector (`sector`) was addressed for those news articles where the sector was initially absent. To do this, a dictionary was constructed that associates each ticker with the most frequent sector in the dataset. This mapping is applied only in cases of sector absence, avoiding overwriting valid information. Thanks to this strategy, the sector field was completed for a very high percentage of the news articles, allowing for subsequent sectoral analyses of the ESG scores.

Finally, the cleaned, structured, and enriched dataset was chronologically ordered by publication date, ensuring correct temporal alignment for the construction of dynamic analysis windows. The definitive dataset was stored in the `df_cleaned.csv` file, serving as the basis for the generation of daily ESG scores that feed the sustainable investment strategies.

In conclusion, this data preparation phase not only consolidates all the previously extracted and processed information but also establishes the necessary foundations of integrity, consistency, and robustness to confidently face the predictive modeling of ESG evolution based on news events. Each correction applied in this phase represents a fundamental step towards obtaining reliable, replicable, and scientifically valid results in the later stages of the project.

## 8. MEDIA REPUTATION
Notebook: 10\_MEDIA\_REPUTATION.ipynb

Once the cleaning phase of the news dataset was completed, the next step involves the calculation of the media sources' reputation. The objective of this phase is to weigh each news article considering the reliability and influence of the source that publishes it.

### a. Theoretical Basis
Academic literature has demonstrated that repeated media exposure and the frequency of appearance in media significantly influence public perception and the construction of corporate reputation. Deephouse (2000) defines media reputation as a strategic resource that directly impacts the perceived value of companies and argues that media repetition reinforces corporate legitimacy. Carroll (2010) also points out that the media act as amplifiers of reputation, where frequency and visibility are key indicators of influence. Likewise, Fombrun and Shanley (1990) argue that companies more covered by consolidated media enjoy a more solid and sustained reputation over time, which justifies that weighting by frequency of appearance is a valid proxy to measure the relevance and informative impact of a source.

Based on this theoretical foundation, it is established that the most prolific sources (in terms of the number of news articles) should have a greater specific weight in reputational analyses and, by extension, in the ESG prediction system.

* Carroll, C. E. (2010). *Corporate reputation and the news media: Agenda-setting within business news coverage in developed, emerging, and frontier markets*. Routledge.
* Christophersen, M., & Jurish, B. (2021). *RapidFuzz: A fast string matching library*. Retrieved from https://maxbachmann.github.io/RapidFuzz/
* Deephouse, D. L. (2000). Media reputation as a strategic resource: An integration of mass communication and resource-based theories. *Journal of Management, 26*(6), 1091–1112.
* Fombrun, C. J., & Shanley, M. (1990). What's in a name? Reputation building and corporate strategy. *Academy of Management Journal, 33*(2), 233–258.

### b. Development
To address this issue, the process begins by loading the ‘df\_cleaned.csv’ dataset, which contains all the relevant news already structured and labeled. Due to the inconsistency in the nomenclature of the sources, which can distort future analyses, such as "Reuters", "Reuters News Service", or "Reuters World News", a normalization process based on "fuzzy matching" techniques was implemented.

Using the ‘rapidfuzz’ library, a string comparison algorithm was applied to group similar variants under a single identifier. The process involves iterating through all sources and, for each new source, searching for the most similar match among the already normalized ones, provided that the similarity exceeds a predefined threshold of 90%. In this way, a ‘source\_map’ is constructed that unifies all variants under a representative base name, usually the shortest or standard one.

Once the sources are normalized, the volume of news attributed to each source is calculated, also broken down by sector. For this:

* The news articles are grouped by (sector, source) pairs.
* The number of news articles corresponding to each combination is calculated, thus obtaining a first measure of the relative influence of each source within each economic sector.

Based on these news volumes, a scaling function was designed to generate a `reputation_score` for each source. The assignment of the score follows a two-tier logic:

* Sources with 30 or more news articles in a given sector receive a maximum score of 1, understanding that a high frequency of appearance is indicative of recognized influence and reliability in that area.
* For sources with fewer than 30 news articles, a logarithmic scaling function was applied that assigned a score between 0.5 and 1.

This function allows for the progressive assignment of scores, slightly penalizing less representative sources without completely excluding them from the analysis. In summary, the incorporation of the `reputation_score` allows for the addition of an information quality dimension to the ESG news analysis. By weighting the importance of each event not only based on its content but also based on the reliability of the source that originates it, the system's ability to prioritize relevant signals and minimize the impact of events reported by marginal or unreliable media is improved. This strategy contributes to increasing the robustness and interpretability of the global ESG prediction system based on media analysis.

## 9. DATA EXPLORATION
Notebook: 11\_DATA\_EXPLORATION.ipynb

Before proceeding with the definitive construction of the dynamic ESG scores, a detailed exploratory analysis of the processed news dataset was carried out. The objective of this phase is twofold: on the one hand, to obtain a structured view of the thematic, sectoral, and sentimental distribution of the news; on the other hand, to identify patterns that could guide the subsequent definition of ESG signal generation strategies.

### a. Data Preparation
The two main datasets are used, the one containing the news and the one with the ESG scores (`NEWS_+_PRESS_ESG.csv` and `ESG_SCORES.csv`), standardizing the common columns.

### b. Distribution of News by Company and ESG Category
The first step consists of quantifying the media coverage for each of the companies, breaking down the total volume of news according to the corresponding ESG category.

As we can observe, there are companies like Tesla, Apple, or Amazon that account for most of the total news, while others like Valero or Abbvie occupy a more residual space.

The analysis confirms the existence of a notable imbalance in ESG coverage: while some companies show high media exposure distributed evenly across the three ESG categories, others are characterized by concentrating their visibility exclusively on environmental or governance aspects. This heterogeneity suggests that the public perception of companies may be strongly conditioned by the most recurrent ESG topics in the press.

An analysis of the temporal evolution of the volume of ESG news was also carried out, where we can observe how the volume of these grows considerably from mid-2023 to mid-2024.

### c. Clustering Analysis: Identification of Mediatic ESG Profiles
With the aim of detecting latent patterns of mediatic ESG behavior, a clustering analysis was carried out on the companies. This analysis allows for the grouping of companies based on characteristics such as the total number of ESG news, the distribution by ESG category, and the average sentiment associated with each dimension.

For the segmentation, the K-Means algorithm is used, configured to generate three clusters. The choice of this algorithm is justified by its simplicity, speed, and effectiveness in handling standardized numerical data. The selection of three clusters seeks to identify differentiated profiles of companies, potentially corresponding to high, medium, and low levels of mediatic ESG exposure.

The analysis confirms the existence of three well-defined groups: a first cluster composed of companies with high media presence and balanced distribution across the three ESG categories; a second group of companies with moderate exposure, generally dominated by a specific category; and a third cluster that groups companies with low media visibility or poorly defined ESG profiles. This segmentation is especially useful for guiding future business strategies and for focusing predictive analyses based on the mediatic profile of each company.

To facilitate the interpretation and visualization of the obtained clusters, the Principal Component Analysis (PCA) technique is used, which allows for the reduction of the dimensionality of the multivariate space to two principal components. This reduction maintains most of the explanatory variance of the data and makes it possible to graphically represent the structure of the clusters in a two-dimensional plane.

Finally, the ‘df\_summary’ dataset, enriched with the cluster assignment and PCA coordinates, has been saved to be used in the following modeling phases.

In conclusion, this exploratory analysis phase allows for a detailed characterization of the universe of companies based on their mediatic ESG exposure, providing a solid foundation for understanding the dynamics of future score generation. The combination of descriptive analysis, construction of summary variables, and clustering techniques has opened the door to more sophisticated approaches in the interpretation of ESG signals and in the personalization of sustainable investment strategies.
## 10. LSTM
This section details the process of preparing the data for and training a Long Short-Term Memory (LSTM) recurrent neural network model to predict daily ESG scores.

### a. Embedding Creation
Notebook: 12\_EMBEDING\_GENERATION.ipynb

The objective of this section is to transform the content of the news articles into dense numerical representations (embeddings) that capture semantic and contextual relationships between different news items. This embedding generation phase constitutes the first step in preparing the input for the LSTM model. This involves transforming daily texts into fixed-dimension vector representations that capture both semantic content and additional information relevant for prediction tasks.

#### i. Data Preparation
The process works with the ‘NEWS\_+\_PRESS\_ESG.csv’ dataset, which contains the preprocessed and structured ESG news and press releases, and the ‘DF\_SUMMARY.csv’ summary, which provides complementary information such as the cluster to which each company belongs (identified in the previous exploration phase).

Before proceeding with embedding generation, a final validation and cleaning of the data is performed, ensuring the absence of null values and the correct typing of key columns. This phase is essential to ensure the quality of the embeddings and avoid the propagation of errors in subsequent phases.

#### ii. Embedding Generation
The generation process is carried out using the SentenceTransformer architecture (all-MiniLM-L6-v2). Each news article (consisting of a title and/or body of text) is transformed into a high-dimensional numerical vector (768 dimensions). The choice of Sentence Transformers is based on their ability to generate contextual and meaningful embeddings at the sentence or complete document level, overcoming the limitations of traditional models such as Word2Vec or TF-IDF, which lack a deep understanding of context.

#### iii. Grouping and Dimensional Reduction
Subsequently, the news articles are grouped by ticker, date, and ESG category, concatenating all news related to the same company and ESG dimension on the same day into a single text. This strategy is justified by the need to compactly capture all available information for each company-date-category combination, avoiding the fragmentation of signals that could occur if individual news items were processed in isolation.

Once the aggregated corpus was built, embeddings were generated using a pre-trained SentenceTransformer model (all-MiniLM-L6-v2). This choice is based on several reasons:

* SentenceTransformer models are specifically designed to generate embeddings that preserve the semantics of phrases and paragraphs, not just individual words.
* The all-MiniLM-L6-v2 model offers an excellent balance between quality and computational efficiency, capable of generating high-quality embeddings with a reasonable cost of resources.
* Being a multilingual model, it ensures adequate coverage even in the case of news containing technical or financial English expressions.

However, working directly with such high-dimensional embeddings can be inefficient and even detrimental to the training of subsequent neural networks, especially in datasets of limited size. Therefore, dimensionality reduction is applied using Principal Component Analysis (PCA), reducing the representation of each embedding to 100 dimensions. The choice of PCA is justified by its ability to:

* Preserve most of the informative variance of the original set.
* Eliminate redundancies and noise in the data.
* Improve the efficiency and stability of the subsequent LSTM model training.

#### iv. Embedding Enrichment with Cluster Information
To provide the embeddings with greater structural context, each reduced embedding is combined with the cluster information to which the company belongs (obtained previously in the clustering analysis). This combination is implemented by concatenating the reduced vector with the numerical value of the cluster, thus generating an enriched embedding that integrates both the semantic representation of the text and the strategic segmentation identified in previous phases. Finally, the set of generated embeddings has been stored in the ‘EMBEDDINGS.pkl’ file. Each row of the file contains:

* The company identifier (ticker).
* The date of the event.
* The ESG category to which the event belongs.
* The combined embedding vector of 101 dimensions (100 from the textual content + 1 from the cluster).

In conclusion, this embedding generation phase effectively transforms the flow of daily ESG news into a vector representation suitable for sequential processing by recurrent neural networks. The combination of semantic textual information and structured business context lays the foundation for robust predictive modeling, capable of capturing the temporal dynamics of ESG perception in financial markets.

### b. Window Creation
Notebook: 13\_WINDOW\_GENERATION.ipynb

Once the embeddings are created, they are transformed into time series, ideal for input into an LSTM network. This process starts with the EMBEDDINGS.pkl dataset, which was built in the previous section.

#### i. Theoretical Basis
Each window is designed to contain a sequence of 90 days of consecutive ESG events. The choice of this window size responds to several considerations:

* It allows capturing short- and medium-term dynamics in the ESG perception of companies.
* It covers a sufficiently broad temporal range to include both one-off events and sustained trends.
* It aligns with the quarterly update logic of many official ESG scores, facilitating subsequent comparisons. In the case of this work, the ESG scores are those registered with the SEC each quarter.

#### ii. Window Generation with Padding
Since in practice not all companies publish ESG news daily, a padding strategy is implemented for incomplete windows. Specifically:

* Up to 20% of the days in the window (18 days) are allowed to have no real events.
* When there is no event for a given day, the corresponding embedding is filled with a vector of zeros.
* If a company does not have at least 72 real event days within a 90-day window, that window is not generated.

Eliminating all windows with incomplete data can drastically reduce the number of available samples, especially for companies with low media coverage. The introduction of padding allows leveraging the available information while maintaining the structural integrity of the window, and at the same time offers the model the possibility to learn to manage the absence of data as an additional feature.

#### iii. Statistics and Validations
Once the windows are generated, a statistical analysis is performed to validate the quality of the process. Among the main metrics evaluated are:

* Total number of windows generated: 63,401.
* Standard dimension of each window: 90 days × 101 features.
* Number of windows that include padding: 1,548.

These statistics confirm that the vast majority of windows are generated from complete data, while a moderate fraction incorporates padding, which is consistent with the defined strategy.

Example: Ticker: TSLA Category: Social Target date: 2023-07-20 First row of the window embedding: `[-0.03310936 -0.01596753 0.08599661 -0.02037817 0.0988157 -0.08739669 -0.19336331 -0.03057285 -0.00702647 -0.21213117]`

#### iv. Export of Results
The resulting dataset is exported in .pkl format under the name WINDOWS.pkl. This file contains the following structures:

* **X:** list of 90×101 matrices corresponding to the windows.
* **y:** labels or target values associated with each window.
* **tickers:** list of company identifiers.
* **categories:** list of ESG categories (Environmental, Social, Governance).
* **target\_dates:** list of target dates.

The export ensures the persistence of the dataset in an efficient and reusable format for its immediate use in the predictive modeling phase. This file will serve as direct input for the construction and training phase of the LSTM model, ensuring efficient organization and rapid access to the data during batch training processes.

In conclusion, the generation of temporal windows represents a crucial step in the modeling pipeline, allowing the structuring of ESG event information in a way that the LSTM model could learn the underlying dynamics over time. The combination of an adequate window length, a flexible padding mechanism, and coherent grouping by company and category laid the foundation for robust and realistic sequential prediction in the ESG domain.

### c. Window Association
Notebook: 14\_MATCHING\_WINDOWS.ipynb

Once the temporal windows of ESG events are generated from daily embeddings, it is necessary to associate each window with a target value that allows training the LSTM model. This target value corresponds to the official ESG score assigned to the company on a date close to the final date of each window.

#### i. Theoretical Basis
In supervised learning models, the quality of the labels is critical to ensure the reliability of the training and the generalization capacity of the model. In this project, the labels are the official ESG scores, which reflect the environmental, social, and governance assessment of each company on specific dates.

Since the windows are generated based on news and do not always coincide exactly with the dates of the quarterly ESG scores, a tolerant matching process is applied. This process seeks the best possible match between the target date of the window and the dates available in the score series, accepting a maximum margin of 90 days. This approach is essential to maximize the amount of usable data and avoid discarding valuable windows due to small temporal mismatches.

#### ii. Data Preparation
The following files are used:

* The embeddings generated per day (EMBEDDINGS.pkl).
* The 90-day windows (WINDOWS.pkl).
* The official ESG scores (ESG\_SCORES.csv).

#### iii. Tolerant Matching Algorithm
The matching is implemented as follows:

1.  For each window, the key variables are identified: ticker, category, and target\_date.
2.  The ESG score DataFrame is searched for an exact match by ticker, category, and date. If this does not exist, the closest score in time is searched for, provided that the difference between the score date and the window date does not exceed a maximum of 90 days.
3.  If a valid score is found (either exact or tolerant), it is assigned as the target (y) to the corresponding window.
4.  If no score is found within the tolerance margin, the window is discarded to avoid unreliable labels.

ESG scores are usually updated quarterly, so there may be a natural lag between daily news and official ratings. Allowing a margin of up to 90 days ensures temporal coherence without losing a significant amount of data, maximizing the use of the available dataset to train robust models.

Example: (The example from the notebook would be inserted here if available in a structured format)

#### iv. Validation and Export
Upon completion of the matching, all windows that have not been associated with a valid ESG score are filtered out, removing incomplete entries from the dataset. Random inspections are performed to validate that the matches are correct and coherent (e.g., reviewing tickers, categories, and matched dates).

Finally, the complete dataset is exported in .pkl format under the name ESG\_TRAIN\_READY\_2.pkl. This file includes:

* **X:** the matrices corresponding to the temporal windows (90 days × number of features).
* **y:** the matched ESG scores (one per window).
* **tickers:** company identifiers.
* **categories:** ESG categories (E, S, or G).
* **target\_dates:** target dates of the windows.

This dataset constitutes the definitive basis for the training of the prediction model, allowing for validations and backtests with real and correctly matched data.

The precise matching between windows and ESG scores ensures that the predictive model works with consistent data and verified labels. The tolerant matching strategy balances the need for temporal precision with the practicality of maximizing the volume of available data, a key aspect for strengthening supervised learning. With this step, the preprocessing phase is closed, and a clean, complete, and structured dataset is available to proceed with the training and evaluation of the model.

### d. Training
Notebook: 15\_LSTM\_TRAINNING.ipynb

This phase aims to train a Long Short-Term Memory (LSTM) recurrent neural network model, whose objective is to predict daily ESG scores from the previously generated temporal windows. The LSTM architecture is selected for its ability to capture complex temporal patterns and manage long-term dependencies in sequences, something crucial in the media evolution of ESG indicators.

#### i. Data Loading and Preparation
The model is trained on the ESG\_TRAIN\_READY\_2.pkl dataset, which contains:

* **X:** 90-day temporal windows × number of features. Each window contains the daily characteristics (embeddings + cluster).
* **y:** official ESG scores associated with each window.
* **tickers, categories, target\_dates:** metadata for traceability.

Before feeding the model, Min-Max normalization is applied to the data, ensuring that all input values are within a [0, 1] range. This technique is key to accelerating model convergence and avoiding scaling issues between variables.

A 70%-30% train-test split is performed to evaluate the model's generalization capacity, ensuring that no information leakage occurs between the sets. The separation was performed using `train_test_split`, stratifying randomly to maintain the representativeness of the ESG score distribution in each subset. To facilitate efficient data loading, the ESGDataset class was defined, which converts the embedding matrices and scores into PyTorch tensors, with support for CPU and GPU devices.

#### ii. LSTM Architecture Design
The constructed model, named ESG\_LSTM, was designed to effectively capture the temporal and semantic dynamics of ESG news. The architecture consists of several functional blocks:

* **Input:** Each input has the shape (batch\_size, 90, 101), where 90 represents the consecutive days and 101 the daily features (embeddings + cluster).
* **LSTM:** an LSTM with the following characteristics was implemented:
    * Two layers (`num_layers=2`), which increases the model's capacity to capture hierarchical dependencies in the sequence.
    * 128 hidden units per layer (`hidden_dim=128`).
    * Bidirectional configuration: this configuration allows the network to process the sequence both in the direct temporal direction (past → future) and in the reverse direction (future → past), thus capturing patterns that could manifest in both senses.
* **Dropout:**
    * Regularization was applied using `Dropout(0.3)` after the LSTM output, thus reducing the risk of overfitting.
* **Feedforward:**
    * Intermediate dense layer (`Linear(256, 128)`) followed by a ReLU activation function.
    * Final output layer (`Linear(128, 1)`) to produce a single continuous ESG score value.

This architecture is designed to maximize the model's ability to capture the dynamics of ESG events over time, incorporating redundancy through bidirectionality and regularization techniques to improve its generalization.

#### iii. Training Configuration
The training is executed for multiple epochs (adjustable based on the results obtained), iterating through the dataset using DataLoaders that optimize batch loading. In each iteration, the training and validation loss are recorded to monitor the process and avoid overfitting.

* **Loss function:** MSELoss (mean squared error), which is the standard option for continuous regression problems. This function quadratically penalizes deviations between the prediction and the real value, being especially effective in avoiding large errors.
* **Optimizer:** Adam with a learning rate of 1e-3, which provides automatic learning rate adaptation per parameter.
* **Number of epochs:** 20.
* **Batch size:** 64.
* **Device:** GPU (cuda) when available.

Random seeds are fixed in NumPy and PyTorch (SEED=33) to ensure the reproducibility of the results. In addition, PyTorch was configured in deterministic mode to avoid variations between executions.

During each epoch:

* The model is trained on the training set (`train_loader`).
* After each epoch, it is evaluated on the validation set (`val_loader`).
* The average loss in training and validation is calculated.
* If the validation loss improves compared to the previous best (`best_val_loss`), the model is saved to disk (`best_model.pth`).

This procedure replicates a manual early stopping scheme, seeking to preserve generalization without the need to go through an excessive number of epochs.

#### iv. Evaluation
Once the training was completed, the best model was loaded and evaluated on the test set using the metrics: Mean Squared Error (MSE) and R² Score.

The MSE is a standard metric in regression problems that measures the average of the squared differences between the predictions and the actual values. Mathematically, it is expressed as:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

An MSE of 0.85 indicates that, on average, the model's predictions have a very low quadratic error with respect to the real ESG scores. Although the MSE penalizes large errors more (due to its quadratic nature), the result obtained confirms that the model does not make large deviations and that its precision is consistent even in extreme cases. In an ESG prediction context, a low error implies that the generated scores are very close to the official evaluations, which reinforces the model's reliability as a support tool for sustainable strategies and responsible investment.

The R² Score, also known as the coefficient of determination, measures the proportion of the variance in the dependent variable that is explainable by the independent variables. It is calculated as:

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

where:

* $SS_{res}$ is the sum of the squared residuals.
* $SS_{tot}$ is the total sum of squares.

An R² of 0.986 implies that 98.6% of the observed variance in the official ESG scores is being explained by the model. This value is very close to the maximum possible value (1.0), which confirms that the model captures the relationships and temporal patterns present in the data with very high precision. The fact that almost all of the variance is explained means that the model not only fits the known data well but also has high predictive power over new data. In financial applications, this is especially important because it suggests that the model can be used to monitor and anticipate ESG dynamics robustly.

Test MSE (real): 0.85
Test R² (real): 0.986
Sample of ESG predictions vs. actual scores on the test set:
Predicted: 66.52 | Actual: 66.14 | Absolute Error: 0.38
Predicted: 57.65 | Actual: 56.87 | Absolute Error: 0.78
Predicted: 47.30 | Actual: 45.81 | Absolute Error: 1.49
Predicted: 72.68 | Actual: 73.46 | Absolute Error: 0.78

## 11. CREATION OF A COMPLETE DATASET
Notebook: 16\_COMPLETE\_DATASET.ipynb

The inference phase results in a set of predictions which, when added to the official ESG scores, form an almost complete dataset. We perform a series of actions on this dataset, which I explain below.

### a. Theoretical Basis
In practice, official ESG scores are usually published quarterly or annually, leaving significant temporal gaps in data coverage. To overcome this limitation, a system has been designed where the daily predictions generated by the LSTM model fill these gaps, ensuring a continuous time series. This approach is fundamental to providing an updated and granular view of ESG evolution, something increasingly demanded in the financial and sustainability fields.

The final dataset includes an explicit distinction between data from official sources and those generated by prediction, ensuring data transparency and traceability.

### b. Data Loading and Preparation
The following datasets were loaded:

* `NEWS_+_PRESS_ESG_definitivo.csv`: Dataset of processed news and press releases.
* `DF_SUMMARY.csv`: Statistical summary of each company.
* `EMBEDDINGS.pkl`: Daily embeddings of ESG events by company and category.
* `DAILY_ESG_PREDICTIONS.csv`: Predictions of ESG scores based on 90-day windows.
* `ESG_SCORES.csv`: Historical official ESG scores of the analyzed companies.

### c. Generation of New 7-Day Windows
To complement the predictions based on 90-day windows, 7-day windows of embeddings are also generated:

* The `generate_windows_with_padding` function is used to create 7-day windows with up to 20% padding allowed (i.e., at least 6 real data days).
* For each window, the closest predicted score is searched for within a range of ±3 days from the end of the window.
* This process allows increasing the temporal granularity of the predictions, improving the daily coverage of available scores.

A dataset `X_lstm` is created with the 7-day windows, and their corresponding labels `y_lstm` obtained from the previous predictions.

### d. Training of a New LSTM Model
For these new 7-day windows, an additional LSTM model was trained following the same structure as the previous one. This model yielded the following results on the test set:

Test MSE: 0.4569
Test R²: 0.9917


Once the predictions based on 90-day windows and those generated with 7-day windows are available, both sources are unified to maximize temporal coverage.

### e. Filtering of Companies and Categories
To ensure that the final dataset had adequate quality, filtering criteria were applied:

* Each company was required to have at least 100 days of available predictions per ESG category.
* In addition, each company had to have at least two ESG categories with minimum coverage.

### f. Interpolation of Official Scores and Predictions
Since not all dates had predictions or official scores, a specific procedure was designed to build complete time series:

* **Before the first available prediction for a company and category:**
    * Official scores (`ESG_SCORES.csv`) were used.
    * Forward-fill (ffill) interpolation was applied for days without an explicit value.
* **From the first prediction onwards:**
    * Predictions from the LSTM model were used.
    * If a prediction existed for a given date, the prediction was used.
    * If no prediction existed for a specific day after the first event, the last predicted value was maintained.
* For each date and company-category, it was recorded whether the value came from a prediction (`from_official = False`) or an official score (`from_official = True`).

This procedure ensures that each company has an uninterrupted daily ESG score throughout the 2020–2025 period. The `fill_esg_scores_filtered` function implements this mechanism, generating a final DataFrame `df_esg_final_5y`.

### g. Final Dataset Generated
The final dataset contains:

* **Ticker:** Company identifier.
* **Date:** Corresponding day.
* **ESG Category:** Environmental, Social, or Governance.
* **Daily ESG Score (`predicted_score_final`):** Prediction or interpolated value.
* **Source Indicator (`from_official`):** True if the value comes from an official score, False if it comes from an LSTM prediction.

The dataset was stored in `ESG_SCORES_COMPLETOS_2.csv`, serving as the definitive basis for subsequent analyses and for the design of investment strategies based on the evolution of ESG factors.

Notebook: 17\_COMPLETE\_DATASET\_2.ipynb

Finally, each generated temporal window (with a size of 7 days) is associated with its corresponding target ESG score, thus ensuring the correct alignment between the input sequences and the output values for supervised training. To do this, a matching procedure is implemented that, for each window, searches for the closest ESG score in time within a tolerance margin of 3 days. This margin has been defined to allow for slight temporal flexibility that mitigates possible mismatches in publication dates or the availability of predictions, without compromising the accuracy of the matching. If multiple candidates exist within that temporal range, the one with the smallest temporal distance to the analyzed window is selected.

This step is fundamental to ensure that each input sequence is precisely matched with its respective target, which optimizes the quality and reliability of supervised learning in later phases. Furthermore, this procedure strengthens the internal consistency of the dataset by minimizing the risk of errors arising from temporal lags, a critical aspect when working with time series and financial data sensitive to temporal context.

## 12. DEVELOPMENT OF A TRADING ALGORITHM
Notebook: 18\_TRADING\_ALGO.ipynb

This section describes in detail the design, construction, implementation, and validation of a trading system based on the evolution of dynamic ESG scores and classified news events. The main objective was to develop a systematic investment approach that, based on signals extracted from the ESG behavior of companies, could make buy and sell decisions with objective, replicable, and quantifiable criteria.

### a. Data Loading and Preparation
To feed the trading system, three main datasets are used:

* **ESG News (`news`):** Included events classified as environmental, social, or governance, along with sentiment analysis.
* **Market Prices (`prices`):** Daily information on closing prices, volume, and other financial indicators of the analyzed companies.
* **Daily ESG Scores (`esg_scores`):** Previously generated dataset that combined predictions made using LSTM networks with interpolations of historical official scores.

All datasets must be correctly aligned in time and transformed to datetime format to ensure consistent and safe operations during the algorithm's execution.

### b. Modular Design of the Trading System
The system was structured in a modular way, using three main components:

#### 1) Signal Generation (SignalEngine)
The `SignalEngine` class constitutes the first functional block of the trading system and aims to generate quantitative and qualitative signals based on the daily evolution of ESG scores and the classified news events for each company. This module is crucial because it transforms data into objective operational indicators on which subsequent investment decisions are based.

The constructor of the class (`__init__`) receives as arguments the daily ESG scores dataset, the classified news dataset, and the ESG category to operate on (Environmental, Social, or Governance). Additionally, it allows adjusting several essential technical parameters for signal calculation, such as the momentum window (5 days by default), the Z-Score window (20 days), and the percentile window for ESG level classification (90 days). The inclusion of these parameters ensures the system's flexibility, allowing it to be easily adapted to different time horizons or sectors.

The central method of the class is `get_signals_for_day(date)`. This method processes the information corresponding to a specific date and constructs, for each company, a set of quantitative and qualitative signals. The first part of the analysis focuses on ESG scores. Four key indicators are calculated: momentum, which measures the variation of the score with respect to a recent moving window and captures the company's trend in ESG terms; the Z-Score, which evaluates how extreme the current score is in relation to its historical mean and standard deviation; and moving percentiles (P30 and P70), which classify each score into low, medium, or high levels based on its relative position in the historical distribution. This last point is fundamental to identify structural situations such as companies persistently lagging or excelling in sustainability.

In parallel, the day's news is analyzed and labeled according to its predicted sentiment (bullish, bearish, neutral), transforming them into interpretable qualitative categories: ‘very\_positive’, ‘very\_negative’, or ‘neutral’. This classification allows capturing the emotional tone or the potential impact that news can have on the ESG perception of companies.

The final result is a structured dictionary in which each company (ticker) is associated with a set of key signals: `evento_extremo`, `desacoplamiento`, `momentum`, `zscore`, and the corresponding ESG category. This format allows the direct integration of the signals in the next phase of the system, facilitating the evaluation and execution of trading decisions.

In short, `SignalEngine` plays a strategic role within the global system, acting as the brain that transforms unstructured data into high-quality operational indicators. Its modular and flexible design ensures that it can be applied simultaneously to different ESG categories or expanded to incorporate new signal logics in the future. Furthermore, the combination of statistical techniques (such as percentiles and Z-Score) with the semantic interpretation of news offers a robust and sophisticated approach that seeks to minimize bias and maximize the relevance of the generated signals.

#### 2) Signal Evaluation (SignalEvaluator)
The `SignalEvaluator` class represents the second fundamental piece within the modular ESG signal-based trading system. Its main function is to translate the individual signals generated by the `SignalEngine` module into a single aggregated signal and, based on this, determine the specific operational decision: buy, sell, or hold. This process is key to the system's operability, as it converts multiple scattered metrics into a single clear and quantifiable guideline, suitable for automated execution.

The constructor of the class (`__init__`) allows customizing the weights assigned to each type of signal, distinguishing between the three ESG categories: Environmental, Social, and Governance. By default, the weighting is defined in a balanced way for the three categories, giving greater importance to the extreme event (40%), followed by decoupling (30%), momentum (20%), and finally the Z-Score (10%). This initial configuration responds to the logic that very positive or very negative news usually has a more immediate and decisive impact on a company's ESG perception, while quantitative signals such as momentum or Z-Score provide additional nuances to the evaluation. However, the class architecture is designed to be easily adjustable, allowing these weights to be adapted based on backtests or specific user preferences.

The `calcular_signal_final` method is the functional core of the class. For each row (company/day) of the received DataFrame, this method extracts the associated ESG category and applies the weighted combination of signals according to the defined weights. Each type of signal (extreme event, decoupling, momentum, and Z-Score) is multiplied by its corresponding weight, accumulating to obtain a single final signal. This process ensures that the evaluation is consistent and structured, integrating both qualitative and quantitative information into a single continuous value.

Subsequently, the `decision_operativa` method translates this final signal into a discrete operational decision. The logic used is simple but effective: if the final signal exceeds a positive threshold (by default, +0.25), a buy signal (1) is generated; if it falls below a negative threshold (–0.25), a sell signal (–1) is issued; and if it remains within the intermediate range, the recommendation is to maintain the current position (0). This threshold-based policy allows filtering weak or noisy signals and focusing only on those situations where the confidence in the signal is high enough to justify an action.

In summary, `SignalEvaluator` acts as the decision-making body of the trading system, synthesizing the multiple ESG signals into a clear and calibrated operational guideline.

#### 3) Order Execution (SignalExecutor)
The `SignalExecutor` class constitutes the third and final pillar of the modular trading system, being responsible for materializing trading decisions into concrete buy and sell operations, simulating the realistic execution of orders in the financial market. Its design is oriented towards reproducing real market conditions as faithfully as possible, including considerations such as commissions, slippage, and active portfolio management.

From a functional point of view, the constructor (`__init__`) establishes the key parameters of the simulation, such as the initial capital available (default 1 million euros), the maximum exposure per asset (25% of capital), and the costs associated with operations: commissions (0.1%), stop-loss (30%), and slippage (0.1%). These values have been defined to reflect realistic institutional investment scenarios, although their design allows complete customization to adapt to different risk profiles or operational contexts.

The `portfolio` attribute maintains the updated record of open positions for each asset, while `buy_prices` stores the purchase prices, crucial for correctly calculating stop-loss activation. In addition, each trade is recorded in the `trades` list, and the historical evolution of the portfolio is stored in `history`. The class also maintains the `pending_signals` variable, which saves the signals generated today to be executed in the next session, thus simulating the natural delay between signal generation and its real execution.

The `_execute_order` function implements the operational logic for each individual order. When a buy signal (signal == 1) is detected, the algorithm calculates the amount to invest based on the available capital and the maximum exposure per asset, adjusting this amount by the "score" confidence of the signal (this introduces a very relevant dynamic nuance). The number of shares is calculated in whole numbers (no fractions), and it is verified if sufficient capital is available to cover the operation including commissions. If conditions are favorable, the purchase is made, and the positions and available capital are updated. In the case of a sell signal (signal == -1), the system implements a random partial sale (between 60% and 80% of the position), reflecting the operational reality where the complete position is rarely liquidated except in cases of force majeure. Forced sales due to stop-loss are also considered to protect the portfolio against abrupt price drops.

The `update_daily` method represents the operational core of the class and simulates the complete daily trading cycle. First, it executes the pending signals that were registered the previous day, applying the corresponding slippage to approximate the real execution price. Subsequently, it stores the new signals generated that same day to be executed in the next session, thus simulating a realistic flow of deferred orders. Next, it daily reviews all open positions to verify if any have reached the stop-loss threshold, in which case it automatically executes the sale. Finally, it updates the total value of the portfolio (adding available cash plus the market value of open positions) and records it in the history.

A notable feature is the precision with which the portfolio value is calculated: for each open position, the closing price of the asset in the market that day is consulted and multiplied by the number of shares in the portfolio, thus allowing the daily evolution of the assets to be monitored.

In addition, the class includes the `get_history_df` and `get_trades_df` methods, which facilitate the structured export of operational results. `get_trades_df` adds valuable information such as the gross and net calculation of each operation (considering commissions), which allows detailed financial analyses to be performed such as the calculation of the portfolio turnover ratio, the total cost of friction, or the profitability adjusted for operating costs.

Overall, `SignalExecutor` acts as the transactional engine of the system, transforming signals into concrete actions and dynamically managing the portfolio over time. Its modular design and attention to detail (slippage, stop-loss, maximum exposure, deferred execution) provide a robust and realistic simulation that allows evaluating the operational viability and profitability of the ESG strategy under plausible market conditions. Furthermore, by maintaining a comprehensive record of all operations and their historical evolution, this class lays the foundation for rigorous post-trading analysis, essential in any professional process of backtesting and algorithmic optimization.

#### 4) Function to Analyze Results
The `analizar_resultados` function represents the post-operative analysis module of the trading system, whose objective is to exhaustively evaluate the performance of the implemented strategy. This function synthesizes in a single block all the necessary logic to calculate both standard financial metrics (such as profitability and risk) and specific indicators on the operations carried out (such as win rate or profit factor), allowing a rigorous and quantifiable evaluation of the results obtained.

From a technical point of view, the function receives several key datasets: the daily portfolio history (`df_history`), the detailed trade log (`df_trades`), and optionally the daily prices (`df_prices`). The first fundamental operation is to ensure correct temporal alignment: if a price dataset is provided, historical records are filtered to retain only dates with valid prices, thus avoiding distortions due to days without market activity.

Once the data is cleaned and ordered, the function calculates the daily return of the portfolio (`returns` column) through the percentage change in capital day by day. This constitutes the basis for deriving key metrics such as:

* **Total Return:** the total accumulated return during the entire backtest period.
* **Annualized Return:** the annualized rate of return, which normalizes profitability to a one-year period, allowing direct comparisons with benchmarks.
* **Annualized Volatility:** a risk indicator calculated as the standard deviation of daily returns scaled by the square root of 252 (annual business days).
* **Sharpe Ratio:** a classic measure of risk-adjusted performance that compares the excess return over the risk-free rate with volatility. This ratio is especially relevant to understand if the strategy generates value beyond simple market exposure.

In addition, the function incorporates a comprehensive drawdown calculation: it measures the largest percentage decline from a historical peak in the evolution of capital. The `max_drawdown` metric quantifies the worst decline suffered, while `calmar_ratio` relates the annualized return to the magnitude of the drawdown, providing a risk/return balance complementary to the Sharpe Ratio.

Another key aspect is the microstructural analysis of operations. To do this, the function pairs each sell operation with the last corresponding buy operation (matching by ticker and ESG category) and calculates the net profit (pnl) as well as the duration of the operation (`holding_days`). From these buy-sell pairs, fundamental operational statistics are derived:

* Total number of operations and BUY vs SELL distribution.
* **Win Rate:** proportion of closed operations with positive profit, a key metric to evaluate the effectiveness of the strategy.
* **Profit Factor:** ratio of total gains to total losses, considered a robust measure of the overall efficiency of the strategy.
