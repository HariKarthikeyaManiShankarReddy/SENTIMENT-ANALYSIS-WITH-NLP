# Sentiment Analysis with NLP using Logistic Regression

## Project Description
Sentiment Analysis is one of the most vital and commonly used applications in Natural Language Processing (NLP). It is the computational process of identifying, extracting, and categorizing opinions expressed in a piece of text—specifically to determine whether the writer’s attitude toward a particular topic, product, movie, or service is positive, negative, or neutral. 

This task is widely implemented in industries such as entertainment, e-commerce, marketing, customer care, and social media tracking to understand public reception at scale.

Automating sentiment analysis helps organizations process massive volumes of customer feedback rapidly without manual review bottlenecks. This allows companies to respond to customer grievances promptly, track entertainment reception trends, and make data-driven decisions based on genuine user experiences. 

* **Positive Example:** *"A masterful production about one of the great masters of comedy and his life."*
* **Negative Example:** *"This movie is slower than a soap opera... muddled, implausible plot."*

This program builds a text classification pipeline using a **Term Frequency-Inverse Document Frequency Vectorizer (TF-IDF Vectorizer)** for advanced feature extraction and a **Logistic Regression** linear classifier to perform binary sentiment prediction.

---

## Code Implementation Workflow

### 1. Library Ecosystem
* **Pandas:** Utilized for loading, managing, and slicing tabular structures from the dataset.
* **re & string:** Deployed inside a custom preprocessing function for regex-based numeric substitution and rapid string translations.
* **Scikit-learn:** An advanced machine learning suite used for dataset partitioning (`train_test_split`), text vectorization (`TfidfVectorizer`), model training (`LogisticRegression`), and evaluation matrix extraction.
* **Matplotlib & Seaborn:** Integrated to map, plot, and render a colored confusion matrix heatmap.

### 2. Loading and Robust Error Handling
* The file is loaded via `pd.read_csv()`. 
* To ensure execution resilience, an embedded validation structure verifies whether the required `review` and `sentiment` targets exist. If the file is missing or contains structural errors, the pipeline gracefully alerts the user and initializes structurally sound synthetic mock data to prevent script crashing.
* Data cleansing filters the dataset down to strict `positive` and `negative` classes, dynamically dropping any row containing null records.

### 3. Text Preprocessing Pipeline
Text features are passed through a custom `clean_data()` function to standardize strings before numerical conversion:
* **Lowercasing:** Transforms characters to lowercase via `.lower()` to treat "Amazing" and "amazing" identically.
* **Punctuation Stripping:** Clears punctuation symbols efficiently utilizing Python's built-in `string.punctuation` mapping dictionaries.
* **Digit Sanitization:** Substitutes numeric characters smoothly with a whitespace using regex (`re.sub(r'\d+', ' ')`) to preserve adjacent text words from fusing together.
* **Vectorized Stop Words Optimization:** To achieve peak performance, removing English stop words was shifted out of the slow, manual Python row-by-row loops. Instead, Scikit-learn's optimized native `'english'` dictionary parameter handles exclusion at compile time directly inside the vectorizer.

### 4. Data Partitioning and Stratification
The collection is split into a **70% training configuration** and a **30% testing split** using `train_test_split()`. Crucially, `stratify=y` is enforced. This maintains an identical ratio of positive vs. negative targets across both split groups, ensuring the model trains and tests on balanced data representations without bias.

### 5. TF-IDF Feature Extraction
Because machine learning classifiers cannot read raw text, words are converted into numerical numerical feature arrays using TF-IDF. The parameter `ngram_range=(1, 2)` is applied, extracting individual terms (unigrams) and contiguous word pairings (bigrams). This helps capture semantic sequences like distinguishing between "good" and "not good".

The mathematical formula for calculating a term's TF-IDF weight is:

$$\text{TF-IDF}_{i,d} = \text{TF}_{i,d} \times \ln\left(1 + \frac{N}{N_i}\right)$$

Where:
* $\text{TF}_{i,d} = \frac{\text{Occurrences of term } i \text{ in document } d}{\text{Total number of words in document } d}$
* $N_i = \text{Total number of documents containing term } i$
* $N = \text{Total number of documents within the dataset}$

### 6. Model Training
**Logistic Regression** serves as the binary classifier. Because TF-IDF feature sets on text produce wide dimensions (up to 5,000 distinct columns in this setup), the optimizer parameter is configured with `max_iter=1000`. This allows the gradient descent algorithms plenty of mathematical room to safely reach optimal convergence parameters.

### 7. Performance Evaluation
* **Accuracy Score:** Identifies the correct predictions out of all samples.
* **Classification Report:** Yields granular metrics mapping **Precision** (prediction validity), **Recall** (target search coverage), and the **F1-Score** (harmonic mean balancing precision and recall).
* **Confusion Matrix:** Provides a clear visual layout displaying true positives, true negatives, false positives, and false negatives mapped to a Seaborn-colored heatmap.

---

## Dataset Properties
The dataset utilized for this program consists of **50,000 highly polar movie reviews** compiled from the Internet Movie Database (IMDb). The rows are balanced perfectly containing 25,000 positive reviews and 25,000 negative reviews, presenting an ideal, unbiased environment for natural language processing pipelines.

---

## Model Evaluation Output

Successfully loaded 50000 rows from the dataset!

--- Model Evaluation ---
Accuracy score:  0.8873333333333333

Classification report of model is:
               precision    recall  f1-score   support

    negative       0.90      0.88      0.89      7500
    positive       0.88      0.90      0.89      7500

    accuracy                           0.89     15000
   macro avg       0.89      0.89      0.89     15000
weighted avg       0.89      0.89      0.89     15000

# Resources
GeeksforGeeks (NLP Tokenization and Regular Expressions)

Scikit-learn API Documentation (Logistic Regression & TfidfVectorizer documentation)

Natural Language Processing by Pushpak Bhattacharyya and Aditya Joshi
