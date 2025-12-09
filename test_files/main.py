import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import contractions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
import gensim.downloader as api
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
import joblib

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

def expand_contractions(text):
    return contractions.fix(text) # Expand contractions like don't to do not

def normalize_repeats(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)  # Words like soooo become soo

newline_re       = re.compile(r'\n')
url_re           = re.compile(r'https?://\S+|www\.\S+')
email_re         = re.compile(r'\S+@\S+')
number_re        = re.compile(r'\d+')
allowed_chars_re = re.compile(r"[^a-zA-Z0-9!?'* ]")
multi_space_re   = re.compile(r'\s+')

def clean_text(text):
    text = text.lower()  # Lowercase
    text = expand_contractions(text)  # Expand contractions
    text = normalize_repeats(text)  # Normalize repeated characters
    text = newline_re.sub(' ', text) # Remove newlines
    text = url_re.sub(' URL ', text)  # Replace URLs
    text = email_re.sub(' EMAIL ', text) # Replace emails
    text = number_re.sub(' NUMBER ', text) # Replace numbers
    text = allowed_chars_re.sub(' ', text) # Keep letters, numbers, ! ? ' *
    text = multi_space_re.sub(' ', text).strip() # Remove extra spaces
    return text

train['comment_text'] = train['comment_text'].apply(clean_text)
test['comment_text']  = test['comment_text'].apply(clean_text)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['toxicity'] = train[labels].max(axis=1)

MAX_VOCAB = 50000
MAX_LEN = 200
EMBEDDING_DIM = 300

word2vec = api.load("word2vec-google-news-300")

def balance_and_split(df):
    toxic = df[df['toxicity'] == 1]
    non_toxic = df[df['toxicity'] == 0].sample(len(toxic))  # balance

    df_bal = pd.concat([toxic, non_toxic]).sample(frac=1)   # shuffle

    X = df_bal['comment_text']
    y = df_bal['toxicity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=None       # IMPORTANT
    )
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
    X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')

    return X_train, X_test, y_train, y_test, X_train_seq, X_test_seq, tokenizer

def create_cnn_model(embedding_layer, X_train_seq, y_train, X_test_seq, y_test, input_layer):
    conv3 = Conv1D(128, 3, activation='relu')(embedding_layer)
    pool3 = GlobalMaxPooling1D()(conv3)

    conv4 = Conv1D(128, 4, activation='relu')(embedding_layer)
    pool4 = GlobalMaxPooling1D()(conv4)

    conv5 = Conv1D(128, 5, activation='relu')(embedding_layer)
    pool5 = GlobalMaxPooling1D()(conv5)

    merged = Concatenate()([pool3, pool4, pool5])
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.005, momentum=0.9), metrics=['accuracy'])

    history = model.fit(
        X_train_seq,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test_seq, y_test)
    )
    
    return model, history

def save_model(model, modelName, history, historyName):
    model.save(f'models/{modelName}')

    # Save training history
    joblib.dump(history.history, f'models/{historyName}')
    print(f"Saved: models/{modelName}, models/{historyName}")

def load_classifier_model(modelName, historyName):
    model = load_model(f'models/{modelName}')

    history = joblib.load(f'models/{historyName}')

    return model, history

def calc_specificity_fdr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # ravel converts the multi dimensional array to 1D

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0

    return specificity, fdr

metrics = ['accuracy', 'specificity', 'fdr']
models = ['svm', 'nb', 'lda', 'knn', 'cnn_rand', 'cnn_fix']
model_metrics = {
    'svm': {
        'accuracy': [],
        'specificity': [],
        'fdr': []
    },
    'nb': {
        'accuracy': [],
        'specificity': [],
        'fdr': []
    },
    'lda': {
        'accuracy': [],
        'specificity': [],
        'fdr': []
    },
    'knn': {
        'accuracy': [],
        'specificity': [],
        'fdr': []
    },
    'cnn_rand': {
        'accuracy': [],
        'specificity': [],
        'fdr': []
    },
    'cnn_fix': {
        'accuracy': [],
        'specificity': [],
        'fdr': []
    }
}

for x in range(20):
    X_train, X_test, y_train, y_test, X_train_seq, X_test_seq, tokenizer = balance_and_split(train)

    word_index = tokenizer.word_index
    num_words = min(MAX_VOCAB, len(word_index)+1)

    # CNN rand
    input_layer = Input(shape=(MAX_LEN,))
    embedding_layer_rand = Embedding(num_words, EMBEDDING_DIM, trainable=False)(input_layer)

    # CNN fix
    embedding_index = word2vec
    word_index = tokenizer.word_index
    num_words = min(MAX_VOCAB, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_VOCAB:
            continue

        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]

    embedding_layer_fix = Embedding(
        num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
    )(input_layer)

    # Create and train CNN rand model
    cnn_rand_model, cnn_rand_history = create_cnn_model(embedding_layer_rand, X_train_seq, y_train, X_test_seq, y_test, input_layer)
    cnn_rand_y_pred = (cnn_rand_model.predict(X_test_seq) > 0.5).astype("int32")
    model_metrics['cnn_rand']['accuracy'].append(accuracy_score(y_test, cnn_rand_y_pred))
    cnn_rand_specificity, cnn_rand_fdr = calc_specificity_fdr(y_test, cnn_rand_y_pred)
    model_metrics['cnn_rand']['specificity'].append(cnn_rand_specificity)
    model_metrics['cnn_rand']['fdr'].append(cnn_rand_fdr)

    # Create and train CNN fix model
    cnn_fix_model, cnn_fix_history = create_cnn_model(embedding_layer_fix, X_train_seq, y_train, X_test_seq, y_test, input_layer)
    cnn_fix_y_pred = (cnn_fix_model.predict(X_test_seq) > 0.5).astype("int32")
    model_metrics['cnn_fix']['accuracy'].append(accuracy_score(y_test, cnn_fix_y_pred))
    cnn_fix_specificity, cnn_fix_fdr = calc_specificity_fdr(y_test, cnn_fix_y_pred)
    model_metrics['cnn_fix']['specificity'].append(cnn_fix_specificity)
    model_metrics['cnn_fix']['fdr'].append(cnn_fix_fdr)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # SVM
    svm_model = LinearSVC()
    svm_model.fit(X_train_bow, y_train)
    svm_y_pred = svm_model.predict(X_test_bow)
    model_metrics['svm']['accuracy'].append(accuracy_score(y_test, svm_y_pred))
    svm_specificity, svm_fdr = calc_specificity_fdr(y_test, svm_y_pred)
    model_metrics['svm']['specificity'].append(svm_specificity)
    model_metrics['svm']['fdr'].append(svm_fdr)


    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_bow, y_train)
    model_metrics['nb']['accuracy'].append(accuracy_score(y_test, nb_model.predict(X_test_bow)))
    nb_specificity, nb_fdr = calc_specificity_fdr(y_test, nb_model.predict(X_test_bow))
    model_metrics['nb']['specificity'].append(nb_specificity)
    model_metrics['nb']['fdr'].append(nb_fdr)

    # LDA with SVD
    svd = TruncatedSVD(n_components=200)
    X_train_svd = svd.fit_transform(X_train_bow)
    X_test_svd = svd.transform(X_test_bow)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_svd, y_train)
    model_metrics['lda']['accuracy'].append(accuracy_score(y_test, lda.predict(X_test_svd)))
    lda_specificity, lda_fdr = calc_specificity_fdr(y_test, lda.predict(X_test_svd))
    model_metrics['lda']['specificity'].append(lda_specificity)
    model_metrics['lda']['fdr'].append(lda_fdr)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(X_train_bow, y_train)
    model_metrics['knn']['accuracy'].append(accuracy_score(y_test, knn_model.predict(X_test_bow)))
    knn_specificity, knn_fdr = calc_specificity_fdr(y_test, knn_model.predict(X_test_bow))
    model_metrics['knn']['specificity'].append(knn_specificity)
    model_metrics['knn']['fdr'].append(knn_fdr)

rows = []

for model in model_metrics:
    acc = model_metrics[model]['accuracy']
    spec = model_metrics[model]['specificity']
    fdr = model_metrics[model]['fdr']
    
    rows.append([
        model, 
        np.mean(acc), np.std(acc),
        np.mean(spec), np.std(spec),
        np.mean(fdr), np.std(fdr)
    ])

df_results = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "Accuracy Mean", "Accuracy Std",
        "Specificity Mean", "Specificity Std",
        "Falsedisc.rate Mean", "Falsedisc.rate Std"
    ]
)

order = ["cnn_fix", "cnn_rand", "knn", "lda", "nb", "svm"]
df_results = df_results.set_index("Model").loc[order].reset_index()
df_results = df_results.round(3)

df_results.to_csv('model_performance_summary.csv', index=False)

