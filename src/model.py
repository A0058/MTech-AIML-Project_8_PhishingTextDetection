from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def build_logistic_regression():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), max_features=100000,
            min_df=2, max_df=0.98, sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ))
    ])

def build_linear_svm():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), max_features=100000,
            min_df=2, max_df=0.98, sublinear_tf=True
        )),
        ("classifier", LinearSVC(
            class_weight="balanced", random_state=42
        ))
    ])

def train_embedding_classifier(X_train, y_train, max_train_samples=15000):
    """Train a Logistic Regression classifier on pretrained sentence embeddings."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer

    if len(X_train) > max_train_samples:
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train, train_size=max_train_samples,
            random_state=42, stratify=y_train
        )
    else:
        X_fit, y_fit = X_train, y_train

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    train_embeddings = encoder.encode(
        X_fit.astype(str).str.slice(0, 3000).tolist(),
        batch_size=32, show_progress_bar=True,
        normalize_embeddings=True
    )
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    classifier.fit(train_embeddings, y_fit)
    return encoder, classifier

def predict_embeddings(encoder, classifier, texts):
    embeddings = encoder.encode(
        texts.astype(str).str.slice(0, 3000).tolist(),
        batch_size=32, show_progress_bar=True,
        normalize_embeddings=True
    )
    return classifier.predict(embeddings)
