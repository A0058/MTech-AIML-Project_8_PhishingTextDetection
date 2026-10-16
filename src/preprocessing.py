import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split

REPLACEMENTS = {
    "urgent": ["important", "time sensitive"],
    "immediately": ["right away", "without delay"],
    "verify": ["confirm", "validate"],
    "verification": ["confirmation", "validation"],
    "click": ["open", "select"],
    "account": ["profile", "user account"],
    "password": ["passcode", "login credential"],
    "suspended": ["restricted", "temporarily disabled"],
    "update": ["refresh", "revise"],
    "confirm": ["validate", "check"],
    "security": ["protection", "account security"],
    "login": ["sign in", "account access"],
    "free": ["complimentary", "no cost"],
    "winner": ["selected recipient", "prize winner"],
    "prize": ["reward", "award"],
    "payment": ["transaction", "billing"],
    "invoice": ["bill", "payment statement"],
    "limited": ["restricted", "short term"],
    "expire": ["end", "become invalid"],
    "bank": ["financial institution", "banking service"],
    "link": ["web address", "access link"]
}

def load_dataset(path):
    df = pd.read_csv(path)
    required = {"text_combined", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required}; found {df.columns.tolist()}"
        )
    df = df[["text_combined", "label"]].dropna().copy()
    df["text_combined"] = df["text_combined"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[
        df["text_combined"].str.strip().ne("")
        & df["label"].isin([0, 1])
    ]
    if df.empty:
        raise ValueError("No usable data remained after cleaning.")
    return df.reset_index(drop=True)

def split_dataset(df, test_size=0.20, random_state=42, max_test_samples=4000):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_combined"], df["label"],
        test_size=test_size, random_state=random_state,
        stratify=df["label"]
    )
    if len(X_test) > max_test_samples:
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=max_test_samples,
            random_state=random_state, stratify=y_test
        )
    return tuple(
        series.reset_index(drop=True)
        for series in (X_train, X_test, y_train, y_test)
    )

def modify_message(text, seed=42):
    """Create a reproducible synthetic lexical modification."""
    rng = random.Random(seed)
    modified = str(text).lower()
    words = list(REPLACEMENTS)
    rng.shuffle(words)
    changed = 0

    for word in words:
        pattern = r"\\b" + re.escape(word) + r"\\b"
        if re.search(pattern, modified):
            modified = re.sub(
                pattern, rng.choice(REPLACEMENTS[word]),
                modified, count=1
            )
            changed += 1
            if changed >= 3:
                break

    if rng.random() < 0.35:
        modified = re.sub(
            r"\\b(the|a|an|please|kindly)\\b", "", modified, count=1
        )
    return re.sub(r"\\s+", " ", modified).strip()

def make_modified_test_set(X_test, random_state=42):
    return pd.Series([
        modify_message(text, random_state + i)
        for i, text in enumerate(X_test.reset_index(drop=True))
    ])
