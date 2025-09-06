import os, json, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

from Scanners.Complete_Scanner import AttackFamilyFeatureExtractor

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'Classifier Dataset')

extractor = AttackFamilyFeatureExtractor()
x, y = [], []

for filename in os.listdir(DATASET_DIR):
    if filename.endswith(".json") and "feature" not in filename and "csv" not in filename:
        family_name = filename.replace(".json", "")
        file_path = os.path.join(DATASET_DIR, filename)
        with open(file_path, 'r') as f:
            policies = json.load(f)
            for policy in policies:
                policy_obj = policy.get("policy", policy) if isinstance(policy, dict) else None
                if isinstance(policy_obj, dict):
                    features = extractor.extract_all_features(policy_obj)
                    x.append(features)
                    y.append(family_name)

df = pd.DataFrame(x)

# Handle booleans
for col in df.select_dtypes(include=[bool]).columns:
    df[col] = df[col].astype(int)

# TF-IDF for actions_text
tfidf = TfidfVectorizer(
    max_features=2000,   # much larger vocab
    ngram_range=(1,1),   # unigrams + bigrams
    min_df=2             # ignore ultra-rare terms
)

tfidf_features = tfidf.fit_transform(df["actions_text"].fillna("")).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])])

# Merge structured + tfidf
df = pd.concat([df.drop(columns=["actions_text", "resources_text"]), tfidf_df], axis=1)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale numeric features, but keep DataFrame with names
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print("✅ Train Accuracy:", model.score(X_train, y_train))
print("✅ Test Accuracy:", model.score(X_test, y_test))
print("✅ CV Accuracy:", cv_scores.mean())
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

# Save
joblib.dump({
    "model": model,
    "scaler": scaler,
    "label_encoder": le,
    "tfidf": tfidf,
    "feature_names": list(df.columns),
    "attack_families": list(le.classes_),
    "model_type": "RandomForestClassifier",
    "cv_accuracy": cv_scores.mean()
}, "Models/cipher_cloud_family_model.pkl")