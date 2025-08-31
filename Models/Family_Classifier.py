import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class AttackFamilyClassifier:
    def __init__(self):
        # Define the 10 attack families
        self.attack_families = {
            0: "Privilege Escalation",
            1: "Shadow Admin", 
            2: "Persistence",
            3: "Service Abuse Escalation",
            4: "Data Exfiltration",
            5: "Lateral Movement",
            6: "Policy Backdooring",
            7: "DoS / Destructive Actions",
            8: "Secrets/Credential Theft",
            9: "Key Management Abuse"
        }
        
        # Dataset file paths
        self.dataset_files = [
            "Classifier Dataset/privilege_escalation.json",
            "Classifier Dataset/shadow_admin.json",
            "Classifier Dataset/persistence.json", 
            "Classifier Dataset/service_abuse.json",
            "Classifier Dataset/data_exfiltration.json",
            "Classifier Dataset/lateral_movement.json",
            "Classifier Dataset/policy_backdooring.json",
            "Classifier Dataset/pdos_destructive.json",
            "Classifier Dataset/iam_roles_secrets_theft_2000.json",
            "Classifier Dataset/iam_roles_kms_abuse_2000.json"
        ]

    def load_attack_datasets(self):
        """Load all attack family datasets"""
        all_data = []
        labels = []
        
        print("Loading attack family datasets...")
        
        for family_id, filepath in enumerate(self.dataset_files):
            try:
                print(f"Loading {self.attack_families[family_id]}: {filepath}")
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Add family label to each sample
                for item in data:
                    all_data.append(item)
                    labels.append(family_id)
                
                print(f"  Loaded {len(data)} samples")
                
            except FileNotFoundError:
                print(f"  WARNING: File not found - {filepath}")
                continue
            except Exception as e:
                print(f"  ERROR loading {filepath}: {e}")
                continue
        
        print(f"\nTotal samples loaded: {len(all_data)}")
        print(f"Label distribution:")
        for family_id in range(10):
            count = labels.count(family_id)
            if count > 0:
                print(f"  {family_id}: {self.attack_families[family_id]} - {count} samples")
        
        return all_data, np.array(labels)

    def extract_attack_features(self, policy: Dict) -> Dict:
        """Extract features specific to attack classification"""
        
        actions = []
        resources = []
        statements = policy.get('Statement', [])
        if not isinstance(statements, list):
            statements = [statements]
            
        for statement in statements:
            if statement.get('Effect') == 'Allow':
                stmt_actions = statement.get('Action', [])
                if isinstance(stmt_actions, str):
                    stmt_actions = [stmt_actions]
                actions.extend(stmt_actions)
                
                stmt_resources = statement.get('Resource', [])
                if isinstance(stmt_resources, str):
                    stmt_resources = [stmt_resources]
                resources.extend(stmt_resources)
        
        action_set = set(actions)
        
        features = {
            # Basic policy structure
            'num_statements': len(statements),
            'num_actions': len(actions),
            'num_resources': len(resources),
            'num_services': len(set(action.split(':')[0] for action in actions if ':' in action)),
            
            # Wildcard usage
            'has_wildcard_action': any('*' in action for action in actions),
            'has_wildcard_resource': any(resource == '*' for resource in resources),
            'wildcard_action_ratio': sum(1 for action in actions if '*' in action) / len(actions) if actions else 0,
            
            # Service-specific features
            'iam_action_count': sum(1 for action in actions if action.startswith('iam:')),
            'ec2_action_count': sum(1 for action in actions if action.startswith('ec2:')),
            's3_action_count': sum(1 for action in actions if action.startswith('s3:')),
            'lambda_action_count': sum(1 for action in actions if action.startswith('lambda:')),
            'kms_action_count': sum(1 for action in actions if action.startswith('kms:')),
            'secretsmanager_action_count': sum(1 for action in actions if action.startswith('secretsmanager:')),
            'sts_action_count': sum(1 for action in actions if action.startswith('sts:')),
            'dynamodb_action_count': sum(1 for action in actions if action.startswith('dynamodb:')),
            
            # Attack-specific patterns
            'has_pass_role': 'iam:PassRole' in action_set,
            'has_create_user': 'iam:CreateUser' in action_set,
            'has_create_role': 'iam:CreateRole' in action_set,
            'has_attach_policy': any('AttachPolicy' in action for action in actions),
            'has_put_policy': any('PutPolicy' in action for action in actions),
            'has_assume_role': 'sts:AssumeRole' in action_set,
            'has_run_instances': 'ec2:RunInstances' in action_set,
            'has_create_function': 'lambda:CreateFunction' in action_set,
            'has_kms_decrypt': 'kms:Decrypt' in action_set,
            'has_get_secret': 'secretsmanager:GetSecretValue' in action_set,
            'has_delete_actions': any('Delete' in action for action in actions),
            'has_terminate_actions': any('Terminate' in action for action in actions),
            
            # Combination patterns (key for attack family detection)
            'iam_ec2_combo': ('iam:PassRole' in action_set and 'ec2:RunInstances' in action_set),
            'iam_lambda_combo': ('iam:PassRole' in action_set and 'lambda:CreateFunction' in action_set),
            'user_policy_combo': ('iam:CreateUser' in action_set and any('AttachPolicy' in action for action in actions)),
            'access_key_combo': ('iam:CreateUser' in action_set and 'iam:CreateAccessKey' in action_set),
            'kms_secret_combo': ('kms:Decrypt' in action_set and 'secretsmanager:GetSecretValue' in action_set),
            
            # Cross-account indicators
            'has_cross_account': any('arn:aws:' in resource and '::*:' in resource for resource in resources),
            'has_external_assume': 'sts:AssumeRole' in action_set and any('*' in resource for resource in resources),
            
            # Data access patterns
            'has_s3_wildcard': any(action == 's3:*' for action in actions),
            'has_dynamodb_scan': 'dynamodb:Scan' in action_set,
            'has_rds_describe': any('rds:Describe' in action for action in actions),
            
            # Text features for TF-IDF
            'actions_text': ' '.join(actions),
            'resources_text': ' '.join(resources),
            'combined_text': ' '.join(actions + resources)
        }
        
        return features

    def process_all_datasets(self):
        """Process all datasets and extract features"""
        
        # Load datasets
        all_data, labels = self.load_attack_datasets()
        
        if len(all_data) == 0:
            print("ERROR: No data loaded. Check dataset file paths.")
            return None, None, None
        
        print(f"\nExtracting features from {len(all_data)} policies...")
        
        # Extract features
        features_list = []
        valid_indices = []
        
        for i, item in enumerate(all_data):
            try:
                policy = item.get('policy', item)  # Handle different JSON structures
                features = self.extract_attack_features(policy)
                features_list.append(features)
                valid_indices.append(i)
                
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(all_data)} policies")
                    
            except Exception as e:
                print(f"  Warning: Failed to process policy {i}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        valid_labels = labels[valid_indices]
        
        # Handle text features for TF-IDF
        text_columns = ['actions_text', 'resources_text', 'combined_text']
        text_features = df[text_columns].copy()
        df_numeric = df.drop(columns=text_columns)
        
        # Convert boolean columns to int
        bool_columns = df_numeric.select_dtypes(include=[bool]).columns
        df_numeric[bool_columns] = df_numeric[bool_columns].astype(int)
        
        print(f"Feature extraction complete!")
        print(f"Numeric features: {len(df_numeric.columns)}")
        print(f"Samples: {len(df_numeric)}")
        
        return df_numeric, valid_labels, text_features

    def add_tfidf_features(self, df_numeric, text_features, max_features=50):
        """Add TF-IDF features for text analysis"""
        
        print("Adding TF-IDF text features...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            token_pattern=r'[a-zA-Z0-9_:*-]+',
            stop_words=None
        )
        
        tfidf_matrix = vectorizer.fit_transform(text_features['combined_text'].fillna(''))
        tfidf_feature_names = [f'tfidf_{name}' for name in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df_numeric.index)
        
        # Combine features
        combined_df = pd.concat([df_numeric, tfidf_df], axis=1)
        
        print(f"Added {len(tfidf_feature_names)} TF-IDF features")
        print(f"Total features: {len(combined_df.columns)}")
        
        return combined_df, vectorizer

    def train_classifier(self):
        """Train the attack family classifier"""
        
        print("=" * 70)
        print("CIPHERCLOUD ATTACK FAMILY CLASSIFIER TRAINING")
        print("=" * 70)
        
        # Process datasets
        X_numeric, y, text_features = self.process_all_datasets()
        
        if X_numeric is None:
            return
        
        # Add TF-IDF features
        X_full, vectorizer = self.add_tfidf_features(X_numeric, text_features)
        
        # Split data
        print("\nSplitting data for training...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, max_depth=20),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr'),
            'SVM': SVC(random_state=42, probability=True, kernel='rbf')
        }
        
        print("\nTraining models...")
        results = {}
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train
            if name == 'Random Forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_data = X_test
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                test_data = X_test_scaled
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, test_data, y_test, cv=3, scoring='accuracy')
            print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Detailed classification report
            print(f"\nClassification Report for {name}:")
            print("-" * 50)
            target_names = [self.attack_families[i] for i in sorted(self.attack_families.keys())]
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_score': cv_scores.mean(),
                'predictions': y_pred
            }
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = name
        
        print(f"\nBest Model: {best_model} (Accuracy: {best_score:.3f})")
        
        # Feature importance for Random Forest
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            importances = pd.DataFrame({
                'feature': X_full.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 15 Most Important Features (Random Forest):")
            print("-" * 60)
            for idx, row in importances.head(15).iterrows():
                print(f"{row['feature'][:45]:45s} {row['importance']:.4f}")
        
        # Save the best model
        best_model_obj = results[best_model]['model']
        model_data = {
            'model': best_model_obj,
            'scaler': scaler,
            'vectorizer': vectorizer,
            'feature_names': list(X_full.columns),
            'attack_families': self.attack_families,
            'model_type': best_model,
            'accuracy': best_score
        }
        
        joblib.dump(model_data, 'cipher_cloud_family_model.pkl')
        print(f"\nModel saved to: cipher_cloud_family_model.pkl")
        
        # Save processed data for analysis
        X_full.to_csv("cipher_cloud_family_features_X.csv", index=False)
        pd.DataFrame({'label': y}).to_csv("cipher_cloud_family_features_y.csv", index=False)
        
        with open("cipher_cloud_family_feature_names.json", 'w') as f:
            json.dump(list(X_full.columns), f, indent=2)
        
        print("Feature data saved for analysis")
        
        # Plot confusion matrix for best model
        if len(np.unique(y_test)) > 1:
            self.plot_confusion_matrix(y_test, results[best_model]['predictions'], best_model)
        
        return results

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        
        try:
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_true, y_pred)
            
            # Create labels
            labels = [self.attack_families.get(i, f"Class {i}") for i in range(len(cm))]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.title(f'Attack Family Classification - {model_name}')
            plt.xlabel('Predicted Attack Family')
            plt.ylabel('True Attack Family')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('attack_family_confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: attack_family_confusion_matrix.png")
            plt.show()
            
        except Exception as e:
            print(f"Could not generate confusion matrix plot: {e}")

    def test_family_prediction(self, policy: Dict):
        """Test the trained model on a new policy"""
        
        try:
            # Load trained model
            model_data = joblib.load('cipher_cloud_family_model.pkl')
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            attack_families = model_data['attack_families']
            
            # Extract features
            features = self.extract_attack_features(policy)
            feature_df = pd.DataFrame([features])
            
            # Handle text features (simplified - zeros for TF-IDF)
            text_columns = ['actions_text', 'resources_text', 'combined_text']
            if any(col in feature_df.columns for col in text_columns):
                feature_df = feature_df.drop(columns=[col for col in text_columns if col in feature_df.columns])
            
            # Convert boolean to int
            bool_columns = feature_df.select_dtypes(include=[bool]).columns
            feature_df[bool_columns] = feature_df[bool_columns].astype(int)
            
            # Align features
            for feature in feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0
            
            feature_df = feature_df.reindex(columns=feature_names, fill_value=0)
            
            # Scale and predict
            if model_data['model_type'] == 'Random Forest':
                features_for_prediction = feature_df
            else:
                features_for_prediction = scaler.transform(feature_df)
            
            prediction = model.predict(features_for_prediction)[0]
            probabilities = model.predict_proba(features_for_prediction)[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_predictions = []
            
            for idx in top_3_indices:
                if idx in attack_families:
                    top_predictions.append({
                        'family': attack_families[idx],
                        'probability': float(probabilities[idx]),
                        'family_id': int(idx)
                    })
            
            return {
                'primary_attack': attack_families.get(prediction, f"Unknown ({prediction})"),
                'family_id': int(prediction),
                'confidence': float(probabilities[prediction]),
                'top_predictions': top_predictions
            }
            
        except FileNotFoundError:
            print("ERROR: Family model not found. Train the model first.")
            return None
        except Exception as e:
            print(f"ERROR in prediction: {e}")
            return None

def main():
    """Main training function"""
    
    classifier = AttackFamilyClassifier()
    
    print("Starting CipherCloud Attack Family Classifier Training...")
    print("This will train a model to classify risky policies into 10 attack families")
    
    # Check if datasets exist
    missing_files = []
    for filepath in classifier.dataset_files:
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} dataset files not found:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nContinuing with available datasets...")
    
    # Train the classifier
    results = classifier.train_classifier()
    
    if results:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print("Next steps:")
        print("1. Use cipher_cloud_family_model.pkl for attack family classification")
        print("2. Create a family scanner to test the model")
        print("3. Integrate with your binary classifier for full pipeline")
        
        # Test with example policy
        print("\n" + "=" * 70)
        print("TESTING WITH EXAMPLE POLICY")
        print("=" * 70)
        
        example_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["iam:PassRole", "ec2:RunInstances"],
                "Resource": "*"
            }]
        }
        
        result = classifier.test_family_prediction(example_policy)
        if result:
            print(f"Primary Attack Family: {result['primary_attack']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print("Top 3 Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"  {i}. {pred['family']} ({pred['probability']:.1%})")

if __name__ == "__main__":
    main()