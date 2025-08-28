import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class IAMFeatureExtractor:
    def __init__(self):
        # Define high-risk actions and services
        self.high_risk_actions = {
            'iam:PassRole', 'iam:CreateUser', 'iam:AttachUserPolicy', 'iam:CreateAccessKey',
            'iam:UpdateLoginProfile', 'iam:CreatePolicyVersion', 'iam:AttachRolePolicy',
            'iam:UpdateAssumeRolePolicy', 'iam:PutUserPolicy', 'iam:CreatePolicy',
            'iam:DeleteUser', 'iam:DeleteRole', 'iam:DetachUserPolicy',
            'kms:Decrypt', 'kms:CreateKey', 'kms:ScheduleKeyDeletion',
            'secretsmanager:GetSecretValue', 'secretsmanager:CreateSecret',
            'sts:AssumeRole', 'ec2:RunInstances', 'lambda:CreateFunction',
            'lambda:UpdateFunctionCode', 'dynamodb:DeleteTable'
        }
        
        self.admin_equivalent_actions = {'*'}
        
        self.privilege_escalation_combos = [
            ('iam:PassRole', 'ec2:RunInstances'),
            ('iam:PassRole', 'lambda:CreateFunction'),
            ('iam:CreateUser', 'iam:AttachUserPolicy'),
            ('iam:CreatePolicyVersion', 'iam:AttachRolePolicy')
        ]
        
        self.aws_services = {
            'iam', 's3', 'ec2', 'lambda', 'dynamodb', 'kms', 'secretsmanager',
            'sts', 'rds', 'cloudwatch', 'sns', 'sqs', 'logs', 'ssm', 'glue',
            'ecs', 'eks', 'cloudformation', 'apigateway'
        }

    def extract_actions_from_policy(self, policy: Dict) -> List[str]:
        """Extract all actions from a policy"""
        actions = []
        statements = policy.get('Statement', [])
        if not isinstance(statements, list):
            statements = [statements]
            
        for statement in statements:
            if statement.get('Effect') == 'Allow':
                stmt_actions = statement.get('Action', [])
                if isinstance(stmt_actions, str):
                    stmt_actions = [stmt_actions]
                actions.extend(stmt_actions)
                
        return actions

    def extract_resources_from_policy(self, policy: Dict) -> List[str]:
        """Extract all resources from a policy"""
        resources = []
        statements = policy.get('Statement', [])
        if not isinstance(statements, list):
            statements = [statements]
            
        for statement in statements:
            if statement.get('Effect') == 'Allow':
                stmt_resources = statement.get('Resource', [])
                if isinstance(stmt_resources, str):
                    stmt_resources = [stmt_resources]
                resources.extend(stmt_resources)
                
        return resources

    def extract_basic_features(self, policy: Dict) -> Dict[str, Any]:
        """Extract basic statistical features from policy"""
        actions = self.extract_actions_from_policy(policy)
        resources = self.extract_resources_from_policy(policy)
        statements = policy.get('Statement', [])
        if not isinstance(statements, list):
            statements = [statements]
            
        features = {
            # Basic counts
            'num_statements': len(statements),
            'num_actions': len(actions),
            'num_resources': len(resources),
            'num_unique_actions': len(set(actions)),
            'num_unique_resources': len(set(resources)),
            
            # Wildcard indicators
            'has_wildcard_action': any('*' in action for action in actions),
            'has_wildcard_resource': any(resource == '*' for resource in resources),
            'has_service_wildcard': any(action.endswith(':*') for action in actions),
            
            # Admin equivalent
            'has_admin_action': any(action in self.admin_equivalent_actions for action in actions),
            
            # Service diversity
            'num_services': len(set(action.split(':')[0] for action in actions if ':' in action)),
        }
        
        return features

    def extract_risk_features(self, policy: Dict) -> Dict[str, Any]:
        """Extract risk-specific features"""
        actions = self.extract_actions_from_policy(policy)
        resources = self.extract_resources_from_policy(policy)
        action_set = set(actions)
        
        features = {
            # High-risk action counts
            'num_high_risk_actions': sum(1 for action in actions if action in self.high_risk_actions),
            'has_iam_actions': any(action.startswith('iam:') for action in actions),
            'has_kms_actions': any(action.startswith('kms:') for action in actions),
            'has_secrets_actions': any(action.startswith('secretsmanager:') for action in actions),
            
            # Privilege escalation indicators
            'has_pass_role': 'iam:PassRole' in action_set,
            'has_create_user': 'iam:CreateUser' in action_set,
            'has_attach_policy': any('AttachPolicy' in action for action in actions),
            'has_assume_role': 'sts:AssumeRole' in action_set,
            
            # Privilege escalation combinations
            'privesc_combo_1': ('iam:PassRole' in action_set and 'ec2:RunInstances' in action_set),
            'privesc_combo_2': ('iam:PassRole' in action_set and 'lambda:CreateFunction' in action_set),
            'privesc_combo_3': ('iam:CreateUser' in action_set and any('AttachPolicy' in action for action in actions)),
            
            # Resource scope risks
            'all_resources_wildcard': all(resource == '*' for resource in resources) if resources else False,
            'has_cross_account_resource': any('::*:' in resource for resource in resources),
            
            # Persistence indicators
            'has_create_access_key': 'iam:CreateAccessKey' in action_set,
            'has_update_login_profile': 'iam:UpdateLoginProfile' in action_set,
            
            # Data exfiltration risks
            'has_s3_wildcard': any(action == 's3:*' for action in actions),
            'has_dynamodb_scan': 'dynamodb:Scan' in action_set,
            'has_kms_decrypt': 'kms:Decrypt' in action_set,
        }
        
        return features

    def extract_condition_features(self, policy: Dict) -> Dict[str, Any]:
        """Extract features related to conditions"""
        statements = policy.get('Statement', [])
        if not isinstance(statements, list):
            statements = [statements]
            
        has_conditions = any('Condition' in stmt for stmt in statements)
        num_conditions = sum(len(stmt.get('Condition', {})) for stmt in statements)
        
        # Check for common security conditions
        has_mfa_condition = False
        has_ip_condition = False
        has_time_condition = False
        
        for stmt in statements:
            conditions = stmt.get('Condition', {})
            for condition_type, condition_values in conditions.items():
                if 'MFA' in str(condition_values).upper():
                    has_mfa_condition = True
                if 'IpAddress' in condition_type:
                    has_ip_condition = True
                if 'Date' in condition_type or 'Time' in condition_type:
                    has_time_condition = True
        
        return {
            'has_conditions': has_conditions,
            'num_conditions': num_conditions,
            'has_mfa_condition': has_mfa_condition,
            'has_ip_condition': has_ip_condition,
            'has_time_condition': has_time_condition,
        }

    def extract_text_features(self, policy: Dict) -> Dict[str, Any]:
        """Extract text-based features for TF-IDF"""
        actions = self.extract_actions_from_policy(policy)
        resources = self.extract_resources_from_policy(policy)
        
        # Create text representations
        actions_text = ' '.join(actions)
        resources_text = ' '.join(resources)
        combined_text = actions_text + ' ' + resources_text
        
        return {
            'actions_text': actions_text,
            'resources_text': resources_text,
            'combined_text': combined_text
        }

    def extract_all_features(self, policy: Dict) -> Dict[str, Any]:
        """Extract all features from a single policy"""
        features = {}
        features.update(self.extract_basic_features(policy))
        features.update(self.extract_risk_features(policy))
        features.update(self.extract_condition_features(policy))
        features.update(self.extract_text_features(policy))
        
        return features

    def process_dataset(self, dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Process entire dataset and return feature matrix and labels
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Extract features for all policies
        features_list = []
        labels = []
        
        print(f"Processing {len(data)} policies...")
        
        for i, item in enumerate(data):
            if i % 200 == 0:
                print(f"Processed {i}/{len(data)} policies")
                
            policy = item['policy']
            label = item['label']
            
            features = self.extract_all_features(policy)
            features_list.append(features)
            labels.append(label)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Handle text features separately for TF-IDF
        text_columns = ['actions_text', 'resources_text', 'combined_text']
        text_features = df[text_columns].copy()
        
        # Drop text columns from main DataFrame for now
        df_numeric = df.drop(columns=text_columns)
        
        # Convert boolean columns to int
        bool_columns = df_numeric.select_dtypes(include=[bool]).columns
        df_numeric[bool_columns] = df_numeric[bool_columns].astype(int)
        
        print(f"Extracted {len(df_numeric.columns)} numeric features")
        print(f"Feature names: {list(df_numeric.columns)}")
        
        return df_numeric, np.array(labels), text_features

    def add_tfidf_features(self, df_numeric: pd.DataFrame, text_features: pd.DataFrame, 
                          max_features: int = 100) -> pd.DataFrame:
        """Add TF-IDF features to the numeric feature matrix"""
        
        # TF-IDF on combined text (actions + resources)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,  # Don't remove AWS action words
            ngram_range=(1, 2),  # Unigrams and bigrams
            token_pattern=r'[a-zA-Z0-9_:*-]+',  # AWS-specific tokens
        )
        
        tfidf_matrix = vectorizer.fit_transform(text_features['combined_text'].fillna(''))
        
        # Create TF-IDF DataFrame
        tfidf_feature_names = [f'tfidf_{name}' for name in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df_numeric.index)
        
        # Combine with numeric features
        combined_df = pd.concat([df_numeric, tfidf_df], axis=1)
        
        print(f"Added {len(tfidf_feature_names)} TF-IDF features")
        print(f"Total features: {len(combined_df.columns)}")
        
        return combined_df

    def save_processed_data(self, X: pd.DataFrame, y: np.ndarray, 
                          filename_prefix: str = "cipher_cloud_features"):
        """Save processed features and labels"""
        
        # Save features
        X.to_csv(f"{filename_prefix}_X.csv", index=False)
        
        # Save labels
        pd.DataFrame({'label': y}).to_csv(f"{filename_prefix}_y.csv", index=False)
        
        # Save feature names for later reference
        with open(f"{filename_prefix}_feature_names.json", 'w') as f:
            json.dump(list(X.columns), f, indent=2)
        
        print(f"Saved features to {filename_prefix}_X.csv")
        print(f"Saved labels to {filename_prefix}_y.csv")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: Benign={sum(y==0)}, Risky={sum(y==1)}")

# Example usage and ML pipeline
def create_ml_pipeline():
    """Process features and save them for model training/evaluation elsewhere."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Initialize feature extractor
    extractor = IAMFeatureExtractor()

    # Process dataset
    X_numeric, y, text_features = extractor.process_dataset("Dataset/cipher_cloud_dataset.json")

    # Add TF-IDF features
    X_full = extractor.add_tfidf_features(X_numeric, text_features, max_features=50)

    # Split data (optional, but no evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (optional, but no evaluation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    extractor.save_processed_data(X_full, y, "cipher_cloud_features")

    return X_full, y, extractor

if __name__ == "__main__":
    # Only process and save features, no model training/evaluation
    X_full, y, extractor = create_ml_pipeline()