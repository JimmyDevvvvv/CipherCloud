import json
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any
import sys
import os

# Feature extraction class (copied from your Binary.py)
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

    def extract_actions_from_policy(self, policy):
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

    def extract_resources_from_policy(self, policy):
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

    def extract_all_features(self, policy):
        actions = self.extract_actions_from_policy(policy)
        resources = self.extract_resources_from_policy(policy)
        statements = policy.get('Statement', [])
        if not isinstance(statements, list):
            statements = [statements]
        
        action_set = set(actions)
        
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
            
            # Risk features
            'num_high_risk_actions': sum(1 for action in actions if action in self.high_risk_actions),
            'has_iam_actions': any(action.startswith('iam:') for action in actions),
            'has_kms_actions': any(action.startswith('kms:') for action in actions),
            'has_secrets_actions': any(action.startswith('secretsmanager:') for action in actions),
            
            # Privilege escalation
            'has_pass_role': 'iam:PassRole' in action_set,
            'has_create_user': 'iam:CreateUser' in action_set,
            'has_attach_policy': any('AttachPolicy' in action for action in actions),
            'has_assume_role': 'sts:AssumeRole' in action_set,
            
            'privesc_combo_1': ('iam:PassRole' in action_set and 'ec2:RunInstances' in action_set),
            'privesc_combo_2': ('iam:PassRole' in action_set and 'lambda:CreateFunction' in action_set),
            'privesc_combo_3': ('iam:CreateUser' in action_set and any('AttachPolicy' in action for action in actions)),
            
            # Resource risks
            'all_resources_wildcard': all(resource == '*' for resource in resources) if resources else False,
            'has_cross_account_resource': any('::*:' in resource for resource in resources),
            
            # Other risks
            'has_create_access_key': 'iam:CreateAccessKey' in action_set,
            'has_update_login_profile': 'iam:UpdateLoginProfile' in action_set,
            'has_s3_wildcard': any(action == 's3:*' for action in actions),
            'has_dynamodb_scan': 'dynamodb:Scan' in action_set,
            'has_kms_decrypt': 'kms:Decrypt' in action_set,
            
            # Condition features
            'has_conditions': any('Condition' in stmt for stmt in statements),
            'num_conditions': sum(len(stmt.get('Condition', {})) for stmt in statements),
            
            # Text features (simplified)
            'actions_text': ' '.join(actions),
            'resources_text': ' '.join(resources),
            'combined_text': ' '.join(actions + resources)
        }
        
        return features

class CipherCloudScanner:
    def __init__(self, model_path: str = "Models/cipher_cloud_binary_model.pkl"):
        """Initialize the scanner with your trained model"""
        
        # Load the trained model
        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.feature_names = self.model_data['feature_names']
            print(f"Model loaded: {type(self.model).__name__}")
            print(f"Features: {len(self.feature_names)}")
        except FileNotFoundError:
            print(f"ERROR: Model file not found: {model_path}")
            print("Run Binary.py first to train the model")
            sys.exit(1)
        
        # Initialize feature extractor
        self.extractor = IAMFeatureExtractor()

    def scan_policy(self, policy: Dict) -> Dict[str, Any]:
        """Scan a single IAM policy"""
        
        # Extract features
        features = self.extractor.extract_all_features(policy)
        feature_df = pd.DataFrame([features])
        
        # Handle text features (drop for now - simplified approach)
        text_columns = ['actions_text', 'resources_text', 'combined_text']
        if any(col in feature_df.columns for col in text_columns):
            feature_df = feature_df.drop(columns=[col for col in text_columns if col in feature_df.columns])
        
        # Convert boolean to int
        bool_columns = feature_df.select_dtypes(include=[bool]).columns
        feature_df[bool_columns] = feature_df[bool_columns].astype(int)
        
        # Align with training features (add missing TF-IDF features as zeros)
        current_features = set(feature_df.columns)
        expected_features = set(self.feature_names)
        missing_features = expected_features - current_features
        
        for feature in missing_features:
            feature_df[feature] = 0
        
        # Reorder columns to match training
        feature_df = feature_df.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale and predict
        features_scaled = self.scaler.transform(feature_df)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get risk indicators (top contributing features)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            feature_values = features_scaled[0]
            risk_contributions = feature_importance * np.abs(feature_values)
            
            # Get top risk indicators
            top_indices = np.argsort(risk_contributions)[-8:][::-1]
            risk_indicators = []
            
            for idx in top_indices:
                if risk_contributions[idx] > 0.001:
                    feature_name = self.feature_names[idx]
                    # Clean up feature names for display
                    clean_name = feature_name.replace('tfidf_', '').replace('_', ' ').title()
                    if feature_name.startswith('has_') or feature_name.startswith('all_'):
                        clean_name = feature_name.replace('_', ' ').title()
                    
                    risk_indicators.append({
                        'indicator': clean_name,
                        'score': float(risk_contributions[idx]),
                        'raw_feature': feature_name,
                        'value': float(feature_values[idx])
                    })
        else:
            risk_indicators = []
        
        return {
            'prediction': 'RISKY' if prediction == 1 else 'BENIGN',
            'confidence': float(probabilities[1] if prediction == 1 else probabilities[0]),
            'risk_probability': float(probabilities[1]),
            'benign_probability': float(probabilities[0]),
            'risk_indicators': risk_indicators
        }

    def format_policy_summary(self, policy: Dict) -> str:
        """Create a human-readable summary of the policy"""
        actions = []
        resources = []
        
        statements = policy.get('Statement', [])
        if not isinstance(statements, list):
            statements = [statements]
        
        for stmt in statements:
            if stmt.get('Effect') == 'Allow':
                stmt_actions = stmt.get('Action', [])
                if isinstance(stmt_actions, str):
                    stmt_actions = [stmt_actions]
                actions.extend(stmt_actions)
                
                stmt_resources = stmt.get('Resource', [])
                if isinstance(stmt_resources, str):
                    stmt_resources = [stmt_resources]
                resources.extend(stmt_resources)
        
        # Summarize
        action_summary = f"{len(actions)} actions"
        if len(actions) <= 3:
            action_summary += f": {', '.join(actions)}"
        elif any('*' in action for action in actions):
            action_summary += " (includes wildcards)"
        
        resource_summary = f"{len(resources)} resources"
        if resources and resources[0] == '*':
            resource_summary = "All resources (*)"
        elif len(resources) <= 2:
            resource_summary += f": {', '.join(resources)}"
        
        return f"{action_summary} | {resource_summary}"

    def scan_and_display(self, policy: Dict):
        """Scan a policy and display formatted results"""
        
        print("\n" + "="*60)
        print("CIPHERCLOUD POLICY SCANNER")
        print("="*60)
        
        # Policy summary
        summary = self.format_policy_summary(policy)
        print(f"Policy Summary: {summary}")
        print("-" * 60)
        
        # Scan the policy
        try:
            result = self.scan_policy(policy)
            
            # Display prediction
            prediction = result['prediction']
            confidence = result['confidence']
            
            if prediction == 'RISKY':
                print(f"RESULT: {prediction} (confidence: {confidence:.1%})")
                print("WARNING: This policy has been flagged as potentially risky")
            else:
                print(f"RESULT: {prediction} (confidence: {confidence:.1%})")
                print("INFO: This policy appears to follow security best practices")
            
            print(f"\nProbability Breakdown:")
            print(f"  Risky:  {result['risk_probability']:.1%}")
            print(f"  Benign: {result['benign_probability']:.1%}")
            
            # Show risk indicators if any
            if result['risk_indicators']:
                print(f"\nRisk Indicators Detected:")
                for i, indicator in enumerate(result['risk_indicators'][:6], 1):
                    print(f"  {i}. {indicator['indicator']}")
            
            print("="*60)
            
        except Exception as e:
            print(f"ERROR: Failed to scan policy - {e}")

def main():
    """Interactive scanner"""
    scanner = CipherCloudScanner()
    
    print("CipherCloud Policy Scanner Ready!")
    print("\nPaste your IAM policy JSON and press Enter twice to scan.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            print("Enter IAM Policy JSON:")
            print("-" * 30)
            
            lines = []
            while True:
                line = input()
                if line.strip().lower() == 'quit':
                    print("Goodbye!")
                    return
                if line.strip() == "" and lines:
                    break
                lines.append(line)
            
            if lines:
                policy_json = "\n".join(lines)
                try:
                    policy = json.loads(policy_json)
                    scanner.scan_and_display(policy)
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON format - {e}")
                    print("Please check your policy syntax and try again.")
                
                print("\n" + "="*60)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"ERROR: {e}")

def scan_example():
    """Scan some example policies"""
    scanner = CipherCloudScanner()
    
    # Example risky policy
    risky_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "*",
                "Resource": "*"
            }
        ]
    }
    
    print("Scanning example RISKY policy:")
    scanner.scan_and_display(risky_policy)
    
    # Example benign policy
    benign_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject"
                ],
                "Resource": "arn:aws:s3:::my-bucket/*"
            }
        ]
    }
    
    print("\n\nScanning example BENIGN policy:")
    scanner.scan_and_display(benign_policy)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        scan_example()
    else:
        main()