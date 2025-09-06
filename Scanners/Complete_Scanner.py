import json
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List
import sys
import os
from datetime import datetime
import time

# Binary Feature Extractor (from your Binary.py)
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

# Family Feature Extractor (attack-specific patterns)
class AttackFamilyFeatureExtractor:
    def __init__(self):
        # Attack-specific feature patterns
        self.attack_patterns = {
            'privilege_escalation': ['iam:PassRole', 'iam:CreateUser', 'iam:AttachUserPolicy'],
            'shadow_admin': ['iam:CreatePolicyVersion', 'iam:CreateRole'],
            'persistence': ['iam:CreateUser', 'iam:CreateAccessKey'],
            'data_exfiltration': ['s3:GetObject', 's3:ListBucket'],
            'lateral_movement': ['sts:AssumeRole', 'ec2:DescribeInstances'],
            'kms_abuse': ['kms:Decrypt', 'kms:CreateKey'],
            'secrets_theft': ['secretsmanager:GetSecretValue', 'ssm:GetParameter']
        }

    def extract_all_features(self, policy):
        """Extract features for attack family classification"""
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
            # Basic counts
            'num_statements': len(statements),
            'num_actions': len(actions),
            'num_resources': len(resources),
            'num_services': len(set(action.split(':')[0] for action in actions if ':' in action)),
            
            # Service-specific action counts
            'iam_action_count': sum(1 for action in actions if action.startswith('iam:')),
            'ec2_action_count': sum(1 for action in actions if action.startswith('ec2:')),
            's3_action_count': sum(1 for action in actions if action.startswith('s3:')),
            'kms_action_count': sum(1 for action in actions if action.startswith('kms:')),
            'lambda_action_count': sum(1 for action in actions if action.startswith('lambda:')),
            'sts_action_count': sum(1 for action in actions if action.startswith('sts:')),
            
            # Attack-specific indicators
            'has_pass_role': 'iam:PassRole' in action_set,
            'has_create_user': 'iam:CreateUser' in action_set,
            'has_create_role': 'iam:CreateRole' in action_set,
            'has_assume_role': 'sts:AssumeRole' in action_set,
            'has_delete_actions': any('Delete' in action for action in actions),
            'has_create_actions': any('Create' in action for action in actions),
            'has_wildcard_resource': any(resource == '*' for resource in resources),
            'has_cross_account': any('::*:' in resource for resource in resources),
            
            # Text for TF-IDF
            'actions_text': ' '.join(actions),
            'resources_text': ' '.join(resources)
        }
        
        return features

class CompleteCipherCloudScanner:
    def __init__(self, 
                 binary_model_path: str = "Models/cipher_cloud_binary_model.pkl",
                 family_model_path: str = "Models/cipher_cloud_family_model.pkl"):
        """Initialize both binary and family models"""
        
        print("🔄 Loading CipherCloud AI Models...")
        
        # Load binary model
        try:
            self.binary_model_data = joblib.load(binary_model_path)
            self.binary_model = self.binary_model_data['model']
            self.binary_scaler = self.binary_model_data['scaler']
            self.binary_features = self.binary_model_data['feature_names']
            print(f"✅ Binary Model Loaded: {type(self.binary_model).__name__}")
        except FileNotFoundError:
            print(f"❌ Binary model not found: {binary_model_path}")
            sys.exit(1)
        
        # Load family model
        try:
            self.family_model_data = joblib.load(family_model_path)
            self.family_model = self.family_model_data['model']
            self.family_scaler = self.family_model_data['scaler']
            self.family_features = self.family_model_data['feature_names']
            self.attack_families = self.family_model_data['label_encoder'].classes_
            print(f"✅ Family Model Loaded: {type(self.family_model).__name__}")
            print(f"✅ Attack Families: {len(self.attack_families)} types loaded")
        except FileNotFoundError:
            print(f"❌ Family model not found: {family_model_path}")
            print("Note: Only binary classification will be available")
            self.family_model = None
        
        # Initialize extractors
        self.binary_extractor = IAMFeatureExtractor()
        self.family_extractor = AttackFamilyFeatureExtractor()
        
        print("🛡️ CipherCloud Scanner Ready!")

    def binary_scan(self, policy: Dict) -> Dict[str, Any]:
        """Stage 1: Binary risk detection"""
        
        # Extract binary features
        features = self.binary_extractor.extract_all_features(policy)
        feature_df = pd.DataFrame([features])
        
        # Handle text features (drop for simplified approach)
        text_columns = ['actions_text', 'resources_text', 'combined_text']
        feature_df = feature_df.drop(columns=[col for col in text_columns if col in feature_df.columns], errors='ignore')
        
        # Convert boolean to int
        bool_columns = feature_df.select_dtypes(include=[bool]).columns
        feature_df[bool_columns] = feature_df[bool_columns].astype(int)
        
        # Align with training features
        for feature in self.binary_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0
        
        feature_df = feature_df.reindex(columns=self.binary_features, fill_value=0)
        
        # Scale and predict
        features_scaled = self.binary_scaler.transform(feature_df)
        prediction = self.binary_model.predict(features_scaled)[0]
        probabilities = self.binary_model.predict_proba(features_scaled)[0]
        
        return {
            'is_risky': bool(prediction == 1),
            'risk_probability': float(probabilities[1]),
            'benign_probability': float(probabilities[0]),
            'confidence': float(probabilities[1] if prediction == 1 else probabilities[0])
        }

    def family_scan(self, policy: Dict) -> Dict[str, Any]:
        """Stage 2: Attack family classification"""
        
        if self.family_model is None:
            return {'error': 'Family model not available'}
        
        # Extract family features
        features = self.family_extractor.extract_all_features(policy)
        feature_df = pd.DataFrame([features])
        
        # Handle text features (drop for simplified approach)
        text_columns = ['actions_text', 'resources_text']
        feature_df = feature_df.drop(columns=[col for col in text_columns if col in feature_df.columns], errors='ignore')
        
        # Convert boolean to int
        bool_columns = feature_df.select_dtypes(include=[bool]).columns
        feature_df[bool_columns] = feature_df[bool_columns].astype(int)
        
        # Align with training features
        for feature in self.family_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0
        
        feature_df = feature_df.reindex(columns=self.family_features, fill_value=0)
        
        # Scale and predict
        features_scaled = self.family_scaler.transform(feature_df)
        prediction = self.family_model.predict(features_scaled)[0]
        probabilities = self.family_model.predict_proba(features_scaled)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                'family': self.attack_families[idx],
                'probability': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        return {
            'primary_attack_family': self.attack_families[prediction],
            'confidence': float(probabilities[prediction]),
            'top_3_predictions': top_3_predictions
        }

    def complete_scan(self, policy: Dict) -> Dict[str, Any]:
        """Complete CipherCloud scan: Binary + Family analysis"""
        
        print("🔍 Stage 1: Risk Assessment...")
        time.sleep(0.5)  # Visual delay
        
        # Stage 1: Binary classification
        binary_result = self.binary_scan(policy)
        
        result = {
            'policy_summary': self.format_policy_summary(policy),
            'binary_result': binary_result,
            'scan_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Stage 2: Family classification (only if risky)
        if binary_result['is_risky']:
            print("🚨 Risky policy detected! Running attack family analysis...")
            time.sleep(0.5)
            
            family_result = self.family_scan(policy)
            result['family_result'] = family_result
        else:
            print("✅ Policy appears benign - skipping family analysis")
            result['family_result'] = None
        
        return result

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
        
        # Summarize actions
        action_summary = f"{len(actions)} actions"
        if len(actions) <= 3:
            action_summary += f": {', '.join(actions)}"
        elif any('*' in action for action in actions):
            action_summary += " (includes wildcards)"
        
        # Summarize resources
        resource_summary = f"{len(resources)} resources"
        if resources and resources[0] == '*':
            resource_summary = "All resources (*)"
        elif len(resources) <= 2 and resources:
            resource_summary += f": {', '.join(resources[:2])}"
        
        return f"{action_summary} | {resource_summary}"

    def display_results(self, result: Dict[str, Any]):
        """Display formatted scan results"""
        
        print("\n" + "="*70)
        print("🛡️  CIPHERCLOUD COMPLETE SECURITY ANALYSIS")
        print("="*70)
        print(f"📊 Policy Summary: {result['policy_summary']}")
        print(f"⏰ Scan Time: {result['scan_timestamp']}")
        print("-" * 70)
        
        # Binary results
        binary = result['binary_result']
        
        if binary['is_risky']:
            print(f"🚨 SECURITY ASSESSMENT: RISKY POLICY")
            print(f"   Risk Level: {binary['risk_probability']:.1%}")
            print(f"   Confidence: {binary['confidence']:.1%}")
            
            # Family results
            family = result['family_result']
            if family and 'primary_attack_family' in family:
                print(f"\n🎯 ATTACK FAMILY CLASSIFICATION:")
                print(f"   Primary Attack Type: {family['primary_attack_family']}")
                print(f"   Classification Confidence: {family['confidence']:.1%}")
                
                print(f"\n📈 Top Attack Family Predictions:")
                for i, pred in enumerate(family['top_3_predictions'], 1):
                    percentage = pred['probability'] * 100
                    bar_length = int(percentage / 5)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"   {i}. {pred['family']:<25} {percentage:5.1f}% |{bar}|")
            
        else:
            print(f"✅ SECURITY ASSESSMENT: BENIGN POLICY")
            print(f"   Safety Score: {binary['benign_probability']:.1%}")
            print(f"   Confidence: {binary['confidence']:.1%}")
            print(f"\n💚 This policy follows AWS security best practices")
        
        print("="*70)

def main():
    """Interactive complete scanner"""
    
    # ASCII Art Header
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ██████╗██╗██████╗ ██╗  ██╗███████╗██████╗  ██████╗██╗      ║
    ║  ██╔════╝██║██╔══██╗██║  ██║██╔════╝██╔══██╗██╔════╝██║      ║
    ║  ██║     ██║██████╔╝███████║█████╗  ██████╔╝██║     ██║      ║
    ║  ██║     ██║██╔═══╝ ██╔══██║██╔══╝  ██╔══██╗██║     ██║      ║
    ║  ╚██████╗██║██║     ██║  ██║███████╗██║  ██║╚██████╗███████╗ ║
    ║   ╚═════╝╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝ ║
    ║                                                               ║
    ║            🛡️  AI-Powered Cloud Security Scanner  🛡️           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    scanner = CompleteCipherCloudScanner()
    scan_count = 0
    
    print("\n🚀 Welcome to CipherCloud Complete Scanner!")
    print("This AI system performs comprehensive IAM policy security analysis:")
    print("  1️⃣  Binary Risk Detection (Risky vs Benign)")
    print("  2️⃣  Attack Family Classification (10 attack types)")
    print("\nChoose an option:")
    print("  [1] Interactive Policy Scanner")
    print("  [2] Test with Example Policies")
    print("  [3] Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                print("\n" + "="*50)
                print("INTERACTIVE POLICY SCANNER")
                print("="*50)
                print("Paste your IAM policy JSON below.")
                print("Press Enter twice when finished, or type 'back' to return to menu.")
                print("-" * 50)
                
                lines = []
                while True:
                    line = input()
                    if line.strip().lower() == 'back':
                        break
                    if line.strip() == "" and lines:
                        break
                    lines.append(line)
                
                if lines:
                    policy_json = "\n".join(lines)
                    try:
                        policy = json.loads(policy_json)
                        scan_count += 1
                        print(f"\n🔍 Running CipherCloud Scan #{scan_count}...")
                        
                        result = scanner.complete_scan(policy)
                        scanner.display_results(result)
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ ERROR: Invalid JSON format - {e}")
                        print("Please check your policy syntax and try again.")
            
            elif choice == '2':
                print("\n🧪 Testing with Example Policies...")
                test_examples(scanner)
            
            elif choice == '3':
                print("\n👋 Thank you for using CipherCloud!")
                print("Stay secure! 🛡️")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")

def test_examples(scanner):
    """Test with predefined example policies"""
    
    examples = [
        {
            "name": "Ultra Risky Admin Policy",
            "policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "*",
                        "Resource": "*"
                    }
                ]
            }
        },
        {
            "name": "Privilege Escalation Attack",
            "policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "iam:PassRole",
                            "ec2:RunInstances",
                            "iam:CreateUser"
                        ],
                        "Resource": "*"
                    }
                ]
            }
        },
        {
            "name": "Data Exfiltration Policy",
            "policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket",
                            "s3:*"
                        ],
                        "Resource": "*"
                    }
                ]
            }
        },
        {
            "name": "Benign S3 Read Policy",
            "policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "s3:GetObject",
                        "Resource": "arn:aws:s3:::my-specific-bucket/*"
                    }
                ]
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"🧪 Example {i}: {example['name']}")
        print('='*70)
        
        result = scanner.complete_scan(example['policy'])
        scanner.display_results(result)
        
        input("\nPress Enter to continue to next example...")

if __name__ == "__main__":
    main()