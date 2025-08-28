import json
import random
from typing import List, Dict, Tuple

class IAMDatasetGenerator:
    def __init__(self):
        # Common AWS services and actions
        self.safe_actions = {
            's3': ['s3:GetObject', 's3:ListBucket', 's3:GetBucketLocation'],
            'ec2': ['ec2:DescribeInstances', 'ec2:DescribeImages', 'ec2:DescribeSnapshots'],
            'lambda': ['lambda:InvokeFunction', 'lambda:GetFunction'],
            'dynamodb': ['dynamodb:GetItem', 'dynamodb:Query', 'dynamodb:BatchGetItem'],
            'logs': ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
            'ssm': ['ssm:GetParameter', 'ssm:GetParameters'],
            'rds': ['rds:DescribeDBInstances', 'rds:DescribeDBClusters'],
            'cloudwatch': ['cloudwatch:PutMetricData', 'cloudwatch:GetMetricStatistics'],
            'sns': ['sns:Publish', 'sns:Subscribe'],
            'sqs': ['sqs:SendMessage', 'sqs:ReceiveMessage', 'sqs:DeleteMessage']
        }
        
        self.risky_actions = {
            'iam': ['iam:PassRole', 'iam:CreateUser', 'iam:AttachUserPolicy', 'iam:CreateAccessKey',
                   'iam:UpdateLoginProfile', 'iam:CreatePolicyVersion', 'iam:AttachRolePolicy',
                   'iam:UpdateAssumeRolePolicy', 'iam:PutUserPolicy', 'iam:CreatePolicy'],
            's3': ['s3:*', 's3:DeleteBucket', 's3:PutBucketPolicy'],
            'ec2': ['ec2:*', 'ec2:RunInstances', 'ec2:TerminateInstances'],
            'lambda': ['lambda:CreateFunction', 'lambda:UpdateFunctionCode'],
            'dynamodb': ['dynamodb:*', 'dynamodb:DeleteTable', 'dynamodb:Scan'],
            'kms': ['kms:*', 'kms:Decrypt', 'kms:CreateKey', 'kms:ScheduleKeyDeletion'],
            'secretsmanager': ['secretsmanager:*', 'secretsmanager:GetSecretValue'],
            'sts': ['sts:AssumeRole'],
            'rds': ['rds:*', 'rds:DeleteDBInstance', 'rds:CreateDBSnapshot'],
            '*': ['*']  # Ultimate risky action
        }
        
        self.safe_resources = [
            'arn:aws:s3:::my-specific-bucket/*',
            'arn:aws:dynamodb:us-east-1:123456789012:table/MyTable',
            'arn:aws:lambda:us-east-1:123456789012:function:MyFunction',
            'arn:aws:ec2:us-east-1:123456789012:instance/i-1234567890abcdef0'
        ]
        
        self.risky_resources = [
            '*',
            'arn:aws:s3:::*',
            'arn:aws:iam::*:role/*',
            'arn:aws:ec2:*:*:*'
        ]

    def generate_benign_policy(self) -> Dict:
        """Generate a benign IAM policy with least privilege principles"""
        service = random.choice(list(self.safe_actions.keys()))
        actions = random.sample(self.safe_actions[service], 
                              random.randint(1, min(3, len(self.safe_actions[service]))))
        resource = random.choice(self.safe_resources)
        
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": actions,
                    "Resource": resource
                }
            ]
        }
        
        # Sometimes add conditions for extra security
        if random.random() < 0.3:
            policy["Statement"][0]["Condition"] = {
                "StringEquals": {
                    "aws:RequestedRegion": "us-east-1"
                }
            }
            
        return policy

    def generate_risky_policy(self) -> Dict:
        """Generate a risky IAM policy with common misconfigurations"""
        risk_patterns = [
            self._generate_wildcard_actions,
            self._generate_privilege_escalation,
            self._generate_overpermissive_resources,
            self._generate_dangerous_combinations,
            self._generate_admin_equivalent
        ]
        
        pattern_func = random.choice(risk_patterns)
        return pattern_func()

    def _generate_wildcard_actions(self) -> Dict:
        """Generate policy with wildcard actions"""
        service = random.choice(['s3', 'ec2', 'lambda', 'dynamodb'])
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [f"{service}:*"],
                    "Resource": "*"
                }
            ]
        }

    def _generate_privilege_escalation(self) -> Dict:
        """Generate policy enabling privilege escalation"""
        escalation_combos = [
            (["iam:PassRole", "ec2:RunInstances"], "*"),
            (["iam:AttachUserPolicy"], "arn:aws:iam::*:user/*"),
            (["iam:CreatePolicyVersion"], "arn:aws:iam::*:policy/*"),
            (["iam:UpdateAssumeRolePolicy"], "arn:aws:iam::*:role/*")
        ]
        
        actions, resource = random.choice(escalation_combos)
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": actions,
                    "Resource": resource
                }
            ]
        }

    def _generate_overpermissive_resources(self) -> Dict:
        """Generate policy with overpermissive resource access"""
        service = random.choice(list(self.safe_actions.keys()))
        actions = random.sample(self.safe_actions[service], 
                              random.randint(1, len(self.safe_actions[service])))
        
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": actions,
                    "Resource": "*"  # Too broad
                }
            ]
        }

    def _generate_dangerous_combinations(self) -> Dict:
        """Generate policy with dangerous action combinations"""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "iam:CreateUser",
                        "iam:CreateAccessKey",
                        "iam:AttachUserPolicy"
                    ],
                    "Resource": "*"
                }
            ]
        }

    def _generate_admin_equivalent(self) -> Dict:
        """Generate policy equivalent to admin access"""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "*",
                    "Resource": "*"
                }
            ]
        }

    def generate_dataset(self, num_samples: int = 1000, benign_ratio: float = 0.6) -> List[Tuple[Dict, int]]:
        """
        Generate dataset for binary classification
        Returns list of (policy_dict, label) where label: 0=Benign, 1=Risky
        """
        dataset = []
        num_benign = int(num_samples * benign_ratio)
        num_risky = num_samples - num_benign
        
        # Generate benign samples
        for _ in range(num_benign):
            policy = self.generate_benign_policy()
            dataset.append((policy, 0))  # 0 = Benign
            
        # Generate risky samples
        for _ in range(num_risky):
            policy = self.generate_risky_policy()
            dataset.append((policy, 1))  # 1 = Risky
            
        # Shuffle the dataset
        random.shuffle(dataset)
        return dataset

    def save_dataset(self, dataset: List[Tuple[Dict, int]], filename: str = "iam_dataset.json"):
        """Save dataset to JSON file"""
        formatted_data = []
        for policy, label in dataset:
            formatted_data.append({
                "policy": policy,
                "label": label,
                "label_name": "Benign" if label == 0 else "Risky"
            })
            
        with open(filename, 'w') as f:
            json.dump(formatted_data, f, indent=2)
            
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(dataset)}")
        print(f"Benign: {sum(1 for _, label in dataset if label == 0)}")
        print(f"Risky: {sum(1 for _, label in dataset if label == 1)}")

# Example usage
if __name__ == "__main__":
    generator = IAMDatasetGenerator()
    
    # Generate dataset
    dataset = generator.generate_dataset(num_samples=2000, benign_ratio=0.6)
    
    # Save to file
    generator.save_dataset(dataset, "cipher_cloud_dataset.json")
    
    # Print some examples
    print("\n=== SAMPLE BENIGN POLICY ===")
    benign_samples = [item for item in dataset if item[1] == 0]
    print(json.dumps(benign_samples[0][0], indent=2))
    
    print("\n=== SAMPLE RISKY POLICY ===")
    risky_samples = [item for item in dataset if item[1] == 1]
    print(json.dumps(risky_samples[0][0], indent=2))