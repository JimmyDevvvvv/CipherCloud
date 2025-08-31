import json
import random
import os
from typing import List, Dict


class PersistenceGenerator:
    """Generate IAM policies that enable persistence attacks"""

    def __init__(self):
        self.persistence_patterns = [
            # Access key creation for persistence
            {
                'actions': ['iam:CreateAccessKey'],
                'resources': ['arn:aws:iam::*:user/*', '*'],
                'description': 'Create long-lived access key'
            },
            # MFA device manipulation
            {
                'actions': ['iam:DeactivateMFADevice', 'iam:DeleteVirtualMFADevice'],
                'resources': ['*'],
                'description': 'Disable MFA protection for persistence'
            },
            # Login profile creation
            {
                'actions': ['iam:CreateLoginProfile'],
                'resources': ['arn:aws:iam::*:user/*'],
                'description': 'Create password login for user persistence'
            },
            # Update login profile
            {
                'actions': ['iam:UpdateLoginProfile'],
                'resources': ['arn:aws:iam::*:user/*'],
                'description': 'Update existing login to maintain persistence'
            },
            # Access key rotation abuse
            {
                'actions': ['iam:UpdateAccessKey'],
                'resources': ['arn:aws:iam::*:user/*'],
                'description': 'Rotate access keys to maintain access'
            },
            # Add user to group with privileges
            {
                'actions': ['iam:AddUserToGroup'],
                'resources': ['*'],
                'description': 'Maintain persistence via group membership'
            },
        ]

    def generate_policy(self) -> Dict:
        pattern = random.choice(self.persistence_patterns)
        actions = pattern['actions'].copy()
        resource = random.choice(pattern['resources'])

        # Add noise actions 20% of the time
        if random.random() < 0.2:
            noise = random.sample([
                'iam:ListUsers', 'iam:GetUser', 'logs:CreateLogGroup',
                'sts:GetCallerIdentity', 's3:ListAllMyBuckets'
            ], random.randint(1, 2))
            actions.extend(noise)

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

        return policy

    def generate_dataset(self, num_samples: int = 2000) -> List[Dict]:
        print(f"Generating {num_samples} Persistence policies...")
        dataset = []
        for i in range(num_samples):
            if i % 500 == 0 and i > 0:
                print(f"  Generated {i}/{num_samples} samples")

            policy = self.generate_policy()
            dataset.append({
                "policy": policy,
                "attack_type": "Persistence"
            })
        return dataset


class ServiceAbuseGenerator:
    """Generate IAM policies that enable service abuse attacks"""

    def __init__(self):
        self.service_abuse_patterns = [
            # EC2 abuse
            {
                'actions': ['ec2:RunInstances'],
                'resources': ['*'],
                'description': 'Abuse EC2 for crypto mining'
            },
            # Lambda abuse
            {
                'actions': ['lambda:CreateFunction', 'lambda:InvokeFunction'],
                'resources': ['*'],
                'description': 'Deploy malicious Lambda'
            },
            # S3 abuse
            {
                'actions': ['s3:PutObject', 's3:GetObject'],
                'resources': ['arn:aws:s3:::*/*'],
                'description': 'Abuse S3 buckets for storage/exfiltration'
            },
            # SNS abuse
            {
                'actions': ['sns:Publish'],
                'resources': ['arn:aws:sns:*:*:*'],
                'description': 'Send malicious SNS messages'
            },
            # SQS abuse
            {
                'actions': ['sqs:SendMessage'],
                'resources': ['arn:aws:sqs:*:*:*'],
                'description': 'Abuse SQS for message flooding'
            },
            # Glue abuse
            {
                'actions': ['glue:CreateJob', 'glue:StartJobRun'],
                'resources': ['*'],
                'description': 'Use Glue jobs for computation abuse'
            },
        ]

    def generate_policy(self) -> Dict:
        pattern = random.choice(self.service_abuse_patterns)
        actions = pattern['actions'].copy()
        resource = random.choice(pattern['resources'])

        # Add noise 25% of the time
        if random.random() < 0.25:
            noise = random.sample([
                'logs:CreateLogStream', 'ec2:DescribeInstances', 'iam:ListRoles',
                's3:ListBucket', 'sts:GetCallerIdentity'
            ], random.randint(1, 2))
            actions.extend(noise)

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
        return policy

    def generate_dataset(self, num_samples: int = 2000) -> List[Dict]:
        print(f"Generating {num_samples} Service Abuse policies...")
        dataset = []
        for i in range(num_samples):
            if i % 500 == 0 and i > 0:
                print(f"  Generated {i}/{num_samples} samples")

            policy = self.generate_policy()
            dataset.append({
                "policy": policy,
                "attack_type": "Service Abuse"
            })
        return dataset


def save_dataset(dataset: List[Dict], filename: str):
    # Always save in the top-level Classifier Dataset folder
    base_dir = os.path.join(os.path.dirname(__file__), "..", "Classifier Dataset")
    os.makedirs(base_dir, exist_ok=True)

    filename = os.path.join(base_dir, filename)

    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n📊 Dataset saved to {filename}")
    print(f"Total samples: {len(dataset)}")


def show_examples(dataset: List[Dict], attack_type: str):
    print("\n" + "="*80)
    print(f"SAMPLE ATTACK POLICIES: {attack_type}")
    print("="*80)
    samples = [item for item in dataset if item['attack_type'] == attack_type]

    for i in range(min(2, len(samples))):
        print(f"\nExample {i+1}:")
        print(json.dumps(samples[i]['policy'], indent=2))


if __name__ == "__main__":
    print("🚀 CipherCloud Attack Dataset Generation")
    print("="*50)

    # Generate Persistence dataset
    persistence_gen = PersistenceGenerator()
    persistence_dataset = persistence_gen.generate_dataset(2000)
    save_dataset(persistence_dataset, "persistence.json")
    show_examples(persistence_dataset, "Persistence")

    # Generate Service Abuse dataset
    abuse_gen = ServiceAbuseGenerator()
    abuse_dataset = abuse_gen.generate_dataset(2000)
    save_dataset(abuse_dataset, "service_abuse.json")
    show_examples(abuse_dataset, "Service Abuse")

    print("\n✅ All datasets generated!")
    print("\n📁 Files created in 'Classifier Dataset' folder:")
    print("  - persistence.json (2000 samples)")
    print("  - service_abuse.json (2000 samples)")
    print("\n🏷️  Labels: 'Persistence' and 'Service Abuse'")
    print("🤖 Ready for multi-class classifier training!")