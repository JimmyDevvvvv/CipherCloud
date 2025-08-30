import json
import random
from typing import List, Dict

class PrivilegeEscalationGenerator:
    """Generate IAM policies that enable privilege escalation attacks"""
    
    def __init__(self):
        # Core privilege escalation patterns
        self.escalation_patterns = [
            # PassRole + Service Abuse (most common)
            {
                'actions': ['iam:PassRole', 'ec2:RunInstances'],
                'resources': ['*', 'arn:aws:iam::*:role/*'],
                'description': 'Pass admin role to EC2 instance'
            },
            {
                'actions': ['iam:PassRole', 'lambda:CreateFunction'],
                'resources': ['*'],
                'description': 'Pass role to Lambda for code execution'
            },
            {
                'actions': ['iam:PassRole', 'ecs:RunTask'],
                'resources': ['*'],
                'description': 'Pass role to ECS container'
            },
            {
                'actions': ['iam:PassRole', 'glue:CreateDevEndpoint'],
                'resources': ['*'],
                'description': 'Pass role to Glue development endpoint'
            },
            
            # Direct Policy Attachment
            {
                'actions': ['iam:AttachUserPolicy'],
                'resources': ['arn:aws:iam::*:user/*', '*'],
                'description': 'Attach admin policy to user'
            },
            {
                'actions': ['iam:AttachRolePolicy'],
                'resources': ['arn:aws:iam::*:role/*', '*'],
                'description': 'Attach admin policy to role'
            },
            
            # Policy Version Abuse
            {
                'actions': ['iam:CreatePolicyVersion'],
                'resources': ['arn:aws:iam::*:policy/*', '*'],
                'description': 'Create malicious policy version'
            },
            
            # Trust Policy Modification
            {
                'actions': ['iam:UpdateAssumeRolePolicy'],
                'resources': ['arn:aws:iam::*:role/*', '*'],
                'description': 'Modify role trust policy'
            },
            
            # Multi-step escalation
            {
                'actions': ['iam:CreateUser', 'iam:AttachUserPolicy'],
                'resources': ['*'],
                'description': 'Create user and attach admin policy'
            },
            
            # Service-specific escalation
            {
                'actions': ['iam:PassRole', 'sagemaker:CreateNotebookInstance'],
                'resources': ['*'],
                'description': 'Pass role to SageMaker notebook'
            }
        ]

    def generate_policy(self) -> Dict:
        """Generate a single privilege escalation policy"""
        # Choose random pattern
        pattern = random.choice(self.escalation_patterns)
        
        # Get base actions and resource
        actions = pattern['actions'].copy()
        resource = random.choice(pattern['resources'])
        
        # Add noise actions (realistic but harmless) 25% of the time
        if random.random() < 0.25:
            noise = random.sample([
                'ec2:DescribeInstances', 'iam:ListRoles', 'sts:GetCallerIdentity',
                'logs:CreateLogGroup', 's3:ListBucket'
            ], random.randint(1, 2))
            actions.extend(noise)
        
        # Create policy
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
        
        # Add conditions 10% of the time (sometimes bypassed anyway)
        if random.random() < 0.1:
            policy["Statement"][0]["Condition"] = {
                "StringEquals": {"aws:RequestedRegion": "us-east-1"}
            }
        
        return policy

    def generate_dataset(self, num_samples: int = 2000) -> List[Dict]:
        """Generate dataset of privilege escalation policies"""
        print(f"Generating {num_samples} Privilege Escalation policies...")
        
        dataset = []
        for i in range(num_samples):
            if i % 500 == 0 and i > 0:
                print(f"  Generated {i}/{num_samples} samples")
            
            policy = self.generate_policy()
            dataset.append({
                "policy": policy,
                "attack_type": "Privilege Escalation"
            })
        
        return dataset

class ShadowAdminGenerator:
    """Generate IAM policies that enable shadow admin creation"""
    
    def __init__(self):
        # Core shadow admin patterns - focus on stealth and hidden access
        self.shadow_patterns = [
            # Inline Policy Abuse (most stealthy)
            {
                'actions': ['iam:PutUserPolicy'],
                'resources': ['arn:aws:iam::*:user/*', '*'],
                'description': 'Embed admin policy inline to user'
            },
            {
                'actions': ['iam:PutRolePolicy'],
                'resources': ['arn:aws:iam::*:role/*', '*'],
                'description': 'Embed admin policy inline to role'
            },
            {
                'actions': ['iam:PutGroupPolicy'],
                'resources': ['arn:aws:iam::*:group/*', '*'],
                'description': 'Embed admin policy inline to group'
            },
            
            # Custom Policy Creation
            {
                'actions': ['iam:CreatePolicy', 'iam:AttachRolePolicy'],
                'resources': ['*'],
                'description': 'Create custom admin policy and attach'
            },
            {
                'actions': ['iam:CreatePolicy', 'iam:AttachUserPolicy'],
                'resources': ['*'],
                'description': 'Create and attach policy to user'
            },
            
            # Policy Version Backdooring
            {
                'actions': ['iam:CreatePolicyVersion', 'iam:SetDefaultPolicyVersion'],
                'resources': ['arn:aws:iam::*:policy/*', '*'],
                'description': 'Backdoor via policy version manipulation'
            },
            
            # Trust Policy Backdooring
            {
                'actions': ['iam:UpdateAssumeRolePolicy', 'sts:AssumeRole'],
                'resources': ['*'],
                'description': 'Backdoor via trust policy modification'
            },
            
            # Group-based Shadow Access
            {
                'actions': ['iam:CreateGroup', 'iam:AttachGroupPolicy', 'iam:AddUserToGroup'],
                'resources': ['*'],
                'description': 'Create admin group and add users'
            },
            
            # Stealth User Creation
            {
                'actions': ['iam:CreateUser', 'iam:PutUserPolicy', 'iam:CreateAccessKey'],
                'resources': ['*'],
                'description': 'Create user with hidden admin access'
            },
            
            # Policy Discovery + Attachment
            {
                'actions': ['iam:ListPolicies', 'iam:GetPolicy', 'iam:AttachUserPolicy'],
                'resources': ['*'],
                'description': 'Discover existing policies and attach'
            }
        ]

    def generate_policy(self) -> Dict:
        """Generate a single shadow admin policy"""
        # Choose random pattern
        pattern = random.choice(self.shadow_patterns)
        
        # Get base actions and resource
        actions = pattern['actions'].copy()
        resource = random.choice(pattern['resources'])
        
        # Add noise actions (stealthy reconnaissance) 20% of the time
        if random.random() < 0.2:
            noise = random.sample([
                'iam:GetUser', 'iam:ListAttachedUserPolicies', 'iam:GetRole',
                'iam:ListRoles', 'sts:GetCallerIdentity'
            ], random.randint(1, 2))
            actions.extend(noise)
        
        # Create policy
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
        
        # Shadow admin policies rarely have conditions (avoid detection)
        # But occasionally add weak ones
        if random.random() < 0.05:
            policy["Statement"][0]["Condition"] = {
                "Bool": {"aws:SecureTransport": "true"}
            }
        
        return policy

    def generate_dataset(self, num_samples: int = 2000) -> List[Dict]:
        """Generate dataset of shadow admin policies"""
        print(f"Generating {num_samples} Shadow Admin policies...")
        
        dataset = []
        for i in range(num_samples):
            if i % 500 == 0 and i > 0:
                print(f"  Generated {i}/{num_samples} samples")
            
            policy = self.generate_policy()
            dataset.append({
                "policy": policy,
                "attack_type": "Shadow Admin"
            })
        
        return dataset

def create_combined_dataset(privesc_samples: int = 2000, shadow_samples: int = 2000):
    """Create combined dataset with both attack types"""
    
    # Generate Privilege Escalation data
    privesc_gen = PrivilegeEscalationGenerator()
    privesc_data = privesc_gen.generate_dataset(privesc_samples)
    
    # Generate Shadow Admin data  
    shadow_gen = ShadowAdminGenerator()
    shadow_data = shadow_gen.generate_dataset(shadow_samples)
    
    # Combine datasets
    combined_data = privesc_data + shadow_data
    
    # Shuffle
    random.shuffle(combined_data)
    
    return combined_data

def save_dataset(dataset: List[Dict], filename: str = "cipher_cloud_attack_dataset.json"):
    """Save dataset to JSON file"""
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Print statistics
    attack_counts = {}
    for item in dataset:
        attack_type = item['attack_type']
        attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
    
    print(f"\n📊 Dataset saved to {filename}")
    print(f"Total samples: {len(dataset)}")
    for attack_type, count in attack_counts.items():
        print(f"  {attack_type}: {count} samples")

def show_examples(dataset: List[Dict]):
    """Show example policies for each attack type"""
    print("\n" + "="*80)
    print("SAMPLE ATTACK POLICIES")
    print("="*80)
    
    # Get unique attack types
    attack_types = list(set(item['attack_type'] for item in dataset))
    
    for attack_type in attack_types:
        samples = [item for item in dataset if item['attack_type'] == attack_type]
        
        print(f"\n🎯 {attack_type}")
        print("-" * 50)
        
        # Show 2 examples
        for i in range(min(2, len(samples))):
            print(f"\nExample {i+1}:")
            print(json.dumps(samples[i]['policy'], indent=2))

# Standalone generators for other people to use
class PrivilegeEscalationOnly:
    """Standalone Privilege Escalation generator - easy for others to use"""
    
    @staticmethod
    def generate_samples(num_samples: int = 2000, filename: str = "privilege_escalation_dataset.json"):
        """Generate privilege escalation dataset only"""
        generator = PrivilegeEscalationGenerator()
        dataset = generator.generate_dataset(num_samples)
        save_dataset(dataset, filename)
        return dataset

class ShadowAdminOnly:
    """Standalone Shadow Admin generator - easy for others to use"""
    
    @staticmethod  
    def generate_samples(num_samples: int = 2000, filename: str = "shadow_admin_dataset.json"):
        """Generate shadow admin dataset only"""
        generator = ShadowAdminGenerator()
        dataset = generator.generate_dataset(num_samples)
        save_dataset(dataset, filename)
        return dataset

# Template for other people to create new attack types
class AttackTypeTemplate:
    """
    TEMPLATE: Copy this class to create new attack type generators
    
    Example usage for creating "Data Exfiltration" generator:
    
    class DataExfiltrationGenerator:
        def __init__(self):
            self.exfiltration_patterns = [
                {
                    'actions': ['s3:*'],
                    'resources': ['*'],
                    'description': 'Full S3 access for data theft'
                },
                # Add more patterns...
            ]
        
        def generate_policy(self):
            # Copy the pattern from PrivilegeEscalationGenerator.generate_policy()
            # but use self.exfiltration_patterns
        
        def generate_dataset(self, num_samples=2000):
            # Copy the pattern from PrivilegeEscalationGenerator.generate_dataset()
            # but set attack_type to "Data Exfiltration"
    """
    pass

# Main execution
if __name__ == "__main__":
    print("🚀 CipherCloud Attack Dataset Generation")
    print("="*50)
    
    # Option 1: Generate combined dataset (recommended for multi-class training)
    print("Generating combined dataset...")
    combined_dataset = create_combined_dataset(privesc_samples=2000, shadow_samples=2000)
    save_dataset(combined_dataset, "cipher_cloud_privesc_shadow_combined.json")
    show_examples(combined_dataset)
    
    print("\n" + "="*50)
    
    # Option 2: Generate separate datasets (for individual analysis)
    print("Generating separate datasets...")
    
    # Privilege Escalation only
    PrivilegeEscalationOnly.generate_samples(2000, "privilege_escalation_only.json")
    
    # Shadow Admin only  
    ShadowAdminOnly.generate_samples(2000, "shadow_admin_only.json")
    
    print("\n✅ All datasets generated!")
    print("\n📁 Files created:")
    print("  - cipher_cloud_privesc_shadow_combined.json (4000 samples)")
    print("  - privilege_escalation_only.json (2000 samples)")
    print("  - shadow_admin_only.json (2000 samples)")
    print("\n🏷️  Labels: 'Privilege Escalation' and 'Shadow Admin'")
    print("🤖 Ready for multi-class classifier training!")