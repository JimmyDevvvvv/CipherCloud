import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CipherCloudBinaryClassifier:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_names = []
        
    def load_data(self, X_path="cipher_cloud_features_X.csv", 
                  y_path="cipher_cloud_features_y.csv",
                  feature_names_path="cipher_cloud_features_feature_names.json"):
        """Load the preprocessed feature data"""
        
        # Load features (X)
        print("Loading feature matrix...")
        self.X = pd.read_csv(X_path)
        print(f"Feature matrix shape: {self.X.shape}")
        
        # Load labels (y)
        print("Loading labels...")
        y_df = pd.read_csv(y_path)
        self.y = y_df['label'].values
        print(f"Labels shape: {self.y.shape}")
        print(f"Label distribution: Benign={sum(self.y==0)}, Risky={sum(self.y==1)}")
        
        # Load feature names
        print("Loading feature names...")
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        print(f"Number of features: {len(self.feature_names)}")
        
        # Verify data consistency
        assert self.X.shape[0] == len(self.y), "X and y have different number of samples"
        assert self.X.shape[1] == len(self.feature_names), "Feature matrix and feature names mismatch"
        
        print("✅ Data loaded successfully!")
        return self.X, self.y

    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into train/test and handle preprocessing"""
        
        print(f"Splitting data (test_size={test_size})...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y  # Ensure balanced splits
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Train distribution: Benign={sum(self.y_train==0)}, Risky={sum(self.y_train==1)}")
        print(f"Test distribution: Benign={sum(self.y_test==0)}, Risky={sum(self.y_test==1)}")
        
        # Initialize scaler for models that need it
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """Train multiple models and compare performance"""
        
        print("Training models...")
        
        # Define models
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'use_scaling': False
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'use_scaling': True
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'use_scaling': True
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            print(f"\n🔧 Training {name}...")
            
            model = config['model']
            use_scaling = config['use_scaling']
            
            # Choose appropriate data
            X_train = self.X_train_scaled if use_scaling else self.X_train
            X_test = self.X_test_scaled if use_scaling else self.X_test
            
            # Train model
            model.fit(X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, self.y_test)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'use_scaling': use_scaling
            }
            
            print(f"✅ {name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
        
        self.models = results
        
        # Find best model (by AUC)
        best_model_name = max(results, key=lambda x: results[x]['auc'])
        self.best_model = results[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name} (AUC: {self.best_model['auc']:.3f})")
        
        return results

    def evaluate_models(self):
        """Detailed evaluation of all models"""
        
        print("\n" + "="*60)
        print("DETAILED MODEL EVALUATION")
        print("="*60)
        
        for name, result in self.models.items():
            print(f"\n🔍 {name} Results:")
            print("-" * 40)
            
            y_pred = result['predictions']
            
            # Classification report
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['Benign', 'Risky']))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"Confusion Matrix:")
            print(f"              Predicted")
            print(f"           Benign  Risky")
            print(f"Benign       {cm[0,0]:4d}   {cm[0,1]:4d}")
            print(f"Risky        {cm[1,0]:4d}   {cm[1,1]:4d}")
            
            # Metrics summary
            print(f"Accuracy: {result['accuracy']:.3f}")
            print(f"AUC Score: {result['auc']:.3f}")

    def analyze_feature_importance(self, top_n=15):
        """Analyze feature importance for interpretability"""
        
        if self.best_model_name == 'Random Forest':
            model = self.best_model['model']
            importances = model.feature_importances_
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top {top_n} Most Important Features ({self.best_model_name}):")
            print("-" * 60)
            for i, row in feature_importance_df.head(top_n).iterrows():
                print(f"{row['feature']:40s} {row['importance']:.4f}")
            
            return feature_importance_df
        
        elif self.best_model_name == 'Logistic Regression':
            model = self.best_model['model']
            coefficients = abs(model.coef_[0])  # Take absolute values
            
            # Create coefficient DataFrame
            coef_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients
            }).sort_values('coefficient', ascending=False)
            
            print(f"\n🔍 Top {top_n} Most Important Features ({self.best_model_name}):")
            print("-" * 60)
            for i, row in coef_df.head(top_n).iterrows():
                print(f"{row['feature']:40s} {row['coefficient']:.4f}")
            
            return coef_df
        
        else:
            print(f"Feature importance not available for {self.best_model_name}")
            return None

    def predict_new_policy(self, policy_features):
        """Predict risk level for a new policy"""
        
        if self.best_model is None:
            raise ValueError("No trained model available. Run train_models() first.")
        
        model = self.best_model['model']
        use_scaling = self.best_model['use_scaling']
        
        # Ensure features are in the right format
        if isinstance(policy_features, pd.Series):
            policy_features = policy_features.values.reshape(1, -1)
        elif isinstance(policy_features, list):
            policy_features = np.array(policy_features).reshape(1, -1)
        
        # Apply scaling if needed
        if use_scaling:
            policy_features = self.scaler.transform(policy_features)
        
        # Make prediction
        prediction = model.predict(policy_features)[0]
        probability = model.predict_proba(policy_features)[0]
        
        risk_level = "Risky" if prediction == 1 else "Benign"
        confidence = max(probability)
        
        return {
            'prediction': prediction,
            'risk_level': risk_level,
            'confidence': confidence,
            'probabilities': {
                'benign': probability[0],
                'risky': probability[1]
            }
        }

    def save_model(self, filename="cipher_cloud_binary_model.pkl"):
        """Save the best trained model"""
        import pickle
        
        model_data = {
            'model': self.best_model['model'],
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name,
            'use_scaling': self.best_model['use_scaling']
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filename}")

    def cross_validate(self, cv=5):
        """Perform cross-validation for more robust evaluation"""
        
        print(f"\n🔄 Performing {cv}-fold cross-validation...")
        
        for name, config in [('Random Forest', False), ('Logistic Regression', True)]:
            model = RandomForestClassifier(n_estimators=100, random_state=42) if name == 'Random Forest' else LogisticRegression(random_state=42, max_iter=1000)
            X = self.X_train_scaled if config else self.X_train
            
            cv_scores = cross_val_score(model, X, self.y_train, cv=cv, scoring='roc_auc')
            
            print(f"{name:20s} - Mean AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def save_evaluation_plots(self, output_dir="Model/Evaluation_Plots"):
        """Save confusion matrix and ROC curve images for each model."""
        os.makedirs(output_dir, exist_ok=True)
        for name, result in self.models.items():
            y_pred = result['predictions']
            y_proba = result['probabilities']
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign','Risky'], yticklabels=['Benign','Risky'])
            plt.title(f"{name} - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
            plt.close()
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr, label=f"{name} (AUC={result['auc']:.2f})")
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{name} - ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"))
            plt.close()
        print(f"✅ Evaluation plots saved to {output_dir}")

# Example usage
def main():
    # Initialize classifier
    classifier = CipherCloudBinaryClassifier()
    
    # Load data
    X, y = classifier.load_data()
    
    # Prepare data (train/test split)
    classifier.prepare_data(test_size=0.2)
    
    # Train models
    results = classifier.train_models()
    
    # Evaluate models
    classifier.evaluate_models()
    
    # Analyze feature importance
    classifier.analyze_feature_importance(top_n=15)
    
    # Cross-validation
    classifier.cross_validate()
    
    # Save best model
    classifier.save_model()
    
    # Save evaluation plots
    classifier.save_evaluation_plots()
    
    print("\n🎯 CipherCloud Binary Classifier Training Complete!")
    print(f"Best Model: {classifier.best_model_name}")
    print(f"Best AUC Score: {classifier.best_model['auc']:.3f}")
    
    return classifier

if __name__ == "__main__":
    classifier = main()