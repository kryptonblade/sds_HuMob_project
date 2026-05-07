import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from model import MobilityBERTMoE
from data_loader import train_test_generate_mob_time_series_dataloader
import torch.nn.functional as F

class ExpertAnalyzer:
    def __init__(self, model):
        self.model = model
        self.expert_activations = []
        self.expert_inputs = []
        self.expert_outputs = []
        
    def hook_expert_activations(self):
        """Hook into the MoE layer to capture expert routing information"""
        def hook_fn(module, input, output):
            if isinstance(module, type(self.model.moe)):
                x = input[0]
                batch_size, seq_len, _ = x.size()
                x_reshaped = x.reshape(batch_size * seq_len, -1)
                
                # Get gating probabilities
                gate_logits = module.gate(x_reshaped)
                gate_probs = F.softmax(gate_logits, dim=1)
                
                # Store expert routing information
                self.expert_activations.append({
                    'gate_probs': gate_probs.detach().cpu().numpy(),
                    'input_features': x_reshaped.detach().cpu().numpy(),
                    'expert_outputs': output[0].detach().cpu().numpy() if isinstance(output, tuple) else output.detach().cpu().numpy()
                })
        
        # Register hook
        self.model.moe.register_forward_hook(hook_fn)
    
    def analyze_expert_specialization(self, data_loader, device, num_batches=10):
        """Analyze which experts specialize in which types of patterns"""
        self.expert_activations = []
        self.hook_expert_activations()
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                    
                input_seq_feature, historical_locations, predict_seq_feature, future_locations = [b.to(device) for b in batch]
                
                # Forward pass
                if hasattr(self.model, 'moe'):
                    logits, aux_loss = self.model(input_seq_feature, historical_locations, predict_seq_feature)
                else:
                    logits = self.model(input_seq_feature, historical_locations, predict_seq_feature)
        
        return self.analyze_activations()
    
    def analyze_activations(self):
        """Analyze collected expert activations to identify patterns"""
        if not self.expert_activations:
            return None
        
        # Combine all activations
        all_gate_probs = np.concatenate([act['gate_probs'] for act in self.expert_activations])
        all_inputs = np.concatenate([act['input_features'] for act in self.expert_activations])
        
        num_experts = all_gate_probs.shape[1]
        
        # Analyze expert usage patterns
        expert_usage = all_gate_probs.mean(axis=0)
        expert_std = all_gate_probs.std(axis=0)
        
        # Find which expert dominates for each input
        dominant_experts = np.argmax(all_gate_probs, axis=1)
        
        # Analyze input features that trigger each expert
        expert_patterns = {}
        for expert_id in range(num_experts):
            mask = dominant_experts == expert_id
            if np.sum(mask) > 0:
                expert_inputs = all_inputs[mask]
                expert_patterns[expert_id] = {
                    'usage_rate': np.mean(mask),
                    'avg_input_features': np.mean(expert_inputs, axis=0),
                    'std_input_features': np.std(expert_inputs, axis=0),
                    'num_samples': np.sum(mask)
                }
        
        return {
            'expert_usage': expert_usage,
            'expert_std': expert_std,
            'expert_patterns': expert_patterns,
            'dominant_experts': dominant_experts
        }
    
    def interpret_expert_patterns(self, analysis_results, feature_names=None):
        """Interpret what patterns each expert specializes in"""
        if feature_names is None:
            feature_names = ['day', 'time', 'uid', 'day_of_week', 'weekday', 'delta_time', 'location_emb']
        
        interpretations = {}
        
        for expert_id, pattern in analysis_results['expert_patterns'].items():
            avg_features = pattern['avg_input_features']
            std_features = pattern['std_input_features']
            
            # Find most distinctive features for this expert
            feature_importance = np.abs(avg_features) / (std_features + 1e-8)
            top_features_idx = np.argsort(feature_importance)[-5:]  # Top 5 most important features
            
            interpretations[expert_id] = {
                'usage_rate': pattern['usage_rate'] * 100,
                'top_features': [(feature_names[i] if i < len(feature_names) else f'feature_{i}', 
                                avg_features[i], feature_importance[i]) for i in top_features_idx],
                'interpretation': self._interpret_features(top_features_idx, feature_names, avg_features)
            }
        
        return interpretations
    
    def _interpret_features(self, top_features_idx, feature_names, avg_features):
        """Interpret what the top features suggest about expert specialization"""
        interpretations = []
        
        for idx in top_features_idx:
            if idx < len(feature_names):
                feature_name = feature_names[idx]
                feature_val = avg_features[idx]
                
                if feature_name == 'time':
                    if feature_val > 24:
                        interpretations.append("specializes in late night/early morning patterns")
                    elif feature_val < 12:
                        interpretations.append("specializes in morning/daytime patterns")
                    else:
                        interpretations.append("specializes in afternoon/evening patterns")
                
                elif feature_name == 'day_of_week':
                    if feature_val < 2:
                        interpretations.append("specializes in Monday/Tuesday patterns")
                    elif feature_val >= 5:
                        interpretations.append("specializes in weekend patterns")
                    else:
                        interpretations.append("specializes in weekday patterns")
                
                elif feature_name == 'delta_time':
                    if feature_val > 100:
                        interpretations.append("specializes in long gaps/irregular patterns")
                    elif feature_val < 10:
                        interpretations.append("specializes in regular/frequent patterns")
                    else:
                        interpretations.append("specializes in moderate temporal patterns")
                
                elif 'location' in feature_name:
                    interpretations.append("specializes in spatial location patterns")
                
                elif 'uid' in feature_name:
                    interpretations.append("specializes in user-specific patterns")
                
                elif 'day' in feature_name:
                    if feature_val > 50:
                        interpretations.append("specializes in end-of-month patterns")
                    else:
                        interpretations.append("specializes in beginning/mid-month patterns")
        
        return interpretations if interpretations else ["specializes in complex mixed patterns"]
    
    def visualize_expert_usage(self, analysis_results, save_path=None):
        """Visualize expert usage patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Expert usage rates
        expert_usage = analysis_results['expert_usage']
        axes[0, 0].bar(range(len(expert_usage)), expert_usage)
        axes[0, 0].set_title('Expert Usage Rates')
        axes[0, 0].set_xlabel('Expert ID')
        axes[0, 0].set_ylabel('Average Usage Probability')
        
        # Expert usage variability
        expert_std = analysis_results['expert_std']
        axes[0, 1].bar(range(len(expert_std)), expert_std)
        axes[0, 1].set_title('Expert Usage Variability')
        axes[0, 1].set_xlabel('Expert ID')
        axes[0, 1].set_ylabel('Standard Deviation')
        
        # Dominant expert distribution
        dominant_experts = analysis_results['dominant_experts']
        axes[1, 0].hist(dominant_experts, bins=len(np.unique(dominant_experts)))
        axes[1, 0].set_title('Dominant Expert Distribution')
        axes[1, 0].set_xlabel('Expert ID')
        axes[1, 0].set_ylabel('Frequency')
        
        # Expert specialization heatmap
        patterns = analysis_results['expert_patterns']
        if patterns:
            expert_matrix = np.array([pattern['usage_rate'] for pattern in patterns.values()])
            sns.heatmap(expert_matrix.reshape(1, -1), annot=True, ax=axes[1, 1], cmap='YlOrRd')
            axes[1, 1].set_title('Expert Specialization Heatmap')
            axes[1, 1].set_xlabel('Expert ID')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def generate_report(self, analysis_results, interpretations, save_path=None):
        """Generate a detailed report of expert specializations"""
        report = []
        report.append("=" * 60)
        report.append("EXPERT SPECIALIZATION ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append(f"\nTotal Experts: {len(analysis_results['expert_usage'])}")
        report.append(f"Total Samples Analyzed: {len(analysis_results['dominant_experts'])}")
        
        report.append("\n" + "=" * 40)
        report.append("EXPERT USAGE SUMMARY")
        report.append("=" * 40)
        
        for expert_id, usage in enumerate(analysis_results['expert_usage']):
            report.append(f"Expert {expert_id}: {usage:.3f} ({usage*100:.1f}%) usage rate")
        
        report.append("\n" + "=" * 40)
        report.append("EXPERT SPECIALIZATION PATTERNS")
        report.append("=" * 40)
        
        for expert_id, interp in interpretations.items():
            report.append(f"\nExpert {expert_id}:")
            report.append(f"  Usage Rate: {interp['usage_rate']:.1f}%")
            report.append(f"  Top Features:")
            for feature_name, feature_val, importance in interp['top_features']:
                report.append(f"    - {feature_name}: {feature_val:.3f} (importance: {importance:.3f})")
            report.append(f"  Interpretation: {', '.join(interp['interpretation'])}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def analyze_model_experts(model_path, city='A', device='cuda', num_batches=20):
    """Main function to analyze expert specializations in a trained model"""
    
    # Instantiate model first
    model = MobilityBERTMoE(
        num_location_ids=61505, 
        hidden_size=128, 
        hidden_layers=6, 
        attention_heads=4,
        day_embedding_size=32, 
        time_embedding_size=32, 
        day_of_week_embedding_size=16, 
        weekday_embedding_size=4,
        location_embedding_size=128, 
        dropout=0.1, 
        max_seq_length=720, 
        num_experts=8
    )
    
    # Load state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Create data loader
    train_loader, test_df, generate_df = train_test_generate_mob_time_series_dataloader(
        city=city, input_seq_length=240, predict_seq_length=30, 
        batch_size=32, look_back_len=24, subsample=True, subsample_number=50
    )
    
    # Create analyzer
    analyzer = ExpertAnalyzer(model)
    
    # Analyze expert specializations
    print("Analyzing expert specializations...")
    analysis_results = analyzer.analyze_expert_specialization(train_loader, device, num_batches)
    
    if analysis_results:
        # Interpret patterns
        feature_names = ['day', 'time', 'uid', 'day_of_week', 'weekday', 'delta_time', 'location_emb_1', 'location_emb_2', 'location_emb_3', 'location_emb_4']
        interpretations = analyzer.interpret_expert_patterns(analysis_results, feature_names)
        
        # Visualize
        analyzer.visualize_expert_usage(analysis_results, 'expert_usage.png')
        
        # Generate report
        report = analyzer.generate_report(analysis_results, interpretations, 'expert_analysis_report.txt')
        print(report)
        
        return analysis_results, interpretations
    else:
        print("No expert activations captured. Make sure the model has MoE layers.")
        return None, None


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/best_mobility_bert_moe_dtw_130.47.pth"  # Replace with actual model path
    analyze_model_experts(model_path, city='A')
