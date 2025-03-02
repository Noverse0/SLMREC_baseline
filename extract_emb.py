"""
Extract item embeddings from trained SASRec model and save them to a pickle file.
This script loads the best checkpoint from SASRec training and extracts the item embedding layer.
"""

import torch
import pickle
import argparse
import os

def extract_embeddings(model_path, output_path):
    """Extract item embeddings from trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        output_path: Path to save the extracted embeddings
    """
    # Load model state dict
    model_state_dict = torch.load(model_path)
    
    # Extract embedding layer weights
    item_embeddings = model_state_dict['embedding.weight']
    
    # Save embeddings to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump({'item_embedding': item_embeddings}, f)
    print(f'Saved item embeddings to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings from trained SASRec model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save extracted embeddings')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Extract and save embeddings
    extract_embeddings(args.model_path, args.output_path)