import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from train_gcl import load_data
from models import LogisticRegression

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GraphCL Embeddings")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the .pt embeddings file")
    parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    return parser.parse_args()

def train_and_evaluate(X_train, y_train, X_test, y_test, device, name, class_weights=None):
    print(f"\n" + "-"*50)
    print(f"SCENARIO: {name}")
    print("-"*(len(name) + 10))
    
    input_dim = X_train.shape[1]
    classifier = LogisticRegression(input_dim, 2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Using Class Weights: {class_weights.tolist()}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print(f"  Using Uniform Weights (None)")

    classifier.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = classifier(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        out_test = classifier(X_test)
        probs = F.softmax(out_test, dim=1)
        preds = probs.argmax(dim=1)
        
        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs[:, 1].cpu().numpy()
        
        acc = (y_true == y_pred).sum() / len(y_true)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"  Accuracy:       {acc:.4f}")
        print(f"  Precision (1):  {prec:.4f}")
        print(f"  Recall (1):     {rec:.4f}")
        print(f"  F1 Score (1):   {f1:.4f}")
        print(f"  AUC-ROC:        {auc:.4f}")
        print(f"  Avg Precision:  {ap:.4f}")
        print(f"  Confusion Matrix: \n{cm}")
        
    return {"name": name, "f1": f1, "auc": auc, "ap": ap}

def main():
    args = parse_args()

    if args.project_root:
        project_root = args.project_root
    else:
        project_root = os.environ.get("PROJECT_ROOT")
        if not project_root:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = load_data(project_root)

    print(f"\nLoading embeddings from {args.embeddings}...")
    embeddings = torch.load(args.embeddings, map_location=device)
    
    if torch.isnan(embeddings).any():
        print("⚠️ Warning: Embeddings contain NaNs. Replacing with 0.")
        embeddings = torch.nan_to_num(embeddings, nan=0.0)

    embeddings = F.normalize(embeddings, dim=1)

    y = data.y.to(device)
    labeled_mask = (y != -1)
    train_mask = data.train_mask.to(device) & labeled_mask
    test_mask = data.test_mask.to(device) & labeled_mask
    
    X = embeddings.to(device)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTrain set size: {len(y_train)}")
    print(f"Test set size:  {len(y_test)}")

    train_class_counts = torch.bincount(y_train)
    weights = 1.0 / train_class_counts.float()
    weights = weights / weights.sum()
    weights = weights.to(device)

    results = []

    results.append(train_and_evaluate(X_train, y_train, X_test, y_test, device, 
                                    name="Weighted (Recall Focused)", 
                                    class_weights=weights))

    results.append(train_and_evaluate(X_train, y_train, X_test, y_test, device, 
                                    name="Unweighted (Standard)", 
                                    class_weights=None))
                                    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for r in results:
        print(f"{r['name']:<25} | F1: {r['f1']:.4f} | AUC: {r['auc']:.4f} | AP: {r['ap']:.4f}")

if __name__ == "__main__":
    main()
