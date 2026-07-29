import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import logging

sys.path.insert(0, '/mnt/c/work/profiler_testing/dl_model')
from train_rnn_v6 import process_dataframe, ExonDataset, collate_fn, ExonRNN, apply_scalers
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def main():
    logger.info('Loading V6 PyTorch Model Checkpoint...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load('/mnt/c/work/profiler_testing/dl_model/dl_orthology_model_v6.pth', map_location=device, weights_only=False)
    model = ExonRNN(exon_dim=checkpoint['exon_dim'], global_dim=checkpoint['global_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    exon_scaler = StandardScaler()
    exon_scaler.mean_ = checkpoint['exon_scaler_mean']
    exon_scaler.scale_ = checkpoint['exon_scaler_scale']
    
    global_scaler = StandardScaler()
    global_scaler.mean_ = checkpoint['global_scaler_mean']
    global_scaler.scale_ = checkpoint['global_scaler_scale']
    
    logger.info('Loading test_cleaned.csv...')
    df = pd.read_csv('/mnt/c/work/profiler_testing/test_cleaned.csv')
    X_ex, X_g, y, _ = process_dataframe(df, 'Test')
    
    X_ex, X_g = apply_scalers(X_ex, X_g, exon_scaler, global_scaler)
    
    dataset = ExonDataset(X_ex, X_g, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    
    all_probs = []
    all_targets = []
    
    logger.info('Running inference...')
    with torch.no_grad():
        for batch_X_ex, lengths, batch_X_g, batch_y in loader:
            batch_X_ex, batch_X_g = batch_X_ex.to(device), batch_X_g.to(device)
            logits, _ = model(batch_X_ex, lengths, batch_X_g)
            probs = torch.sigmoid(logits).squeeze(-1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    preds = (all_probs >= 0.40).astype(int)
    cm = confusion_matrix(all_targets, preds, labels=[0, 1])
    
    # Handle potentially single-class confusion matrices
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if len(np.unique(all_targets)) == 1:
            if np.unique(all_targets)[0] == 0:
                tn = cm[0, 0]
                fp = np.sum(preds == 1)
            else:
                tp = cm[0, 0]
                fn = np.sum(preds == 0)

    acc = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    
    logger.info(f'\n{"="*60}')
    logger.info(f' V6 Model Performance on Test Data')
    logger.info(f'{"="*60}')
    logger.info(f' TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    logger.info(f' Accuracy:  {acc:.4f}')
    logger.info(f' Precision: {prec:.4f}')
    logger.info(f' Recall:    {rec:.4f}')
    logger.info(f' F1 Score:  {f1:.4f}')
    logger.info(f'{"="*60}\n')

if __name__ == '__main__':
    main()
