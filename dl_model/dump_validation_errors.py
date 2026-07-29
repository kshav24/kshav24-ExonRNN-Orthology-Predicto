import torch
import pandas as pd
import numpy as np
import sys
import logging
from tqdm import tqdm

sys.path.insert(0, '/mnt/c/work/profiler_testing/dl_model')
sys.path.insert(0, '/mnt/c/work/profiler_testing')
from train_rnn_v4 import process_dataframe, ExonRNN, apply_scalers
import extract_features_minimal
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SPECIES_LIST = ['bisBis1', 'monDom5', 'myoLuc2', 'ornAna2', 'phyCat1', 'rn6']

def main():
    logger.info('Loading V4 PyTorch Model Checkpoint...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load('/mnt/c/work/profiler_testing/dl_model/dl_orthology_model_v4.pth', map_location=device, weights_only=False)
    model = ExonRNN(exon_dim=13).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    exon_scaler = StandardScaler()
    exon_scaler.mean_ = checkpoint['exon_scaler_mean']
    exon_scaler.scale_ = checkpoint['exon_scaler_scale']
    global_scaler = StandardScaler()
    global_scaler.mean_ = checkpoint['global_scaler_mean']
    global_scaler.scale_ = checkpoint['global_scaler_scale']
    
    error_df_records = []
    
    for species in SPECIES_LIST:
        logger.info(f'\n{"="*60}')
        logger.info(f' Tracking Errors for Species: {species}')
        logger.info(f'{"="*60}')
        
        csv_path = f'/mnt/c/work/profiler_testing/ensembl_validation/raw_validation_data_{species}.csv'
        try:
            df = pd.read_csv(csv_path)
            if 'label' not in df.columns:
                logger.error(f'No label column in {species}! Skipping...')
                continue
                
            grouped = df.groupby(['transcript_id', 'species'])
            metadata = [(tid, sp) for (tid, sp), _ in grouped]
            
            X_ex, X_g, y = process_dataframe(df, species)
            X_ex, X_g = apply_scalers(X_ex, X_g, exon_scaler, global_scaler)
            
            error_meta = []
            
            with torch.no_grad():
                for i in tqdm(range(len(y)), desc=f'Inference {species}'):
                    ex_tensor = torch.from_numpy(X_ex[i]).unsqueeze(0).to(device)
                    len_tensor = torch.tensor([X_ex[i].shape[0]])
                    g_tensor = torch.from_numpy(X_g[i]).unsqueeze(0).to(device)
                    target = y[i]
                    
                    logits = model(ex_tensor, len_tensor, g_tensor)
                    prob = torch.sigmoid(logits).item()
                    pred = 1 if prob >= 0.50 else 0
                    
                    if pred != target:
                        tid, sp = metadata[i]
                        error_type = 'FP' if pred == 1 else 'FN'
                        error_meta.append({'transcript_id': tid, 'species': sp, 'error_type': error_type, 'prob': prob})
                        
            logger.info(f'Found {len(error_meta)} errors in {species}. Re-extracting full feature dictionaries...')
            
            for err in tqdm(error_meta, desc=f'Extracting {species}'):
                tid, sp, err_type, prob = err['transcript_id'], err['species'], err['error_type'], err['prob']
                t_df = df[(df['transcript_id'] == tid) & (df['species'] == sp)].sort_values('exon_number')
                
                total_exons = len(t_df)
                for _, row in t_df.iterrows():
                    ref = str(row['ref_seq']) if pd.notnull(row['ref_seq']) else ''
                    query = str(row['query_seq']) if pd.notnull(row['query_seq']) else ''
                    
                    raw_feats = extract_features_minimal.extract_features(
                        ref, query, 
                        exon_number=row.get('exon_number', 1), 
                        total_exons=total_exons
                    )
                    
                    record = {
                        'transcript_id': tid,
                        'species': sp,
                        'error_type': err_type,
                        'model_probability': round(prob, 4),
                        'exon_number': row.get('exon_number', 1),
                        'ref_seq': ref,
                        'query_seq': query,
                    }
                    record.update(raw_feats)
                    error_df_records.append(record)
                    
        except Exception as e:
            logger.error(f'Failed to process {species}: {e}')

    out_path = '/mnt/c/work/profiler_testing/dl_model/validation_errors_detailed.csv'
    pd.DataFrame(error_df_records).to_csv(out_path, index=False)
    logger.info(f'\nSuccessfully dumped {len(error_df_records)} error exons across all validation species to {out_path}')

if __name__ == '__main__':
    main()
