import os

files_to_upgrade = {
    'evaluate_test_v4.py': 'evaluate_test_v5.py',
    'evaluate_6species_v4.py': 'evaluate_6species_v5.py',
    'dump_errors_v8.py': 'dump_errors_v5.py',
    'dump_validation_errors.py': 'dump_validation_errors_v5.py'
}

for old, new in files_to_upgrade.items():
    with open(f'/mnt/c/work/profiler_testing/dl_model/{old}', 'r') as f:
        content = f.read()

    # Replacements
    content = content.replace('from train_rnn_v4 import', 'from train_rnn_v5 import')
    content = content.replace('dl_orthology_model_v4.pth', 'dl_orthology_model_v5.pth')
    content = content.replace('ExonRNN(exon_dim=13)', "ExonRNN(exon_dim=checkpoint['exon_dim'], global_dim=checkpoint['global_dim'])")
    content = content.replace('X_ex, X_g, y = process_dataframe(', 'X_ex, X_g, y, _ = process_dataframe(')
    content = content.replace('logits = model(batch_X_ex, lengths, batch_X_g)', 'logits, _ = model(batch_X_ex, lengths, batch_X_g)')
    content = content.replace('logits = model(ex_tensor, len_tensor, g_tensor)', 'logits, _ = model(ex_tensor, len_tensor, g_tensor)')
    content = content.replace('V4 Model', 'V5 Model')
    content = content.replace('V4 PyTorch', 'V5 PyTorch')
    content = content.replace('v4_test_errors_detailed.csv', 'v5_test_errors_detailed.csv')
    
    with open(f'/mnt/c/work/profiler_testing/dl_model/{new}', 'w') as f:
        f.write(content)

print("Upgraded all scripts to V5.")
