import torch

def load_partial_parameters(model, checkpoint):
    """Load partial pretrained model parameters."""
    loaded_params = dict()
    for name, val in checkpoint.items():
        name_new = name.replace('module.', '') if 'module.' in name else name
        loaded_params[name_new] = val
    
    model_params = dict()
    for name, val in model.state_dict().items():
        name_new = name.replace('module.', '') if 'module.' in name else name
        model_params[name_new] = val
        
    valid_params = dict()
    valid_num_condition_encoder = 0
    for src_name, src_val in loaded_params.items():
        if src_name not in model_params.keys():
            continue
        src_val_shape = ', '.join(map(str, src_val.size()))
        dst_val = model_params[src_name]
        dst_val_shape = ', '.join(map(str, dst_val.size()))
        if src_val_shape != dst_val_shape:
            continue
        suffix = 'module.' if hasattr(model, "module") else ''
        valid_params[suffix + src_name] = src_val
    
    percentage = len(valid_params) / len(model_params) * 100.0
    model.load_state_dict(valid_params, strict=False)
    print('Loading {:.5f}%% pretrained weights'.format(percentage))
    return model, percentage