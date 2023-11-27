def replace_all_module(model, if_replace_hook, get_new_module):
    for name, module in model.named_children():
        if if_replace_hook(name, module):
            tgt_module = get_new_module(name, module)
            setattr(model, name, tgt_module)
        else:
            replace_all_module(module, if_replace_hook, get_new_module)
    return model