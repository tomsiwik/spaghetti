def freeze(metamodel):
    for param in metamodel.parameters():
        param.requires_grad = False  # freeze the meta model except mem_tokens
    metamodel.model.mem_tokens.requires_grad = True