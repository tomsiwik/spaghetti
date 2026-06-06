from utils.myddp import is_main_process, barrier

def debug_print_ids(ids, name, tokenizer):
    if is_main_process():
        res = f"{name}"
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for i, t in enumerate(tokens):
            res = f"{res}\n{i}: token_ids: {t}"
        res = f"{res}\nevidence"
        print(res)