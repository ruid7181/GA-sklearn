class GADefaultHyperParameters:
    variants = {
        'mini': {'idu_points': 0,
                 'seq_len': 81,
                 'n_attn_layer': 1},

        'small': {'idu_points': 4,
                  'seq_len': 144,
                  'n_attn_layer': 2},

        'large': {'idu_points': 8,
                  'seq_len': 256,
                  'n_attn_layer': 3},
    }
