logs = {
    '0': {
        'forward': {
            'original': [], # contain operation objects
            'optimized': [] # contain operation objects
        },
        'backward': {
            'original': [], # contain operation objects
            'optimized': [] # contain operation objects
        },
        'loss': None,
    }
}

# operation objects example

operation_object = {
    'name': 'Linear1',
    'input': {
        'input1': [], # some tensor/array
        'input2': [] # some tensor/array
    },
    'output': [], # some tensor/array
}