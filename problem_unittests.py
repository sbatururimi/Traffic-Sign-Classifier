import numpy as np

def __print_success_message():
    print('Tests passed')

def test_normalize(normalize):
    test_shape = (np.random.choice(range(1000)), 32, 32, 3)
    test_numbers = np.random.choice(range(256), test_shape)
    normalize_out = normalize(test_numbers)

    assert type(normalize_out).__module__ == np.__name__, \
        'Not Numpy Object'

    assert normalize_out.shape == test_shape, \
        'Incorrect Shape. {} shape found'.format(normalize_out.shape)

    assert normalize_out.max() <= 1 and normalize_out.min() >= 0, \
        'Incorrect Range. {} to {} found'.format(normalize_out.min(), normalize_out.max())

    __print_success_message()

def test_grayscale(grayscale):
    test_shape = (np.random.choice(range(1000)), 32, 32, 3)
    test_numbers = np.random.choice(range(256), test_shape)
    grayscale_out = grayscale(test_numbers)

    assert type(grayscale_out).__module__ == np.__name__, \
        'Not Numpy Object'
        
    assert len(grayscale_out.shape) < 2,  \
        'Incorrect Shape. {} shape found'.format(grayscale_out.shape)

    __print_success_message()

