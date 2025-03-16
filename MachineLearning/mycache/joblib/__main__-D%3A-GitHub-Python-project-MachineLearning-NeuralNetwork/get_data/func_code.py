# first line: 9
@mem.cache
def get_data():
    data = load_svmlight_file(
        r'mnist.scale')
    return data[0], data[1]
