import pickle

class SymbolStorage:
    symbol_data = {}

    @staticmethod
    def get_data(filename):
        if filename in SymbolStorage.symbol_data:
            return SymbolStorage.symbol_data[filename]
        else:
            with open(filename, 'rb') as file:
                data = pickle.load(file)
                SymbolStorage.symbol_data[filename] = data
            return data
