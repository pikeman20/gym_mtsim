import os


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FOREX_DATA_PATH = os.path.join(DATA_DIR, 'symbols_forex.pkl')
STOCKS_DATA_PATH = os.path.join(DATA_DIR, 'symbols_stocks.pkl')
CRYPTO_DATA_PATH = os.path.join(DATA_DIR, 'symbols_crypto.pkl')
MIXED_DATA_PATH = os.path.join(DATA_DIR, 'symbols_mixed.pkl')
BINANCE_DATA_PATH = os.path.join(DATA_DIR, 'symbols_binance.pkl')
BINANCE_DATA_GPU_PATH = os.path.join(DATA_DIR, 'symbols_binance_gpu.pkl')
BINANCE_SYMBOL_PATH = os.path.join(DATA_DIR, 'binance_symbol_info.json')
BINANCE_SYMBOL_CSV_PATH = os.path.join(DATA_DIR, 'csv/')
