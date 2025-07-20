import os
import configparser
from colorama import Fore, init
init(autoreset=True)

class ConfigManager:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.config_file = os.path.join(project_root, 'config', 'config.ini')
        self.config = configparser.ConfigParser()
        
        self.default_config = {
            'Settings': {
                'load_model': 'False',
                'auto_save_model': 'False',
                'optimizer(sgd, sgd_momentum, adam)': 'adam',
                'epoch_count': '20',
                'train_batch_size': '64',
                'test_batch_size': '64',
                'show_confusion_matrix': 'False',
                'learning_rate': '0.01',
                'momentum': '0.9',
                'early_stopping_patience': '150',
                'accepting_accuracy_rate': '0.97',
            }
        }

        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            self.config.read_dict(self.default_config)
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
            print(Fore.LIGHTYELLOW_EX + 'Default configuration was created\n')
            self.config.read(self.config_file)
        else:
            print(Fore.LIGHTYELLOW_EX + 'Configuration configured from config file\n')
            self.config.read(self.config_file)

    @property
    def load_model(self):
        return self.config.getboolean('Settings', 'load_model')
    
    @property
    def auto_save_model(self):
        return self.config.getboolean('Settings', 'auto_save_model')
    
    @property
    def optimizer(self): 
        return self.config.get('Settings', 'optimizer(sgd, sgd_momentum, adam)')

    @property
    def epoch_count(self):
        return self.config.getint('Settings', 'epoch_count')
    
    @property
    def train_batch_size(self):
        return self.config.getint('Settings', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self.config.getint('Settings', 'test_batch_size')
    
    @property
    def show_confusion_matrix(self):
        return self.config.getboolean('Settings', 'show_confusion_matrix')

    @property
    def learning_rate(self):
        return self.config.getfloat('Settings', 'learning_rate')
    
    @property
    def momentum(self):
        return self.config.getfloat('Settings', 'momentum')
    
    @property
    def early_stopping_patience(self):
        return self.config.getint('Settings', 'early_stopping_patience')
    
    @property
    def accepting_accuracy_rate(self):  
        return self.config.getfloat('Settings', 'accepting_accuracy_rate')

