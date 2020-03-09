import importlib
from models.interaction_net import Inet


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_Inet(opt):
    instance = Inet()
    print("Interaction net was created")

    return instance

