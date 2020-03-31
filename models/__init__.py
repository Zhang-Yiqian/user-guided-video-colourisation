import importlib
from models.interaction_net import Inet


def create_Inet(opt):
    instance = Inet(opt)
    print("Interaction net was created")

    return instance

