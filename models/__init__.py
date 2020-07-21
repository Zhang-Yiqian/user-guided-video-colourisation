import importlib
from models.interaction_net import Inet
from models.propagation_net_v2 import Pnet


def create_Inet(opt):
    instance = Inet(opt)
    print("Interaction net is created")

    return instance

def create_Pnet(opt):
    instance = Inet(opt)
    print("Propagation net is created")

    return instance