import warnings

class DefaultConfig(object):
    def __init__(self):
      self.phase = 'I_auto'
      self.gpu_ids = 1
      # self.load_model = False
      # self.model_path = '/home/zhangy23/github/user-guided-video-colourisation/snapshot/I_auto_ep14_val_6.pkl'
      self.no_prev = True
      self.is_regression = True
      
      self.loadSize = 256
      self.fineSize = 224
      self.batch_size = 40
      self.batch_size_val = 8
        
      self.isTrain = 'train'
      self.val_freq = 300
      self.print_freq = 20
      self.epoch_count = 10
      self.lr = 0.0001
      self.beta1 = 0.9
      self.niter = 15
      self.epoch_count = 0
      self.niter_decay = 0
      self.sample_p = 0.125
      self.l_norm = 100.0
      self.l_cent = 50.0
      self.ab_norm = 110.0
      self.ab_quant = 10.0
      self.ab_max = 110.0
      self.A = 2 * self.ab_max / self.ab_quant + 1
      self.sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      self.mask_cent = 0.5
      # self.env = 'default'
      self.seed = 2020
      self.dataroot_train = None
      self.dataroot_val = None
    
      self.save_root = '/home/zhangy23/github/user-guided-video-colourisation/snapshot/auto_lr_0.0001/'
      self.dataroot = '/home/zhangy23/dataset/coco/'

    def parse(self,kwargs):

        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k,getattr(self,k))