import warnings

class DefaultConfig(object):
    def __init__(self):
      self.phase = 'I_prev'
      self.gpu_ids = 1
      self.no_prev = False
      self.is_regression = True
      self.load_P = False
      self.P_path = None
      self.loadSize = 256
      self.fineSize = 224
      self.batch_size = 32
      self.batch_size_val = 32
        
      self.isTrain = 'True'
      self.val_freq = 150
      self.print_freq = 20
      self.epoch_count = 10
      self.lr = 0.00001
      self.beta1 = 0.9
      self.niter = 15
      self.epoch_count = 0
      self.niter_decay = 0
      self.weight_decay = 0.001
      self.sample_p = 0.125
      self.l_norm = 100.0
      self.l_cent = 50.0
      self.ab_norm = 110.0
      self.ab_quant = 10.0
      self.ab_max = 110.0
      self.A = 2 * self.ab_max / self.ab_quant + 1
      self.sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      # self.sample_Ps = [7]
      self.mask_cent = 0.5
      self.mean_kernel = 32
      self.seed = 2020
      self.dataroot_train = None
      self.dataroot_val = None
        
      self.load_I = True
      self.I_path = '/home/zhangy23/github/user-guided-video-colourisation/snapshot/I_auto_ep14_val_2_27.26.pkl'
      self.save_root = '/home/zhangy23/github/user-guided-video-colourisation/snapshot/prev_I_lr_0.00001_wd/'
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
                