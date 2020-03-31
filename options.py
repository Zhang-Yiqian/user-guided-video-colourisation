import warnings

class DefaultConfig(object):
    def __init__(self):
      self.phase = 'I_auto'

      self.loadSize = 256
      self.fineSize = 224
      self.batch_size = 64
      self.batch_size_val = 32
      self.gpu_ids = 0
      self.isTrain = 'train'
      self.val_freq = 50
      self.print_freq = 5
      self.load_model = False
      self.epoch_count = 10
      self.lr = 0.0001
      self.weight_decay = 1e-4 
      self.niter =10
      self.epoch_count = 0
      self.niter_decay = 0
      self.sample_p = 0.125
      # self.l_norm = 100.0
      # self.l_cent = 50.0
      # self.ab_norm = 110.0
      self.sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      self.mask_cent = 0.5
      # self.env = 'default'
      self.seed = 2020
      self.dataroot_train = None
      self.dataroot_val = None
      self.save_root = '/content/gdrive/My Drive/user-guided-video-colourisation/model/'
      self.dataroot = '/content/gdrive/My Drive/dataset/flickr30k/'

    def parse(self,kwargs):

        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k,getattr(self,k))