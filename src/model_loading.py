import tensorflow as tf
import os
from six.moves import urllib
from model import DeepLabModel

def loading_model(DeepLabModel):
    MODEL_NAME = 'mobilenetv2_coco_cityscapes_trainfine'
    #MODEL_NAME = 'xception65_cityscapes_trainfine'
    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_cityscapes_trainfine':
            'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
        'xception65_cityscapes_trainfine':
            'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
    }

    _TARBALL_NAME = 'deeplab_model.tar.gz'

    model_dir = tempfile.mkdtemp()
    tf.io.gfile.makedirs(model_dir)
    download_path = os.path.join(model_dir, _TARBALL_NAME)
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
    MODEL = DeepLabModel(download_path)
    return MODEL
