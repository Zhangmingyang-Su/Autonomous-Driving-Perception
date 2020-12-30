import tarfile
import cv2 as cv
from six.moves import urllib
from segmentation_color import vis_segmentation


class DriveSeg(object):
    """Class to load MIT DriveSeg Dataset."""

    def __init__(self, tarball_path):
        self.tar_file = tarfile.open(tarball_path)
        self.tar_info = self.tar_file.getmembers()

    def fetch(self, index):
        """Get ground truth by index.

        Args:
            index: The frame number.

        Returns:
            gt: Ground truth segmentation map.
        """
        tar_info = self.tar_info[index + 1]  # exclude index 0 which is the parent directory
        file_handle = self.tar_file.extractfile(tar_info)
        gt = np.fromstring(file_handle.read(), np.uint8)
        gt = cv.imdecode(gt, cv.IMREAD_COLOR)
        gt = gt[:, :, 0]  # select a single channel from the 3-channel image
        gt[gt==255] = 19  # void class, does not count for accuracy
        return gt

if __name__ == "__main__":
    SAMPLE_GT = 'mit_driveseg_sample_gt.tar.gz'
    if not os.path.isfile(SAMPLE_GT):
        print('downloading the sample ground truth...')
        SAMPLE_GT = urllib.request.urlretrieve('https://github.com/lexfridman/mit-deep-learning/raw/master/tutorial_driving_scene_segmentation/mit_driveseg_sample_gt.tar.gz')[0]

    dataset = DriveSeg(SAMPLE_GT)
    print('visualizing ground truth annotation on the sample image...')

    original_im = Image.open(SAMPLE_IMAGE)
    gt = dataset.fetch(0)  # sample image is frame 0
    vis_segmentation(original_im, gt)
