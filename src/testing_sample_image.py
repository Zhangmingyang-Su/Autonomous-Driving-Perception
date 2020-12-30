from segmentation_color import vis_segmentation
from model_loading import loading_model

def run_visualization(SAMPLE_IMAGE, MODEL):
    """Inferences DeepLab model and visualizes result."""
    original_im = Image.open(SAMPLE_IMAGE)
    seg_map = MODEL.run(original_im)
    vis_segmentation(original_im, seg_map)

if __name__ == "__main__":
    MODEL = loading_model(DeepLabModel)
    SAMPLE_IMAGE = 'mit_driveseg_sample.png'
    if not os.path.isfile(SAMPLE_IMAGE):
        print('downloading the sample image...')
        SAMPLE_IMAGE = urllib.request.urlretrieve('https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/mit_driveseg_sample.png?raw=true')[0]
    print('running deeplab on the sample image...')

    run_visualization(SAMPLE_IMAGE, MODEL)
