from io import BytesIO
import Ipython
from matplotlib import pyplot as plt
from model_loading import loading_model


def vis_segmentation_stream(image, seg_map, index):

    plt.figure(figsize=(12, 7))

    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay | frame #%d'%index)
    plt.grid('off')
    plt.tight_layout()

    # Show visualization in a streaming fashion.
    f = BytesIO()
    plt.savefig(f, format='jpeg')
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
    f.close()
    plt.close()

def run_visualization_video(frame, index):
    """Inferences DeepLab model on a video file and stream the visualization."""
    original_im = Image.fromarray(frame[..., ::-1])
    MODEL = loading_model(DeepLabModel)
    seg_map = MODEL.run(original_im)
    vis_segmentation_stream(original_im, seg_map, index)


if __name__ == "__main__":
    SAMPLE_VIDEO = 'mit_driveseg_sample.mp4'
    if not os.path.isfile(SAMPLE_VIDEO):
        print('downloading the sample video...')
        SAMPLE_VIDEO = urllib.request.urlretrieve('https://github.com/lexfridman/mit-deep-learning/raw/master/tutorial_driving_scene_segmentation/mit_driveseg_sample.mp4')[0]
    print('running deeplab on the sample video...')

    video = cv.VideoCapture(SAMPLE_VIDEO)
    # num_frames = 598  # uncomment to use the full sample video
    num_frames = 350

    try:
        for i in range(num_frames):
            _, frame = video.read()
            if not _: break
            run_visualization_video(frame, i)
            IPython.display.clear_output(wait=True)
    except KeyboardInterrupt:
        plt.close()
        print("Stream stopped.")
