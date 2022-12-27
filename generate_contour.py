import argparse

# Add arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='Path to vocal contour dataset')
ap.add_argument('-o', '--output', required=False, help='Path to generated feature')
args = ap.parse_args()

from setting_loaders import VocalContourSettings
from vocal_contour import app as vcapp

# Change settings to match arguments
config_path = 'defaults/vocal_contour.yaml'
config = VocalContourSettings(config_path)
if args.output is not None:
    config.dataset.feature_save_path = args.output

# Generate features
vc_transcription = vcapp.VocalContourTranscription(config_path)
vc_transcription.generate_feature(dataset_path=args.input, vocalcontour_settings=config)