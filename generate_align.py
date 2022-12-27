import argparse

# Add arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='Path to vocal alignment dataset')
ap.add_argument('-o', '--output', required=False, help='Path to generated feature')
args = ap.parse_args()

from setting_loaders import VocalSettings
from vocal import app

# Change settings to match arguments
config_path = 'defaults/vocal.yaml'
config = VocalSettings(config_path)
if args.output is not None:
    config.dataset.feature_save_path= args.output

# Generate features
va_transcription = app.VocalTranscription(config_path)
va_transcription.generate_feature(dataset_path=args.input, vocal_settings=config)