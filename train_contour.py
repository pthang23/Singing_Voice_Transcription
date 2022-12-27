# Add arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--feature', required=True, help='Path to the folder of extracted feature')
ap.add_argument('-i', '--input_model', required=False, help='If given, the training will continue to fine-tune the pre-trained model')
ap.add_argument('-o', '--output_model', required=False, help='Name for the output model')
ap.add_argument('-e', '--epochs', required=False, help='Number of training epochs')
ap.add_argument('-b', '--batch_size', required=False, help='Batch size of each training step')
ap.add_argument('-s', '--steps', required=False, help='Number of step each training epochs (virtual epochs)')
ap.add_argument('-vb', '--val_batch_size', required=False, help='Batch size of each validation step')
ap.add_argument('-vs', '--val_steps', required=False, help='Number of step each validation epochs (virtual epochs)')
ap.add_argument('-lr', '--learning_rate', required=False, help='Initial learning rate')
ap.add_argument('--early_stop', required=False, help='Stop the training if validation accuracy does not improve over the given number of epochs')
args = ap.parse_args()

from setting_loaders import VocalContourSettings
from vocal_contour import app as vcapp

# Change settings to match arguments
config_path = 'defaults/vocal_contour.yaml'
config = VocalContourSettings(config_path)
output_model = args.output_model if args.output_model is not None else None
input_model = args.input_model if args.input_model is not None else None
if args.epochs is not None:
    config.training.epoch = int(args.epochs)
if args.batch_size is not None:
    config.training.batch_size = int(args.batch_size)
if args.steps is not None:
    config.training.steps = int(args.steps)
if args.val_batch_size is not None:
    config.training.val_batch_size = int(args.val_batch_size)
if args.val_steps is not None:
    config.training.val_steps = int(args.val_steps)
if args.learning_rate is not None:
    config.training.init_learning_rate = float(args.learning_rate)
if args.early_stop is not None:
    config.training.early_stop = int(args.early_stop)

# Training
vc_transcription = vcapp.VocalContourTranscription(config_path)
vc_transcription.train(feature_folder=args.feature, model_name=output_model, input_model_path=input_model, vocalcontour_settings=config)