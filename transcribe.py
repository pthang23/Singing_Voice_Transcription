# Add arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='Path to input audio (wav)')
ap.add_argument('-o', '--output', required=False, help='Path to the folder will contain predictions')
ap.add_argument('-m', '--model', required=False, help='Path to the transcribe model')
args = ap.parse_args()

from vocal import app

# Change settings to match arguments
config_path = 'defaults/vocal.yaml'
model = args.model if args.model is not None else None
output = args.output if args.output is not None else "./"

# Transcribe
va_transcription = app.VocalTranscription(config_path)
va_transcription.transcribe(input_audio=args.input, model_path=model, output=output)
