# Singing Voice Transcription
Singing Voice Transcription is a module that aim to automatic transcript singing voice into music note. Given a polyphonic music, it is able to find and extract main melody at note-level from this music singing voice. You can find some demo samples [here](https://drive.google.com/drive/folders/1o-FqYGEZao_5H8FRuiHVoU4RqJQnFOBo?usp=sharing).

This module is referenced and improved from a part of [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) by [Music and Culture Technology (MCT) Lab](https://github.com/Music-and-Culture-Technology-Lab).

## Installation
Before starting, make sure you have conda or [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, then create new environment:
```bash
conda create -n voice_transcription python=3.7
conda activate voice_transcription
```
First, install [Spleeter by deezer](https://github.com/deezer/spleeter) and its required system packages:
```bash
conda install -c conda-forge ffmpeg libsndfile fluidsynth
conda install numpy Cython
pip install spleeter==2.3.0
```
Then install this module and dependencies:
```bash
git clone https://github.com/pthang23/Singing_Voice_Transcription
cd Singing_Voice_Transcription
pip install -r requirements.txt
```
Finally, download pretrained models from [here](https://drive.google.com/file/d/1y3M_rutkUW5xvp88z8eoFyzRbKwRhfa3/view?usp=sharing), put in `Singing_Voice_Transcription` and unzip it. Then you can access all the module features.

## Transcribe Music
Transcribe a single audio by running the command, output will be saved in MIDI format with the same basename as the given audio:
```bash
python transcribe.py -i <path/to/input/audio.wav> -o <path/to/output/folder>
```
You can also view more transcribe options with `--help` command

## Training
You can train a our module using your own custom dataset. This module contain 2 main part:
- **Vocal Alignment**: align onset, offset of each note
- **Vocal Contour Estimation**: segment pitch line

Before training, you need to make sure the data structure look like following:
<pre>
 |  dataset
 |  ├── audios
 |  │   └── audio1.wav ...
 |  └── labels
 |      └── label1.csv ...
</pre>
**Vocal Alignment** label file contain 3 columns: <i>onset</i>, <i>offset</i>, <i>midi_pitch</i><br>
**Vocal Contour Estimation** label file contain 2 columns: <i>onset</i>, <i>pitch (hz)</i>

### Training Vocal Alignment
First of all, generate the features that are necessary for training and testing. You can use simultaneously semi-supervised learning with unlabeled dataset, just eliminate **'labels'** part form the above data structure:
```bash
python generate_align.py -i <path/to/dataset/folder> -o <path/to/feature/folder>
```
The processed labeled features will be stored in `<path/to/feature/folder>/train_feature` and `<path/to/feature/folder>/test_feature`. The semi-supervised feature will be stored in `<path/to/feature/folder>/semi_feature`.

Then training a new model or continue to train on a pretrained model:
```bash
python train_align.py -f <path/to/train/feature/folder> -fs <path/to/semi/feature/folder> -i <path/to/pretrained/model>
```
You can view more training options with `--help` command or access `defaults/vocal.yaml`

### Training Vocal Contour Estimation
You also need to generate feature first, the processed features will be stored in `<path/to/feature/folder>/train_feature` and `<path/to/feature/folder>/test_feature`:
```bash
python generate_contour.py -i <path/to/dataset/folder> -o <path/to/feature/folder>
```
Then training from scratch or finetuning contour model:
```bash
python train_contour.py -f <path/to/train/feature/folder> -i <path/to/pretrained/model>
```
Once more time check `--help` command or access `defaults/vocal_contour.yaml` if you want to view more training options

## Reference
- [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) by Music and Culture Technology (MCT) Lab<br>
- [Spleeter](https://github.com/deezer/spleeter) by deezer<br>
- [TONAS](https://zenodo.org/record/1290722#.Y6q2RadBxH4), [VOCADITO](https://zenodo.org/record/5578807#.Y6q2hKdBxH4), [CSD](https://zenodo.org/record/4785016#.Y6q22adBxH4), [MEDLEYDB](https://medleydb.weebly.com/) and [MIR1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) dataset
