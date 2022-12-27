import os

from utils import load_yaml, json_serializable

SETTING_DIR = f'{os.path.split(__file__)[0]}/defaults'


class Settings:
    default_setting_file = None

    def __init__(self, conf_path=None):
        # Load default settings
        if conf_path is not None:
            self.from_json(load_yaml(conf_path))  # pylint: disable=E1101
        else:
            conf_path = os.path.join(SETTING_DIR, self.default_setting_file)
            self.from_json(load_yaml(conf_path))  # pylint: disable=E1101


@json_serializable(key_path="./General", value_path="./Value")
class VocalContourSettings(Settings):
    default_setting_file: str = "vocal_contour.yaml"

    def __init__(self, conf_path=None):
        self.transcription_mode: str = None
        self.checkpoint_path: str = None
        self.feature = self.VocalContourFeature()
        self.dataset = self.VocalContourDataset()
        self.model = self.VocalContourModel()
        self.training = self.VocalContourTraining()

        super().__init__(conf_path=conf_path)

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalContourFeature():
        def __init__(self):
            self.hop_size: float = None
            self.sampling_rate: int = None
            self.window_size: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalContourDataset():
        def __init__(self):
            self.save_path: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalContourModel():
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalContourTraining():
        def __init__(self):
            self.epoch: int = None
            self.early_stop: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.timesteps: int = None


@json_serializable(key_path="./General", value_path="./Value")
class VocalSettings(Settings):
    default_setting_file: str = "vocal.yaml"

    def __init__(self, conf_path=None):
        self.transcription_mode: str = None
        self.checkpoint_path: dict = None
        self.feature = self.VocalFeature()
        self.dataset = self.VocalDataset()
        self.model = self.VocalModel()
        self.inference = self.VocalInference()
        self.training = self.VocalTraining()

        super().__init__(conf_path=conf_path)

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalFeature:
        def __init__(self):
            self.hop_size: float = None
            self.sampling_rate: int = None
            self.frequency_resolution: float = None
            self.frequency_center: float = None
            self.time_center: float = None
            self.gamma: list = None
            self.bins_per_octave: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalDataset:
        def __init__(self):
            self.save_path: str = None
            self.feature_save_path: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalModel:
        def __init__(self):
            self.save_prefix: str = None
            self.save_path: str = None
            self.min_kernel_size: int = None
            self.depth: int = None
            self.shake_drop: bool = True
            self.alpha: int = None
            self.semi_loss_weight: float = None
            self.semi_xi: float = None
            self.semi_epsilon: float = None
            self.semi_iterations: int = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalInference:
        def __init__(self):
            self.context_length: int = None
            self.threshold: float = None
            self.min_duration: float = None
            self.pitch_model: str = None

    @json_serializable(key_path="./Settings", value_path="./Value")
    class VocalTraining:
        def __init__(self):
            self.epoch: int = None
            self.steps: int = None
            self.val_steps: int = None
            self.batch_size: int = None
            self.val_batch_size: int = None
            self.early_stop: int = None
            self.init_learning_rate: float = None
            self.context_length: int = None