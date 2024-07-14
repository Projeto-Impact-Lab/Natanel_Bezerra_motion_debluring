
class Config:

    path_root = "/home/natanael.oliveira/deblurring_images"
    modules = f'{path_root}/modules'
    config = f"{path_root}/config"
    models = f"{path_root}/models"
    notebooks = f"{path_root}/notebooks"
    utils = f"{path_root}/utils"
    dataset_name = "goPro"
    dataset_extension = ".zip"
    dataset_file = f'{dataset_name}{dataset_extension}'
    downloads = f"{path_root}/downloads"
    url_download = "https://drive.google.com/file/d/1l6fZoGYa2ZMgMfUyTav6rruLXCY2plI1/view?usp=sharing"
    results_seach = f"{path_root}/results_seach"
    config_telegram = f"{path_root}/config.json"

    def __init__(self) -> None:
        pass
