from .helpers.dataloader import get_loader
from .helpers.metrics import make_metrics_dataframe
#from qgis.core import QgsProject, QgsRasterLayer, QgsMapLayer, QgsVectorFileWriter, QgsRasterFileWriter, QgsRasterPipe
from .helpers.constants import PROJECT_CONFIGS, AttributeDict, countTime

from .model import Middleware

# from helpers.dataloader import get_loader
# from helpers.metrics import make_metrics_dataframe
#
# from helpers.constants import PROJECT_CONFIGS, AttributeDict, countTime, calculate_average_time
#
# from model import Middleware
import os
import json
import pandas as pd
import shutil


def save_layers(path, names, layer, i):
    file_name = path + '/' + names[i] + "1" + '.tif'
    file_writer = QgsRasterFileWriter(file_name)
    pipe = QgsRasterPipe()
    provider = layer.dataProvider()

    if not pipe.set(provider.clone()):
        print("Cannot set pipe provider")
    file_writer.writeRaster(
        pipe,
        provider.xSize(),
        provider.ySize(),
        provider.extent(),
        provider.crs())


def clean_dir(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def save_map_layers(layers):
    path = os.path.join(os.getcwd(), "/root/.local/share/QGIS/QGIS3/profiles/default/python/plugins/changedetection/changesystem/AUB/test")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = ["A/", "B/"]
    clean_dir(path[:-5])
    for n in names:
        os.makedirs(os.path.join(path, n))

    for i, layer in enumerate(layers):
        if layer.type() == QgsMapLayer.RasterLayer:
            save_layers(path, names, layer, i)

    return path[:-5]


def read_json_config(file_path):
    abs_path = os.path.abspath(file_path)
    with open(abs_path, 'r') as file:
        data = json.load(file)
    return data


def update_config_with_data(data, output_dir, method, model, input_dir):
    data["datasets"]["AUB"] = input_dir
    data["results_dir"] = output_dir
    data["METHODS"] = [method]
    data["MODELS"] = [model]


def write_list_of_images(input_dir, list_of_layers_dir):
    list_dir = os.path.join(input_dir, list_of_layers_dir)
    os.makedirs(list_dir, exist_ok=True)
    test_dir = os.path.join(input_dir, "test/A")
    files = os.listdir(test_dir)
    file_path = os.path.join(list_dir, "test.txt")
    with open(file_path, 'w') as file:
        for image in files:
            file.write(image + '\n')


def write_json_config(file_path, data):
    abs_path = os.path.abspath(file_path)
    with open(abs_path, 'w') as file:
        json.dump(data, file, indent=2)


def edit_config(output_dir, method, model, layers):
    #list_of_layers_dir = "list"
    #config_data = read_json_config(PROJECT_CONFIGS)

    #input_dir = save_map_layers(layers)
    update_config_with_data(config_data, output_dir, method, model, input_dir)
    write_list_of_images(input_dir, list_of_layers_dir)
    write_json_config(PROJECT_CONFIGS, config_data)


def main():
    """ 
    Loop through Change Detection models and testing methods
    """
    
    #edit_config(output_dir)
    configs = open(PROJECT_CONFIGS)
    data = read_json_config(PROJECT_CONFIGS)
    print(data)
    input_dir = data["datasets"]["AUB"] 
    write_list_of_images(data["datasets"]["AUB"], "list")
    configs = AttributeDict(json.load(configs))
    dataloader = get_loader(configs.data_name, configs.batch_size)
    final_metrics = make_metrics_dataframe(configs.MODELS, configs.METHODS, configs.METRICS)
    configs.final_metrics = final_metrics
    data_dir = configs.datasets[f'{configs.data_name}']
    model = Middleware(configs)
    os.makedirs(configs.metrics_filename, exist_ok=True)

    columns = pd.MultiIndex.from_product([['Average Inference Time Per Image Pair'], configs.METHODS])
    df = pd.DataFrame(columns=columns, index=configs.MODELS)
    for model_name in configs.MODELS:
        print(f"Initialization of {model_name}.")
        for method in configs.METHODS:
          print(os.path.join(configs.results_dir, model_name, method, configs.data_name))
          os.makedirs(os.path.join(configs.results_dir, model_name, method, configs.data_name), exist_ok=True)
          time_per_one_pair = countTime(lambda: model.predict(model_name, method, dataloader), os.path.join(data_dir, configs.split, configs.first_image_dir))
          df.loc[model_name, ('Average Inference Time Per Image Pair', method)] = time_per_one_pair

    print("end")
if __name__ == '__main__':
    main()  

