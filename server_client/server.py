import socket
import io
import json
import struct
import os
import shutil
import threading
import changedetection.changesystem.main as main
from changedetection.changesystem.helpers.constants import PROJECT_CONFIGS
SERVER_HOST = '192.168.205.151'
SERVER_PORT = 12345

lock = threading.Lock()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(1)
print(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")


def clean_dir(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def write_json_config(file_path, data):
    abs_path = os.path.abspath(file_path)
    with open(abs_path, 'w') as file:
        json.dump(data, file, indent=2)


def read_json_config(file_path):
    abs_path = os.path.abspath(file_path)
    with open(abs_path, 'r') as file:
        data = json.load(file)
    return data


def read_tiff(file_path):        
    with open(file_path, 'rb') as file:
        return file.read()       


def save_tiff(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
            file.write(data)


def get_json_size(client_socket):
    received_data = ""
    json_length = client_socket.recv(4)
    strings = json_length.decode('utf8')
    print(json_length)
    json_length = int(strings)
    print("json length  = ", json_length)
    return json_length


def get_json(client_socket, json_length):
    recieved_data = client_socket.recv(json_length)
    fix_bytes_value = recieved_data.replace(b"'", b'"')
    json_data = json.load(io.BytesIO(fix_bytes_value))
    print("[*] Received JSON data from client:")
    print(json_data)
    json_data["results_dir"] = "changedetection/changesystem/output"
    return json_data


def get_tiff_size(client_socket):
    tiff_length_bytes = client_socket.recv(4)                                                          
    tiff_length = int.from_bytes(tiff_length_bytes, 'big')
    return tiff_length
              

def get_tiff(client_socket, tiff_length):
    tiff_data = b''
    print(tiff_length)
    while len(tiff_data) < tiff_length:
        tiff_data_chunk = client_socket.recv(tiff_length - len(tiff_data))
        if not tiff_data_chunk:
            break
        tiff_data += tiff_data_chunk
    return tiff_data


def send_result(client_socket):
    data = read_json_config(PROJECT_CONFIGS)
    dirname = os.path.join(data["results_dir"],  data["MODELS"][0], data["METHODS"][0],  \
   "AUB/1.tiff")
    tiff_data = read_tiff(dirname)
    tiff_length = len(tiff_data)
    client_socket.send(tiff_length.to_bytes(4, 'big'))
    client_socket.send(tiff_data)          


def handle_client(client_socket):
    try:
        with lock:
            json_length = get_json_size(client_socket)
            clean_dir("changedetection/changesystem/AUB/")
            json_data = get_json(client_socket, json_length)
            write_json_config(PROJECT_CONFIGS, json_data)
            dir_names = ["changedetection/changesystem/AUB/test/A/1.tiff",\
            "changedetection/changesystem/AUB/test/B/1.tiff"]
            for name in dir_names:
                tiff_length = get_tiff_size(client_socket)
                tiff_data = get_tiff(client_socket, tiff_length)
                save_tiff(tiff_data, name)
                print("TIFF image received and saved.")
            main.main()
            send_result(client_socket) 
    except json.decoder.JSONDecodeError as e:
        print("Error decoding JSON:", e)
    finally:
        client_socket.close()   
try:
    while True:
        client_socket, client_address = server_socket.accept()
        print(f"[*] Accepted connection from {client_address[0]}:{client_address[1]}")
        threading.Thread(target=handle_client, args=(client_socket,)).start()

except:
    print(" Closing server socket.")
    server_socket.close()
