import os
from os import path

# Drebin论文提供
DANGEROUS_API_SIMLI_TAGS_PERMISSIONS = {
    "Landroid/content/Intent;->setDataAndType": [],
    "Landroid/content/Intent;->setFlags": [],
    "Landroid/content/Intent;->addFlags": [],
    "Landroid/content/Intent;->putExtra": [],
    "Landroid/content/Intent;->init": [],
    "Ljava/lang/reflect": [],
    "Ljava/lang/Object;->getClass": [],
    "Ljava/lang/Class;->getConstructor": [],
    "Ljava/lang/Class;->getConstructors": [],
    "Ljava/lang/Class;->getDeclaredConstructor": [],
    "Ljava/lang/Class;->getDeclaredConstructors": [],
    "Ljava/lang/Class;->getField": [],
    "Ljava/lang/Class;->getFields": [],
    "Ljava/lang/Class;->getDeclaredField": [],
    "Ljava/lang/Class;->getDeclaredFields": [],
    "Ljava/lang/Class;->getMethod": [],
    "Ljava/lang/Class;->getMethods": [],
    "Ljava/lang/Class;->getDeclaredMethod": [],
    "Ljava/lang/Class;->getDeclaredMethods": [],
    "Ljavax/crypto": [],
    "Ljava/security/spec": [],
    "Ldalvik/system/DexClassLoader": [],
    "Ljava/lang/System;->loadLibrary": [],
    "Ljava/lang/Runtime": [],
    "Landroid/os/Environment;->getExternalStorageDirectory": ["android.permission.READ_EXTERNAL_STORAGE", "android.permission.WRITE_EXTERNAL_STORAGE"],
    "Landroid/telephony/TelephonyManager;->getDeviceId": ["android.permission.READ_PHONE_STATE"],
    "Landroid/telephony/TelephonyManager;->getSubscriberId": ["android.permission.READ_PHONE_STATE"],
    "setWifiEnabled": ["android.permission.CHANGE_WIFI_STATE"],
    "execHttpRequest": ["android.permission.INTERNET"],
    "getPackageInfo": [],
    "Landroid/content/Context;->getSystemService": [],
    "setWifiDisabled": ["android.permission.CHANGE_WIFI_STATE"],
    "Ljava/net/HttpURLconnection;->setRequestMethod": ["android.permission.INTERNET"],
    "Landroid/telephony/SmsMessage;->getMessageBody": ["android.permission.READ_SMS", "android.permission.RECEIVE_SMS"],
    "Ljava/io/IOException;->printStackTrace": [],
    "system/bin/su": [],
}


def retrive_files_set(base_dir, dir_ext, file_ext):
    """
    get file paths given the directory
    :param base_dir: basic directory
    :param dir_ext: directory append at the rear of base_dir
    :param file_ext: file extension
    :return: set of file paths. Avoid the repetition
    """

    def get_file_name(root_dir, file_ext):

        for dir_path, dir_names, file_names in os.walk(root_dir, topdown=True):
            for file_name in file_names:
                _ext = file_ext
                if os.path.splitext(file_name)[1] == _ext:
                    yield os.path.join(dir_path, file_name)
                elif "." not in file_ext:
                    _ext = "." + _ext

                    if os.path.splitext(file_name)[1] == _ext:
                        yield os.path.join(dir_path, file_name)
                    else:
                        pass
                else:
                    pass

    if file_ext is not None:
        file_exts = file_ext.split("|")
    else:
        file_exts = []
    file_path_list = list()
    for ext in file_exts:
        file_path_list.extend(get_file_name(os.path.join(base_dir, dir_ext), ext))
    # remove duplicate elements
    from collections import OrderedDict

    return list(OrderedDict.fromkeys(file_path_list))


def read_txt(path, mode="r"):
    if os.path.isfile(path):
        with open(path, mode) as f_r:
            lines = f_r.read().strip().splitlines()
            return lines
    else:
        raise ValueError("{} does not seen like a file path.\n".format(path))


def java_class_name2smali_name(cls):
    """
    Transform a typical xml format class into smali format

    :param cls: the input class name
    :rtype: string
    """
    if cls is None:
        return
    if not isinstance(cls, str):
        raise ValueError("Expected a string")

    return "L" + cls.replace(".", "/") + ";"


def generate_sensitive_api():
    """
    Generate sensitive API features.
    """
    # 这些敏感API是从Axplorer的权限映射文件中提取的
    dir_path = path.dirname(path.realpath(__file__))
    dir_to_axplorer_permissions_mp = path.join(dir_path + "/res/permissions/")
    txt_file_paths = list(retrive_files_set(dir_to_axplorer_permissions_mp, "", "txt"))

    sensitive_apis = set()
    sensitive_apis_permission = dict()

    for txt_file_path in txt_file_paths:
        file_name = path.basename(txt_file_path)
        if "cp-map" in file_name:
            text_lines = read_txt(txt_file_path)
            for line in text_lines:
                api_name = line.split(" ")[0].strip()
                class_name, method_name = api_name.rsplit(".", 1)
                api_name_smali = java_class_name2smali_name(class_name) + "->" + method_name
                sensitive_apis.add(api_name_smali)

                api_permission = line.split(" ")[1].strip()
                sensitive_apis_permission[api_name_smali] = [perm.strip() for perm in api_permission.split(",")]
        else:
            text_lines = read_txt(txt_file_path)
            for line in text_lines:
                api_name = line.split("::")[0].split("(")[0].strip()
                class_name, method_name = api_name.rsplit(".", 1)
                api_name_smali = java_class_name2smali_name(class_name) + "->" + method_name
                sensitive_apis.add(api_name_smali)

                api_permission = line.split("::")[1].strip()
                sensitive_apis_permission[api_name_smali] = [perm.strip() for perm in api_permission.split(",")]

    return {"sensitive_apis": list(sensitive_apis), "sensitive_apis_permission": sensitive_apis_permission}


if __name__ == "__main__":
    data = generate_sensitive_api()

    for key, value in DANGEROUS_API_SIMLI_TAGS_PERMISSIONS.items():
        if key not in data["sensitive_apis"]:
            data["sensitive_apis"].append(key)
        if key not in data["sensitive_apis_permission"]:
            data["sensitive_apis_permission"][key] = value

    import json

    with open("sensitive_api.json", "w") as f:
        json.dump(data["sensitive_apis"], f, indent=2, ensure_ascii=False)

    with open("sensitive_api_permission_map.json", "w") as f:
        json.dump(data["sensitive_apis_permission"], f, indent=2, ensure_ascii=False)
