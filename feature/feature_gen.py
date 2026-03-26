from os import path, getcwd
import warnings
from androguard.misc import AnalyzeAPK
from feature.utils import dump_pickle, read_pickle
from feature.feature_util import get_permissions, get_components, get_providers, get_intent_actions, get_hardwares, get_apis

import sys
from loguru import logger

logger.remove()  # Remove all handlers
logger.add(sys.stderr, level="ERROR")  # Add the desired log level


def apk2feat_wrapper(kwargs):
    try:
        return apk2features(*kwargs)
    except Exception as e:
        return e


def apk2features(apk_path, max_number_of_smali_files=10000, saving_path=None):
    """
    Extract APK features, including dangerous permissions,
    suspicious intent actions, restricted APIs, and dangerous APIs.

    Each permission: Android permission.
    Each intent action: intent action + '#.tag.#' + related information of that intent.
    Each API: invoke type + ' ' + class name + '->' + method name +
    parameters + return type + '#.tag.#' + related information of the
    class name and method definition path.
    Parameters:

    apk_path: String path to the APK file, otherwise an error is raised.
    max_number_of_smali_files: Integer, maximum number of smali files.
    saving_path: String path to where results are saved.
    """

    if not isinstance(apk_path, str):
        raise ValueError("Expected a path, but got {}".format(type(apk_path)))
    if not path.exists(apk_path):
        raise FileNotFoundError("Cannot find an apk file by following the path {}.".format(apk_path))
    if saving_path is None:
        warnings.warn("Save the features in current direction:{}".format(getcwd()))
        saving_path = path.join(getcwd(), "api-graph")

    try:
        apk_path = path.abspath(apk_path)
        a, d, dx = AnalyzeAPK(apk_path)  # a: app; d: dex; dx: analysis of dex
    except Exception as e:
        raise ValueError("Fail to read and analyze the apk {}:{} ".format(apk_path, str(e)))

    # ------------- Start extracting all feature types -------------------#
    # 1. get permissions
    try:
        permission_list = get_permissions(a)
    except Exception as e:
        raise ValueError("Fail to extract permissions {}:{} ".format(apk_path, str(e)))

    # 2. get components except for providers
    try:
        component_list = get_components(a)
    except Exception as e:
        raise ValueError("Fail to extract components {}:{} ".format(apk_path, str(e)))

    # 3.get providers
    try:
        provider_list = get_providers(a)
    except Exception as e:
        raise ValueError("Fail to extract providers {}:{} ".format(apk_path, str(e)))

    # 4. get intent actions
    try:
        intent_actions = get_intent_actions(a)
    except Exception as e:
        raise ValueError("Fail to extract intents {}:{} ".format(apk_path, str(e)))

    # 5. get hardware
    try:
        hardware_list = get_hardwares(a)
    except Exception as e:
        raise ValueError("Fail to extract hardware {}:{} ".format(apk_path, str(e)))

    # 6. get apis
    try:
        api_sequences = get_apis(d, max_number_of_smali_files)
    except Exception as e:
        raise ValueError("Fail to extract apis {}:{} ".format(apk_path, str(e)))
    # ---------------- Finish extracting all feature types -----------------#

    # features = []
    # features.extend(permission_list + component_list + provider_list + intent_actions + hardware_list)
    # features.extend(api_sequences)

    # save_to_disk(features, saving_path)
    # if len(features) <= 0:
    #     warnings.warn("No features found: " + apk_path)

    res = {
        "permissions": permission_list,
        "components": component_list,
        "providers": provider_list,
        "intent_actions": intent_actions,
        "hardware": hardware_list,
        "api_sequences": api_sequences,
    }

    apis = set()
    for item in api_sequences:
        for api in item:
            apis.add(api)

    print(len(apis))

    import json

    with open(apk_path + "123.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

    return saving_path


def save_to_disk(data, saving_path):
    dump_pickle(data, saving_path)


def read_from_disk(loading_path):
    return read_pickle(loading_path)


def main():

    import os
    dir = "/home/yuan/workspace/RLDroid/datasets/malicious"
    apks = os.listdir(dir) 
    for apk in apks:
        if apk.endswith(".apk"):
            apk_path = os.path.join(dir, apk)
            print(apk_path)
            try:
                apk2features(apk_path, 200000, "")
            except Exception as e:
                print(e)
                continue 
     

    # res = read_from_disk("./abc.feat")
    # print(res)


if __name__ == "__main__":
    main()

    # apk_path = "/home/yuan/workspace/RLDroid/datasets/malicious/01CFA6824CC7C8A611F85B0394E2273DC4E800D859975213D8697B37184ADCE7.apk"
    # a,d,dx = AnalyzeAPK(apk_path)
    # manifest_xml = a.get_android_manifest_xml()
    # import lxml.etree as etree
    # res = etree.tostring(manifest_xml, pretty_print=True)
    # print(res.decode())
