import copy

def split_source_target(data):
    data["target_img"] = data["img"][:,1:]
    data["img"] = data["img"][:,0:1]
    data["target_img_metas"] = copy.deepcopy(data["img_metas"])
    for idx, target_img_meta in enumerate(data["target_img_metas"]):
        for key, item in target_img_meta.items():
            if isinstance(item, list) and len(item) ==2:
                item = [item[1]]
                target_img_meta[key] = item
        data["target_img_metas"][idx] = target_img_meta
    return data