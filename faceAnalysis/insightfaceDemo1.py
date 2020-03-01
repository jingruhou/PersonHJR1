# coding: utf-8

import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import struct
import os
import json

"""
    @Time    : 2020/2/10/0018 10:02
    @Author  : houjingru@semptian.com
    @FileName: insightfaceDemo.py
    @Software: PyCharm
"""


def save_feature(feature_file_name, feature_data, dim_num):
    """
    512维人脸特征embedding写入指定文件
    :param feature_file_name:
    :param feature_data:
    :param dim_num:
    :return:
    """
    with open(feature_file_name, 'wb') as fd:
        fd.write(struct.pack('i', 1))
        fd.write(struct.pack('i', dim_num))
        for i in range(len(feature_data)):
            fd.write(struct.pack('f', feature_data[i]))
        fd.close()


def save_face_metadata(metadata_file_name, metadata):
    """
    保存人脸属性信息/人物元数据
    :param metadata_file_name:写入的文件名
    :param metadata:写入的数据
    :return: face_metadata_file.txt （纯文本，无结构，难以解析获取单个值）
        face = Face(bbox = bbox, landmark = landmark, det_score = det_score, embedding = embedding,
        gender = gender, age = age, normed_embedding=normed_embedding, embedding_norm = embedding_norm)
    """
    with open(metadata_file_name, 'w') as meta:
        meta.write(metadata)


def save_face_meta_data_standard(metadata_file_name, metadata):
    """
    保存人脸属性信息/人物元数据 - 标准化
    :param metadata_file_name:写入的文件名
    :param metadata:写入的数据
    :return: face_meta_data_file.txt （list，每一个元属性为一行list[index]）
        bbox = bbox,
        landmark = landmark,
        det_score = det_score,
        embedding = embedding,
        gender = gender,
        age = age,
        normed_embedding=normed_embedding,
        embedding_norm = embedding_norm
    Author:2020年2月18日 houjingru@semptian.com
    """
    with open(metadata_file_name, "a") as meta:
        meta.write(metadata)
        meta.write("\n")


def save_face_meta_data_json(metadata_file_name, dict_face_meta_data):
    """
    保存人脸属性信息/人物元数据 - JSON文件
    :param metadata_file_name: 写入的文件名
    :param dict_face_meta_data: 写入的数据-字典格式
    :return: face_meta_data_file.json

    Author:2020年2月19日 houjingru@semptian.com
    """
    with open(metadata_file_name, "w", encoding="utf-8") as file:
        file.write(dict_face_meta_data)


def url_to_image(url):
    """
    网络URL请求 字节流解码为image
    :param url:
    :return:
    """
    resp = urllib.request.urlopen(url)  # 打开url
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


"""
    1 读取图片
"""
# img = cv2.imread('C:/Users/user/PycharmProjects/PersonHJR/Resource/XJP6.jpg')
"""
    2 加载相关预训练模型
"""
model = insightface.app.FaceAnalysis()
ctx_id = -1  # CPU模式-初始化时通过ctx参数指定设备
# ctx_id = 0  # GPU模式-初始化时通过ctx参数指定设备
"""
    3 预设模型参数
"""
model.prepare(ctx_id=ctx_id, nms=0.4)
"""
    4 加载图片文件夹到模型,并且循环打印结果
"""
# 视频内人脸去重后结果 2020-02-20 houjingru@semptian.com
video_faces_result = []
video_faces_no_repeat = []
face_counts = []
face_counts_result = []

x = 0
# os.walk()返回结果：文件夹根路径dir_path, 文件夹名称dir_names, 文件名称file_names
for root, dirs, frames in os.walk("../Resource/Video_frames"):
    for d in dirs:
        print(d)  # 打印子文件夹的个数(可以忽略此步骤)
    # 循环读入每一帧
    for frame in frames:
        # 读入图像
        img_path = root + "/" + frame
        img = cv2.imread(img_path, 1)
        faces = model.get(img)
        # 循环读取一帧当中的人脸
        for idx, face in enumerate(faces):
            # 构造人俩属性字典
            dict_face_meta = {'face_id': frame.split('.')[0] + "_" + "face" + str(idx),
                              'age': str(face.age),
                              'gender': str(face.gender),
                              'bbox': str(face.bbox.astype(np.float).flatten()),
                              'landmark': str(face.landmark.astype(np.float).flatten()),
                              'det_score': str(face.det_score),
                              'embedding': str(face.embedding),
                              'embedding_shape': str(face.embedding.shape),
                              'embedding_norm': str(face.embedding_norm),
                              'normed_embedding': str(face.normed_embedding)
                              }
            face_meta_data = json.dumps(dict_face_meta, indent=8, ensure_ascii=False)
            with open("./faces_metadata/" + frame.split('.')[0] + "_" + "face" + str(idx) + ".json", "w", encoding="utf-8") as f:
                f.write(face_meta_data)
            # 如果视频人脸集合不为空，则计算相似度，更新视频人脸集合
            obj1_face_embedding = face.embedding
            # 如果该字典不为空
            if video_faces_result:
                for faceid, face_dict in enumerate(video_faces_result):
                    video_faces_result_tmp = []
                    faces_embedding = face_dict['embedding']
                    obj2_face_embedding_float = []
                    for index, value in enumerate(faces_embedding.strip('[]').split()):
                        # embedding的每个值由str转化为float
                        item_float = float(value)
                        obj2_face_embedding_float.append(item_float)
                    obj2_face_embedding = obj2_face_embedding_float
                    # 计算相似度
                    from numpy.linalg import norm
                    sim = np.dot(obj1_face_embedding, obj2_face_embedding) / (norm(obj1_face_embedding) * norm(obj2_face_embedding))
                    print("{} 与 {}的余弦相似度为: {}".format(face_dict['face_id'], dict_face_meta['face_id'], sim))
                    if sim < 0.5:
                        if dict_face_meta not in video_faces_result:
                            video_faces_result_tmp.append(dict_face_meta)
                    else:
                        face_counts.append(dict_face_meta)
                        break

                # video_faces_result.append(video_faces_result_tmp[0])
                video_faces_result = video_faces_result + video_faces_result_tmp
                # video_faces_result.append(enumerate(video_faces_result_tmp))
            # 如果视频人脸集合为空，直接将face0添加进去
            else:
                video_faces_result.append(dict_face_meta)
print("********************************************最终的人物统计结果**************************************************")
print(*video_faces_result, sep='\n')
