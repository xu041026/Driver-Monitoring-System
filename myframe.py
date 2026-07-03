# 检测的接口函数

import cv2
import mydetect  # yolo检测
import dataPack

cap = cv2.VideoCapture(0)


def yolo_detect(datapack):
    # frame为帧输入
    frame = datapack.srcFrame
    # 定义返回变量
    ret = []
    labellist = []

    # yolo检测
    action = mydetect.predict(frame)
    for label, prob, xyxy in action:
        # 在labellist加入当前label
        labellist.append(label)
        if label == "phone":
            datapack.isPlayPhone = True
        elif label == "smoke":
            datapack.isSmoking = True
        elif label == "drink":
            datapack.isDrinking = True
        # 将标签和置信度何在一起
        text = label + str(prob)

        # 画出识别框
        left = int(xyxy[0])
        top = int(xyxy[1])
        right = int(xyxy[2])
        bottom = int(xyxy[3])
        cv2.rectangle(datapack.dstFrame, (left, top), (right, bottom), (0, 255, 0), 1)

        # 在框的左上角画出标签和置信度
        cv2.putText(datapack.dstFrame, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.imwrite("./images/" + label + ".jpg", datapack.dstFrame)

    return labellist, datapack
