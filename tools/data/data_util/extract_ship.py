# coding=utf-8
import os
from shutil import copyfile
from PIL import Image, ImageDraw

def filterTxt(srcTxtPah, dstTxtPath, selected_class):
    selected_class_num = 0
    #  r:读取文件，若文件不存在则会报错
    with open(srcTxtPah, "r") as rf:
        for line in rf.readlines():
            if (selected_class in line):
                selected_class_num += 1
                #  a:写入文件,若文件不存在则会先创建再写入,但不会覆盖原文件,而是追加在文件末尾
                with open(dstTxtPath, "a") as af:
                    af.write(line)  # 自带文件关闭功能，不需要再写f.close()
    rf.close()
    return selected_class_num

def extract_ship_object():
    """
        从DOTA数据集中提取感兴趣的ship类别数据
    """
    #  DOTA数据的txt文件夹
    txtFolder = "D:\\DataSets\\DOTA\\DOTA-v1.5\\val\\labelTxt-v1.5"
    #  DOTA数据的image文件夹
    imgFolder = "D:\\DataSets\\DOTA\\DOTA-v1.5\\val\\images\\images"
    #  要复制到的image文件夹
    copy_imageFolder = "D:\\DataSets\\DOTA\\ship\\val\\images"
    #  要复制到的txt文件夹
    copy_txtFolder = "D:\\DataSets\\DOTA\\ship\\val\\labelTxt"
    #  感兴趣类别
    selected_class = "ship"

    if not os.path.exists(copy_imageFolder):
        os.makedirs(copy_imageFolder)
    if not os.path.exists(copy_txtFolder):
        os.makedirs(copy_txtFolder)

    txtNameList = os.listdir(txtFolder)
    for i in range(len(txtNameList)):
        #  判断当前文件是否为txt文件
        if (os.path.splitext(txtNameList[i])[1] == ".txt"):
            txt_path = txtFolder + "\\" + txtNameList[i]
            #  设置文件对象
            f = open(txt_path, "r")
            #  读取一行文件，包括换行符
            line = f.readline()
            while line:
                #  若该类是selected_class,则将对应图像复制粘贴,并停止循环
                if (selected_class in line):
                    #  获取txt的索引，不带扩展名的文件名
                    txt_index = os.path.splitext(txtNameList[i])[0]
                    #  获取对应图像文件的地址
                    src = imgFolder + "\\" + txt_index + ".png"
                    dst = copy_imageFolder + "\\" + txt_index + ".png"
                    #  复制图像文件至指定位置
                    copyfile(src, dst)
                    #  筛选txt文件中的selected_class信息并写至指定位置
                    selected_class_num = filterTxt(txt_path, copy_txtFolder + "\\" + txt_index + ".txt", selected_class)
                    print(txt_index, ".png have", selected_class_num, selected_class)
                    break
                #  若第一行不是selected_class，继续向下读，直到读取完文件
                else:
                    line = f.readline()
    f.close()  # 关闭文件


def draw_ship_object():

    # imgPath = r"E:/lrk/trail/datasets/DOTA-v1.5/divide_ship/train/images/P0299__800__0___37.png"
    # txtPath = r"E:/lrk/trail/datasets/DOTA-v1.5/divide_ship/train/annfiles/P0299__800__0___37.txt"

    imgPath = "E:/lrk/trail/datasets/HRSID/testsplit/all/images_bak/1524_0_0.png"
    txtPath = "E:/lrk/trail/datasets/HRSID/testsplit/all/labelTxt_bak/1524_0_0.txt"
    # savePath = os.path.basename(imgPath).split(".")[0] + "_gt" + ".jpg"
    # savePath = "E:/lrk/trail/论文/小论文/图表/图/GT/" + "P0896__800__1200___1200__gt" + ".jpg"
    # savePath = "obb.jpg"
    drawType = "hbb"

    img = Image.open(imgPath)

    draw = ImageDraw.Draw(img)
    i = 0
    with open(txtPath, "r") as f:
        for line in f.readlines():

            # 跳过前两行
            # if i==0 or i==1:
            #     print("line", line)
            #     i = i + 1
            #     continue

            #  去掉列表中每一个元素的换行符
            line = line.strip('\n')
            line = line.split(" ")
            print(line)
            if (drawType == "obb"):
                #  绘制OBB有向边界框
                polygon = []
                for i in range(8):
                    polygon.append(float(line[i]))
                polygon = tuple(polygon)
                draw.polygon(polygon, outline='red')
            elif (drawType == "hbb"):
                #  绘制HBB水平边界框
                xmin = min(float(line[0]), float(line[2]), float(line[4]), float(line[6]))
                xmax = max(float(line[0]), float(line[2]), float(line[4]), float(line[6]))
                ymin = min(float(line[1]), float(line[3]), float(line[5]), float(line[7]))
                ymax = max(float(line[1]), float(line[3]), float(line[5]), float(line[7]))
                draw.rectangle(
                    [xmin, ymin, xmax, ymax],
                    outline='red', width=3)
    savePath = 'D:/lrk/Desktop/GT.jpg'
    img.save(savePath, quality=95)
    # img.show()

if __name__ == '__main__':
    # extract_ship_object()
    draw_ship_object()