import os

def delete_empty_object(rootPath):
    images_path = os.path.join(rootPath, 'images')
    ann_path = os.path.join(rootPath, 'annfiles')
    print('image_path---', images_path)
    print('ann_path---', ann_path)

    ann_list = os.listdir(ann_path)
    #[P011.txt, P001.txt]

    count = 0
    for ann in ann_list:
        # xml完整路径
        # path_xml = os.path.join(annotation_path, axml)
        path_ann = ann_path + '/' + ann   
        #path_ann:"/root/autodl-tmp/datasets/DOTA/ship/trainval/annfiles/p011.txt"  ann:P011.txt

        name = ann[:-4]
        # 图片完整途径
        path_img = images_path + '/' + name + '.png'

        
        if not os.path.getsize(path_ann):
            os.remove(path_ann)
            os.remove(path_img)
            count += 1
            print('{}不含目标，已删除'.format(ann))

    print('共删除{}张'.format(count))

if __name__ == '__main__':
    rootPath = "/root/autodl-tmp/datasets/DOTA/ship/trainval"
    delete_empty_object(rootPath)