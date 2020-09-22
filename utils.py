def getFileName(JPEGImagesPath,MainTrainPath,MainValPath):
    import os
    source_folder = JPEGImagesPath
    dest = MainTrainPath
    dest2 = MainValPath
    file_list = os.listdir(source_folder)
    train_file = open(dest, 'a')
    val_file = open(dest2, 'a')
    for file_obj in file_list:
        file_path = os.path.join(source_folder, file_obj)

        file_name, file_extend = os.path.splitext(file_obj)
        train_file.write(file_name + '\n')
    train_file.close()
    val_file.close()

JPEGImagesPath = 'D://VOCdevkit/VOC2020/JPEGImages/'
MainTrainPath = 'D://VOCdevkit/VOC2020/ImageSets/Main/train.txt'
MainValPath = 'D://VOCdevkit/VOC2020/ImageSets/Main/val.txt'

getFileName(JPEGImagesPath, MainTrainPath, MainValPath)