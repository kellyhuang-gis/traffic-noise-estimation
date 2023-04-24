import json
import torch
from PIL import Image
from torchvision import transforms
from cnn_model import resnet34
import os
import xlwt
import xlrd
from xlutils.copy import copy
import joblib
import pandas as pd


def excel_xls_append(path, value, j):
    index = len(value)
    print(index)
    workbook = xlrd.open_workbook(path)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])
    rows_old = 1
    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(0)
    for i in range(0, index):
        print(i)
        print(value[i])
        new_worksheet.write(i+rows_old, j,  str(value[i]))
    new_workbook.save(path)


def write_excel_xls(path, sheet_name, value):
    index = len(value)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])
    workbook.save(path)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
         transforms.Resize(360),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

## load street imagery that needs noise prediction
    imgs_root = "./CDBSV"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."

    # Read .jpg image in the folder
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create resnet
    model = resnet34(num_classes=5).to(device)

    # load resnet weight
    weights_path = "./cnn_best.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

## traffic noise level prediction
    model.eval()
    batch_size = 1
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)

            probs, classes = torch.max(predict, dim=1)
            log = open("pred_level.txt", mode="a", encoding="utf-8")
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],

                                                                 class_indict[str(cla.numpy())],
                                                          pro.numpy()),file = log)
                print(predict.float(),file = log)
            log.close()

    file = open('pred_level.txt', 'r', encoding='utf-8')
    street_name = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    for idx, line in enumerate(file.readlines()):
        if idx % 2 == 0:
            name = line.strip('image:')
            street_name.append(name.split()[0])

        if idx % 2 != 0:
            a = line.strip("tensor([[")
            b = a.strip()
            level_prob = b.strip("]])")
            x1.append(level_prob.split(',')[0])
            x2.append(level_prob.split(',')[1])
            x3.append(level_prob.split(',')[2])
            x4.append(level_prob.split(',')[3])
            x5.append(level_prob.split(',')[4])


    book_name_xls = 'pred_level.xls'
    sheet_name_xls = 'sheet1'
    value_title = [["x1", "x2", "x3", "x4", "x5", ], ]
    write_excel_xls(book_name_xls, sheet_name_xls, value_title)

    excel_xls_append(book_name_xls, x1, 0)
    excel_xls_append(book_name_xls, x2, 1)
    excel_xls_append(book_name_xls, x3, 2)
    excel_xls_append(book_name_xls, x4, 3)
    excel_xls_append(book_name_xls, x5, 4)

## traffic noise value estimation
    RF = joblib.load("NoiseModel.m")
    data_test = pd.read_excel("pred_level.xls")
    y_test_pred = RF.predict(data_test)
    data_test['y_test_pred'] = y_test_pred
    data_test['street_idx'] = street_name
    data_test.to_excel('result.xlsx')

if __name__ == '__main__':
    main()