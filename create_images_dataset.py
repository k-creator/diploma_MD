import numpy as np
import pydicom as pdc
# import matplotlib.pyplot as plt
import glob as glb
import cv2
import pandas as pd
import os

nodule_path = r'D:\CANCER\data_prepare'
nodules_path_list = glb.glob(nodule_path + r'\*.npy')

counter = 0
check = 0

classification_table = []


def create_mask(points, label):
    for p in points:
        # создание контура маски, 255 видно глазами, но нужно 1
        label[p[1] - 1][p[0] - 1] = 255
    label = cv2.fillPoly(label, pts=[points], color=(255, 255, 255))

    return label


for nodules in nodules_path_list:
    array_data_of_nodules_dicts = np.load(nodules, allow_pickle=True)
    # print(array_data_of_nodules_dicts[0])

    for dict_of_nodule in array_data_of_nodules_dicts:

        dicom_slices_path = glb.glob(dict_of_nodule['dicom_path'] + r'\*.dcm')
        for slice_path in dicom_slices_path:
            _slice = pdc.dcmread(slice_path)

            # сравниваем показатели из xml с полученными со слайса значениями
            if (str(_slice.SOPInstanceUID) == str(dict_of_nodule['imageSOP_UID'])) and\
                    (float(_slice.SliceLocation) == float(dict_of_nodule['imageZposition'])):
                pixel_array = _slice.pixel_array
                coordinates = dict_of_nodule['coordinates']
                coordinates = np.asarray(coordinates)

                mask = np.zeros(np.shape(pixel_array))

                # mask = create_mask(coordinates, mask)
                mask = create_mask(coordinates, mask)

                save_image_path = r'D:\CANCER\ready_dataset\images'
                save_label_path = r'D:\CANCER\ready_dataset\labels'

                if not os.path.exists(save_image_path):
                    os.makedirs(save_image_path)

                if not os.path.exists(save_label_path):
                    os.makedirs(save_label_path)

                np.save(save_image_path + r'\image_' + str(counter) + r'.npy', pixel_array)
                np.save(save_label_path + r'\label_' + str(counter) + r'.npy', mask)
                counter += 1

                classification_row = {}
                classification_row['patient_path'] = dict_of_nodule['dicom_path']
                classification_row['slice_location'] = dict_of_nodule['imageZposition']
                classification_row['SOPInstanceUID'] = dict_of_nodule['imageSOP_UID']
                classification_row['contour'] = dict_of_nodule['coordinates']
                classification_row['pixel_array'] = pixel_array
                classification_row['mask'] = mask
                classification_row['malignancy'] = dict_of_nodule['malignancy']

                classification_table.append(classification_row)

    print('nodule_' + str(check) + ' done', len(nodules_path_list))
    check += 1

df = pd.DataFrame(classification_table)
save_path = r'D:\CANCER\ready_dataset'

if not os.path.exists(save_path):
    os.makedirs(save_path)

df.to_csv(save_path + r'\classification_table.csv')

print('-------------------------------')
print('\n\n Table saved successfully')
print('-------------------------------')


