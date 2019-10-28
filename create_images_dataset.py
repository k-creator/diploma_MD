import numpy as np
import pydicom as pdc
import matplotlib.pyplot as plt
import glob as glb

nodule_path = r'D:\CANCER\data_prepare'
nodules_path_list = glb.glob(nodule_path + r'\*npy')

for nodules in nodules_path_list:
    array_data_of_nodules_dicts = np.load(nodules, allow_pickle=True)
    print(array_data_of_nodules_dicts[0])

    for dict_of_nodule in array_data_of_nodules_dicts:
        # print(dict_of_nodule['dicom_path'])
        dicom_slices_path = glb.glob(dict_of_nodule['dicom_path'] + r'\*.dcm')
        for slice_path in dicom_slices_path:
            _slice = pdc.dcmread(slice_path)
            # сравниваем показатели из xml с полученными со слайса значениями
            if (str(_slice.SOPInstanceUID) == str(dict_of_nodule['imageSOP_UID'])) &\
                    (float(_slice.SliceLocation) == float(dict_of_nodule['imageZposition'])):
                pixel_array = _slice.pixel_array
                coordinates = dict_of_nodule['coordinates']
                mask = np.zeros(np.shape(pixel_array))
                for point in np.asarray(coordinates):
                    mask[int(point[0]) - 1][int(point[1]) - 1] = 255
                plt.imshow(mask)
                plt.imshow(pixel_array)
                plt.show()

    break

# load_path_dicom = r'D:\CANCER\patients_path\dicom_path.npy'
# array_of_dicoms_folders = np.load(load_path_dicom)
#
# for folder in array_of_dicoms_folders:
#     dicom_slices_path = glb.glob(folder + r'\*.dcm')
#     print(dicom_slices_path[0])
#     ct_slice = pdc.dcmread(dicom_slices_path[0])
#     print(ct_slice.SOPInstanceUID)
#     break

# plt.imshow(ct_slices.pixel_array, cmap=plt.cm.bone)
# plt.show()
