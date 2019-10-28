import os
import glob
import numpy as np
from xml.dom import minidom

path_dir = r'D:\CANCER\LIDC\LIDC-IDRI'
patient_dir = os.listdir(path_dir)

dicom_paths = []

folder_counter = 0
counter = 0

for patient_folder in patient_dir:
    patient_path = path_dir + r'\\' + patient_folder
    series_dir = os.listdir(patient_path)

    for series_folder in series_dir:
        series_path = patient_path + r'\\' + series_folder
        another_series_dir = os.listdir(series_path)

        for another_series_folder in another_series_dir:
            dicom_path = series_path + r'\\' + another_series_folder
            if len(os.listdir(dicom_path)) > 15:

                path_xml = glob.glob(dicom_path + r'\\*.xml')[0]

                my_doc = minidom.parse(path_xml)

                unblindedReadNodules = my_doc.getElementsByTagName('unblindedReadNodule')

                nodules = []

                check = 0
                for unblindedReadNodule in unblindedReadNodules:
                    characteristics = unblindedReadNodule.getElementsByTagName('characteristics')
                    nodule = {}

                    # вытаскиваем только тех, которых можно будет классифицировать
                    if len(characteristics) > 0:

                        malignancy = characteristics[0].getElementsByTagName('malignancy')
                        if len(malignancy) == 0:
                            continue
                        malignancy_score = malignancy[0].firstChild.data
                        nodule['malignancy'] = malignancy_score

                        nodule['dicom_path'] = dicom_path
                        if check == 0:
                            dicom_paths.append(dicom_path)

                        # собираем информацию для каждого из узелков (nodules)
                        rois = unblindedReadNodule.getElementsByTagName('roi')
                        for roi in rois:

                            # флаг, нужно ли брать данный узел
                            inclusion = roi.getElementsByTagName('inclusion')
                            inclusion = inclusion[0].firstChild.data

                            if inclusion:
                                # номер слайса
                                position = roi.getElementsByTagName('imageZposition')
                                position = position[0].firstChild.data
                                nodule['imageZposition'] = position

                                # уникальный номер слайса
                                sop_uid = roi.getElementsByTagName('imageSOP_UID')
                                sop_uid = sop_uid[0].firstChild.data
                                nodule['imageSOP_UID'] = sop_uid

                                # вынимаем координаты контура для каждого узелка
                                edgeMaps = roi.getElementsByTagName('edgeMap')
                                points = []
                                for edgeMap in edgeMaps:
                                    xCoord = edgeMap.getElementsByTagName('xCoord')
                                    xCoord = xCoord[0].firstChild.data

                                    yCoord = edgeMap.getElementsByTagName('yCoord')
                                    yCoord = yCoord[0].firstChild.data

                                    points.append([xCoord, yCoord])

                                nodule['coordinates'] = points

                        nodules.append(nodule)
                    check += 1

                nodules = np.asarray(nodules)
                # print(nodules)
                np.save(r'D:\\CANCER\\data_prepare\\unblindedReadNodule_' + str(counter) + '.npy', nodules)
                counter += 1

    print('folder_', folder_counter, ' done')
    folder_counter += 1

dicom_paths = np.asarray(dicom_paths)
np.save(r'D:\\CANCER\\patients_path\\dicom_path.npy', dicom_paths)
