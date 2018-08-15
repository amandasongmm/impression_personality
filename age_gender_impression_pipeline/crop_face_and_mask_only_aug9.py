from __future__ import division
import dlib
import numpy as np
import cv2
import time
import os


def crop_faces(raw_data_dir_postfix='0/',
               raw_data_root_dir='/home/amanda/Documents/VC and Entrepreneur Faces Project/New_data/crunchbase/',
               crop_with_mask_root_dir='/home/amanda/Documents/cropped_new_crunchbase/'
               ):
    '''
        crop_faces() requires two arguments: raw_data_dir and crop_with_mask_dir

        raw_data_dir is the root dir that you want to run the analysis on. e.g.
        '/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/' will search for all the *.jpg
        files under the subfolder of vc recursively. namely, all the vc images.

        crop_with_mask_dir is where you want to save the cropped faces with an oval mask.

        Example: raw_data_dir='/home/amanda/Documents/VC and Entrepreneur Faces Project/Extracted Faces/vc/',
        crop_with_mask_dir='/home/amanda/Documents/cropped_faces/vc_with_mask/'
    '''

    def fetch_file(path_to_folder, flag, key_word):
        '''
        fetch_files() requires three arguments: pathToFolder, flag and keyWord

        flag must be 'start_with' or 'end_with'
        keyWord is a string to search the file's name

        Be careful, the keyWord is case sensitive and must be exact

        Example: fetch_file('/Documents/Photos/','end_with','.jpg')

        returns: _pathToFiles and _fileNames
        '''

        _path_to_files = []
        _file_names = []

        for dir_path, dir_names, file_names in os.walk(path_to_folder):
            if flag == 'end_with':
                raw_file = [item for item in file_names if item.endswith(key_word)]
                selected_file = [item for item in raw_file if not item.startswith('._')]
                selected_file = list(set(selected_file))
                _file_names.extend(selected_file)

                selected_path = [os.path.join(dir_path, item) for item in selected_file]
                _path_to_files.extend(selected_path)

            elif flag == 'start_with':
                raw_file = [item for item in file_names if item.startswith(key_word)]
                selected_file = [item[2:] if item.startswith('._') else item for item in raw_file]
                selected_file = list(set(selected_file))
                _file_names.extend(selected_file)

                selected_path = [os.path.join(dir_path, item) for item in selected_file]
                _path_to_files.extend(selected_path)

            else:
                print fetch_file.__doc__
                break

            # try to remove empty entries if none of the required files are in the directory
            try:
                _path_to_files.remove('')
                _file_names.remove('')
            except ValueError:
                pass

            # warn if nothing was found in the given path
            if selected_file == []:
                print('No files with given parameters were found in {}\n'.format(dir_path))

        print('{} files are found in searched folders.'.format(len(_file_names)))
        return _path_to_files, _file_names

    start_t = time.time()
    raw_data_dir = raw_data_root_dir + raw_data_dir_postfix

    crop_with_mask_dir = crop_with_mask_root_dir + raw_data_dir_postfix
    if not os.path.exists(crop_with_mask_dir):
        os.makedirs(crop_with_mask_dir)

    no_face_dir = crop_with_mask_root_dir + 'No_face/'
    if not os.path.exists(no_face_dir):
        os.makedirs(no_face_dir)

    multi_face_dir = crop_with_mask_root_dir + 'Multi_face/'
    if not os.path.exists(multi_face_dir):
        os.makedirs(multi_face_dir)

    cannot_read_dir = crop_with_mask_root_dir + 'cannot_read/'
    if not os.path.exists(cannot_read_dir):
        os.makedirs(cannot_read_dir)

    end_with_key_word = 'g'

    path_to_files, file_names = fetch_file(raw_data_dir, 'end_with', end_with_key_word)  # all files in subdirectories
    total_file_len = len(file_names)
    print('Total number of files in this folder is {}'.format(total_file_len))

    detector = dlib.get_frontal_face_detector()  # detect faces using dlib detector.

    # Face detection related parameters.
    x_left_margin = 0.15
    x_right_margin = 0.15
    y_top_margin = 0.35
    y_bottom_margin = 0.12

    count = 0
    cannot_read_face_count = 0
    no_face_count = 0
    multi_face_count = 0
    for i, (file_name_only, file_path) in enumerate(zip(file_names, path_to_files)):
        img = cv2.imread(file_path)
        if img is None:
            cannot_read_face_count += 1
            print('#{} photos cannot be read'.format(cannot_read_face_count))
            cv2.imwrite(cannot_read_dir + file_name_only, img)
            continue

        img_h, img_w, _ = np.shape(img)

        detected = detector(img, 1)

        if len(detected) == 0:
            no_face_count += 1
            print('#{} photos have no face'.format(no_face_count))
            cv2.imwrite(no_face_dir + file_name_only, img)

        elif len(detected) > 1:
            multi_face_count += 1
            print('#{} photos have multiple faces'.format(multi_face_count))
            cv2.imwrite(multi_face_dir + file_name_only, img)

        else:
            count += 1
            if count % 100:
                print('{} single face detected out of {}, elapsed time = {}'.format(
                    count, total_file_len, time.time()-start_t))

            for d in detected:
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - x_left_margin * w), 0)
                yw1 = max(int(y1 - y_top_margin * h), 0)
                xw2 = min(int(x2 + x_right_margin * w), img_w - 1)
                yw2 = min(int(y2 + y_bottom_margin * h), img_h - 1)

                roi_color = img[yw1: yw2 + 1, xw1: xw2 + 1, :]

                # add mask and save masked image.
                mask = np.zeros_like(roi_color)
                rows, cols, _ = mask.shape

                # create a black filled ellipse
                cen_x = int(rows // 2)
                cen_y = int(cols // 2)
                cv2.ellipse(mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0, startAngle=0.0,
                            endAngle=360.0, color=(255, 255, 255), thickness=-1)
                # bitwise
                result = np.bitwise_and(roi_color, mask)

                # flipped mask
                new_mask = 255 * np.ones_like(roi_color)
                cv2.ellipse(new_mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0,
                            startAngle=0.0, endAngle=360, color=(0, 0, 0), thickness=-1)

                # final result
                with_mask = result + new_mask

                # save the cropped images in a folder.
                cv2.imwrite(crop_with_mask_dir + file_name_only, with_mask)


