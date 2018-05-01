import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


def add_mask_on_one_im(input_im_dir, output_im_dir):
    im = cv2.imread(input_im_dir)

    if im is None:
        print('None')
        return

    # create a mask image of the same shape as input image, filled with 0s (black color)
    mask = np.zeros_like(im)
    rows, cols, _ = mask.shape

    # create a black filled ellipse
    cen_x = int(rows // 2)
    cen_y = int(cols // 2)
    cv2.ellipse(mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0, startAngle=0.0, endAngle=360.0,
                color=(255, 255, 255), thickness=-1)

    # bitwise
    result = np.bitwise_and(im, mask)

    # flipped mask
    new_mask = 255 * np.ones_like(im)
    cv2.ellipse(new_mask, center=(cen_y, cen_x), axes=(cen_y, cen_x), angle=0.0, startAngle=0.0, endAngle=360,
                color=(0, 0, 0), thickness=-1)

    # final result
    final_result = result + new_mask

    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    # result_rgb = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_im_dir, final_result)
    return


if __name__ == '__main__':
    # im_dir = '/home/amanda/Documents/cropped_face/e_no_mask'  # Amanda office pc
    im_dir = '/home/amanda/Documents/cropped_e/no_mask'  # Amanda home pc
    save_dir = '/home/amanda/Documents/cropped_face/e_with_mask'
    file_names = [f for f in listdir(im_dir) if isfile(join(im_dir, f))]

    file_num = len(file_names)
    for ind, f in enumerate(file_names):
        if ind % 100 == 0:
            print('{} out of {}'.format(ind+1, file_num))

        input_im_path = join(im_dir, f)
        output_im_path = join(save_dir, f)
        add_mask_on_one_im(input_im_path, output_im_path)

