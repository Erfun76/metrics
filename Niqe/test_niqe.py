import cv2

from niqe import calculate_niqe


def main():
    img_path = 'valid_lr.png'
    img = cv2.imread(img_path)
    niqe_result = calculate_niqe(img, 0, input_order='HWC', convert_to='y')
    print("lr",niqe_result)

    img_path = 'valid_gen.png'
    img = cv2.imread(img_path)
    niqe_result = calculate_niqe(img, 0, input_order='HWC', convert_to='y')
    print("gen",niqe_result)

    img_path = 'valid_hr.png'
    img = cv2.imread(img_path)
    niqe_result = calculate_niqe(img, 0, input_order='HWC', convert_to='y')
    print("hr",niqe_result)

if __name__ == '__main__':
    main()
