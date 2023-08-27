import os
import cv2

def find_images(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return image_files

def main():
    folder_path = '../../cut_1024/val_img/'
    image_files = find_images(folder_path)

    if not image_files:
        print("폴더 내에 이미지 파일이 없습니다.")
        return

    current_image_index = 0

    while True:
        image_path = image_files[current_image_index]
        img = cv2.imread(image_path)
        cv2.imshow("Image Viewer 1024", img)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC key
            break
        elif key == 37: # Left
            if current_image_index > 0:
                current_image_index -= 1
        elif key == 39:  # Right
            if current_image_index < (len(image_files)-1):
                current_image_index += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()