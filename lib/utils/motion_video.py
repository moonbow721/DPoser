import os
import cv2
import numpy as np


def resize_or_crop(input_img, width, height):
    h, w, c = input_img.shape
    output_img = np.ones((height, width, c), dtype=np.uint8) * 255

    if w > width:
        left = (w - width) // 2
        right = left + width
        input_img = input_img[:, left:right]
    elif w < width:
        left = (width - w) // 2
        output_img[:, left:left + w] = input_img
        return output_img

    if h > height:
        bottom = h
        top = bottom - height
        input_img = input_img[top:bottom]
    elif h < height:
        bottom = height
        top = bottom - h
        output_img[top:bottom] = input_img
        return output_img

    return input_img


def crop_bottom(input_img, crop_length):
    h, w, _ = input_img.shape
    start_y = 0
    end_y = h - crop_length
    cropped_img = input_img[start_y:end_y, :]

    return cropped_img


def process_body(body_img_path, bottom=20):
    body_img = cv2.imread(body_img_path)
    body_img = crop_bottom(body_img, bottom)
    body_img = resize_or_crop(body_img, 256, 400)
    return body_img


def process_joint(joint_img_path):
    joint_img = cv2.imread(joint_img_path)
    h, w = joint_img.shape[:2]
    new_w = int(w * 0.9)
    new_h = int(h * 0.9)
    joint_img = cv2.resize(joint_img, (new_w, new_h))
    joint_img = resize_or_crop(joint_img, 256, 400)
    return joint_img


def images_to_video(input_folder, output_file, fps=20):
    """
    Convert a sequence of images in a folder to a video.

    Parameters:
    - input_folder: Path to the folder containing image sequence.
    - output_file: Name of the output video file.
    - fps: Frames per second.
    """

    images = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                     f.startswith("merge_") and f.endswith(".png")])

    if not images:
        print("No images found in the specified directory!")
        return

    frame = cv2.imread(images[0])
    h, w, layers = frame.shape
    size = (w, h)

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for image in images:
        img = cv2.imread(image)
        out.write(img)

    out.release()
    print(f"Video {output_file} created successfully!")


def seq_to_video(img_folder_path, output_merge_folder, video_path):
    img_number = len(os.listdir(img_folder_path))

    if not os.path.exists(output_merge_folder):
        os.makedirs(output_merge_folder)

    def add_title(img, title_text, position=None, blank_height=30, font_scale=0.9):
        h, w = img.shape[:2]
        blank = 255 * np.ones((blank_height, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        text_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]

        if position is None:
            x = (w - text_size[0]) // 2
            y = (blank_height + text_size[1]) // 2 - 5
        else:
            x, y = position

        cv2.putText(blank, title_text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return np.vstack((img, blank))

    for i in range(img_number // 3):
        frame_name = os.path.join(img_folder_path, "frame_{:04d}.png".format(i))
        out_name = os.path.join(img_folder_path, "out_{:04d}.png".format(i))
        gt_name = os.path.join(img_folder_path, "gt_{:04d}.png".format(i))

        joint_img = process_joint(frame_name)
        out_img = process_body(out_name)
        gt_img = process_body(gt_name)

        # add titles
        joint_img = add_title(joint_img, "Noisy Joints")
        out_img = add_title(out_img, "DPoser(Ours)")
        gt_img = add_title(gt_img, "GT")

        # concat
        merged_img = np.hstack((joint_img, out_img, gt_img))
        cv2.imwrite(os.path.join(output_merge_folder, "merge_{:04d}.png".format(i)), merged_img)

    # seq -> video
    images_to_video(output_merge_folder, video_path)
