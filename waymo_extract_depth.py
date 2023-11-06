import cv2
import numpy as np
import os
import glob
from multiprocessing import Pool

def extract_depth_maps(datadirs: list[str], saving_dir: str):
    import traceback
    import logging

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    import tensorflow as tf
    from tqdm import tqdm
    from waymo_open_dataset.utils import frame_utils
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset import dataset_pb2 as open_dataset
    import matplotlib.pyplot as plt

    logging.getLogger().setLevel(logging.CRITICAL)

    for file_num, file in enumerate(datadirs):
        try:
            file_name = file.split("/")[-1].split(".")[0]

            os.makedirs(os.path.join(saving_dir, file_name, "depth_images"), exist_ok=True)
            os.makedirs(os.path.join(saving_dir, file_name, "rgb_images"), exist_ok=True)
            print("Procesing %s" % file_name)

            dataset = tf.data.TFRecordDataset(file, compression_type="")
            for f_num, data in tqdm(enumerate(dataset)):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                (
                    range_images,
                    camera_projections,
                    _,
                    range_image_top_pose,
                ) = frame_utils.parse_range_image_and_camera_projection(frame)

                all_points, all_camera_projection_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, 0)

                points = all_points[0].astype(np.float64)
                camera_projection_points = all_camera_projection_points[0].astype(np.float64)

                for im in frame.images:
                    camera_name = open_dataset.CameraName.Name.Name(im.name)

                    depth_path = os.path.join(
                        saving_dir,
                        file_name,
                        "depth_images",
                        "%03d_%s.png" % (f_num, camera_name),
                    )

                    if "FRONT" not in camera_name:
                        continue

                    cp_points_tensor = tf.constant(camera_projection_points, dtype=tf.int32)
                    points_dist = tf.norm(points, axis=-1, keepdims=True)

                    mask = tf.equal(cp_points_tensor[..., 0], frame.images[im.name - 1].name)

                    cp_points_tensor = tf.cast(tf.gather_nd(cp_points_tensor, tf.where(mask)), dtype=tf.float64)
                    points_tensor = tf.gather_nd(points_dist, tf.where(mask))

                    proj = tf.concat([cp_points_tensor[..., 1:3], points_tensor], axis=-1).numpy()

                    camera_image_size = (
                        frame.context.camera_calibrations[im.name - 1].height,
                        frame.context.camera_calibrations[im.name - 1].width,
                    )

                    depth_image = np.zeros(camera_image_size, dtype=np.float64)
                    depth_image[proj[:, 1].astype(np.uint32), proj[:, 0].astype(np.uint32)] = points_dist[mask][:, 0]

                    camera_image = tf.image.decode_jpeg(im.image).numpy()

                    # cv2.imwrite(
                    #     os.path.join(
                    #         saving_dir,
                    #         file_name,
                    #         "rgb_images",
                    #         "%03d_%s.png" % (f_num, camera_name),
                    #     ),
                    #     cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB),
                    # )

                    # cv2.imwrite(depth_path, (depth_image*255).astype(np.uint16))

                    plt.figure(figsize=(20, 12))
                    plt.imshow(camera_image)
                    plt.scatter(proj[:, 0], proj[:, 1], c=np.log(points_dist[mask]), s=5.0, edgecolors="none", cmap='hot_r')
                    plt.axis('off')
                    # plt.savefig(os.path.join('/data/sync', 'proj_by_tfrecord.png'),
                    #             bbox_inches='tight', format='png', dpi=150)
                    plt.show()

            print(f"{file_name} done!")

        except Exception as e:
            print(f"Error occured processing {file_name}")
            print(traceback.format_exc())
            raise e

save_dir='/cluster/project/infk/courses/252-0579-00L/group26/waymo/v1_extracted_test/testing'
datadirs=glob.glob("/cluster/project/infk/courses/252-0579-00L/group26/waymo/waymo_open_dataset_v_1_4_0/testing/*.tfrecord")
# for file_num, file in enumerate(datadirs):
#     print(file)

args = [ ([dir], save_dir) for i, dir in enumerate(datadirs)]

# with Pool(32) as pool:
#     pool.starmap(extract_depth_maps, args)

extract_depth_maps(datadirs, save_dir)