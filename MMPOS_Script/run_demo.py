import os
import glob
import csv
import json
import numpy as np
import traceback
import torch
import gc

from mmcv.image import imread
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules


def load_coco_annotations(json_path):
    """
    Loads COCO ground truth annotations from the given JSON file.
    Assumes the JSON file is in COCO format and that the 'person' category is used.

    Returns:
        gt_dict: A dictionary mapping image file names to ground truth keypoints.
                 Only the first annotation per image is used.
                 Each value is a NumPy array of shape (17, 3) representing [x, y, v].
        keypoint_names: List of keypoint names (in order).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Map image id to file name
    image_id_to_name = {img['id']: img['file_name'] for img in data['images']}

    # Find keypoint names for the 'person' category
    keypoint_names = []
    for cat in data['categories']:
        if cat['name'] == 'person':
            keypoint_names = cat['keypoints']
            break

    gt_dict = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        file_name = image_id_to_name.get(image_id)
        if file_name is None:
            continue
        if file_name in gt_dict:
            continue  # only use the first annotation per image
        kpts = ann.get('keypoints', [])
        if len(kpts) != 51:  # 17 keypoints * 3 values
            continue
        gt_dict[file_name] = np.array(kpts).reshape(17, 3)
    return gt_dict, keypoint_names


def select_leftmost_instance(instances):
    """
    Selects the left-most instance from a list of predicted instances.

    Args:
        instances (list or ndarray): List of predicted keypoints arrays of shape (17, N)

    Returns:
        (selected_index, selected_instance): The index and keypoints array of the left-most instance.
    """
    mean_xs = [np.mean(instance[:, 0]) for instance in instances]
    idx = int(np.argmin(mean_xs))
    return idx, instances[idx]


def compute_image_accuracy(pred_kpts, gt_kpts):
    """
    Computes an accuracy score for a single image based on keypoints.
    For each keypoint where the ground truth visibility is 2, the Euclidean distance between
    predicted and GT (x, y) is computed and transformed into a similarity score: 1 / (1 + distance).
    The image accuracy is the average score over all visible keypoints.

    Args:
        pred_kpts (ndarray): Predicted keypoints of shape (17, N) (N>=3 expected, [x, y, score]).
        gt_kpts (ndarray): Ground truth keypoints of shape (17, 3) ([x, y, v]).

    Returns:
        image_accuracy (float): Average similarity score over visible keypoints, or 0 if none.
    """
    scores = []
    for i in range(len(gt_kpts)):
        if gt_kpts[i, 2] == 2:  # Only use keypoints with visibility == 2
            if pred_kpts[i].shape[0] >= 2:
                pred_xy = pred_kpts[i][:2]
                gt_xy = gt_kpts[i][:2]
                dist = np.linalg.norm(pred_xy - gt_xy)
                scores.append(1 / (1 + dist))
    return np.mean(scores) if scores else 0.0


def candidate_to_str(candidate_params):
    # Create a safe string representation by concatenating keys and values.
    # Replace dot '.' with 'p' to avoid issues.
    parts = []
    for key in sorted(candidate_params):
        val = candidate_params[key]
        # Convert boolean to string without spaces, and replace '.' with 'p' for numbers
        if isinstance(val, float):
            val_str = f"{val:.2f}".replace('.', 'p')
        else:
            val_str = str(val)
        parts.append(f"{key}_{val_str}")
    return "_".join(parts)


def predict_and_save_results(val_img_dir, config_file, checkpoint_file, csv_output, heatmap_folder,
                             candidate_params, device, gt_json, restart_after=50):
    """
    Performs pose estimation on images in val_img_dir using the specified model (config and checkpoint)
    with a given candidate parameter setting.

    For each annotated image:
      - If no person is detected, writes a row with keypoints set to "None".
      - If multiple persons are detected, selects the left-most person.
      - Compares predicted keypoints with GT keypoints to compute an accuracy score.

    Outputs:
      - A CSV file with each row containing: image_name, instance index, each keypoint (x, y, score), and image accuracy.
      - Heatmap visualizations saved under heatmap_folder.

    Parameters:
        val_img_dir (str): Directory containing validation images.
        config_file (str): Path to model configuration file.
        checkpoint_file (str): Path to pretrained checkpoint file.
        csv_output (str): CSV file path to save predictions and accuracy.
        heatmap_folder (str): Folder to save heatmap images.
        candidate_params (dict): Candidate parameters (e.g., {"kpt_thr": 0.3, "flip_test": False}).
        device (str): Inference device (e.g., "cuda:0").
        gt_json (str): Path to the COCO ground truth JSON file.
        restart_after (int): Reload model after processing this many images.

    Returns:
        overall_accuracy (float): The overall average accuracy for the candidate run.
    """
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    gt_dict, gt_keypoint_names = load_coco_annotations(gt_json)
    if keypoint_names != gt_keypoint_names:
        print("Warning: Keypoint order in GT does not match expected COCO order.")

    kpt_thr = candidate_params["kpt_thr"]
    flip_test = candidate_params["flip_test"]
    cfg_options = dict(model=dict(test_cfg=dict(flip_test=flip_test)))

    register_all_modules()

    def load_model():
        return init_model(config_file, checkpoint_file, device=device, cfg_options=cfg_options)

    model = load_model()
    torch.backends.cudnn.benchmark = True

    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = sorted([img for ext in img_extensions for img in glob.glob(os.path.join(val_img_dir, ext))])
    # Filter images: only process those with ground truth annotation
    image_files = [img for img in image_files if os.path.basename(img) in gt_dict]
    print(f"Found {len(image_files)} annotated images in {val_img_dir}.")

    total_scores = []
    processed_count = 0

    with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['image_name', 'instance']
        for kp in keypoint_names:
            header.extend([f"{kp}_x", f"{kp}_y", f"{kp}_score"])
        header.append("image_accuracy")
        writer.writerow(header)

        for img_path in image_files:
            image_name = os.path.basename(img_path)
            print(f"Processing {image_name} ...")
            try:
                img = imread(img_path, channel_order='rgb')
                batch_results = inference_topdown(model, img_path)
                results = merge_data_samples(batch_results)

                # Generate heatmap visualization
                from mmpose.registry import VISUALIZERS
                visualizer = VISUALIZERS.build(model.cfg.visualizer)
                visualizer.set_dataset_meta(model.dataset_meta, skeleton_style="mmpose")
                heatmap_output_path = os.path.join(heatmap_folder, f"heatmap_{image_name}")
                visualizer.add_datasample(
                    'result',
                    img,
                    data_sample=results,
                    draw_gt=False,
                    draw_bbox=True,
                    kpt_thr=kpt_thr,
                    draw_heatmap=True,
                    show=False,
                    out_file=heatmap_output_path
                )

                # Process predictions
                if results is None or results.pred_instances is None or len(results.pred_instances.keypoints) == 0:
                    writer.writerow([image_name, 0] + ["None"] * (len(keypoint_names) * 3) + ["None"])
                else:
                    pred_instances = results.pred_instances.keypoints  # shape: (num_person, 17, N)
                    # If multiple instances are detected, select the left-most instance
                    if len(pred_instances) > 1:
                        selected_idx, selected_instance = select_leftmost_instance(pred_instances)
                    else:
                        selected_idx, selected_instance = 0, pred_instances[0]

                    # Compute accuracy score for the selected instance using ground truth keypoints
                    gt_kpts = gt_dict.get(image_name)
                    image_acc = compute_image_accuracy(selected_instance, gt_kpts)
                    total_scores.append(image_acc)

                    row = [image_name, selected_idx]
                    for kp in selected_instance:
                        if len(kp) >= 3:
                            row.extend([f"{kp[0]:.2f}", f"{kp[1]:.2f}", f"{kp[2]:.3f}"])
                        elif len(kp) == 2:
                            row.extend([f"{kp[0]:.2f}", f"{kp[1]:.2f}", "None"])
                        else:
                            row.extend(["None", "None", "None"])
                    row.append(f"{image_acc:.3f}")
                    writer.writerow(row)

                del results, batch_results
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                with open("error_log.txt", "a", encoding="utf-8") as err_file:
                    err_file.write(f"Error processing {image_name} with candidate {candidate_params}:\n")
                    err_file.write(traceback.format_exc())
                    err_file.write("\n\n")
                print(f"Error processing {image_name}, check error_log.txt")
                continue

            processed_count += 1
            if processed_count % restart_after == 0:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = load_model()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    overall_accuracy = np.mean(total_scores) if total_scores else 0.0
    print(f"Candidate {candidate_params} overall average accuracy: {overall_accuracy:.3f}")
    return overall_accuracy


def main():
    candidate_params_list = [
        {"kpt_thr": 0.25, "flip_test": True},
        {"kpt_thr": 0.27, "flip_test": True},
        {"kpt_thr": 0.3, "flip_test": True},
        {"kpt_thr": 0.32, "flip_test": True},
        {"kpt_thr": 0.35, "flip_test": True},
        {"kpt_thr": 0.45, "flip_test": True},
        {"kpt_thr": 0.5, "flip_test": True},
        {"kpt_thr": 0.6, "flip_test": True},
        {"kpt_thr": 0.7, "flip_test": True},
        {"kpt_thr": 0.25, "flip_test": False},
        {"kpt_thr": 0.27, "flip_test": False},
        {"kpt_thr": 0.3, "flip_test": False},
        {"kpt_thr": 0.32, "flip_test": False},
        {"kpt_thr": 0.35, "flip_test": False},
        {"kpt_thr": 0.45, "flip_test": False},
        {"kpt_thr": 0.5, "flip_test": False},
        {"kpt_thr": 0.6, "flip_test": False},
        {"kpt_thr": 0.7, "flip_test": False},
        # {"kpt_thr": 0.3, "flip_test": True, "input_scale": 0.9, "heatmap_sigma": 2.0},
    ]

    model_candidates = [
        {
            "model_name": "hrnet_256x192",
            "config_file": os.path.join("test_config", "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"),
            "checkpoint_file": os.path.join("test_config",
                                            "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth")
        },
        {
            "model_name": "hrnet_384x288",
            "config_file": os.path.join("test_config", "td-hm_hrnet-w48_8xb32-210e_coco-384x288.py"),
            "checkpoint_file": os.path.join("test_config",
                                            "td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth")
        }
    ]

    # device = 'cuda:0'
    device = 'cpu'
    val_img_dir = './test_image/val2017'
    gt_json = './test_image/person_keypoints_val2017_True_val/person_keypoints_val2017.json'

    # output_csv_folder = os.path.join("D:\\PythonProject\\MMPOS", "test_image_pred_results")
    # output_heatmap_folder = os.path.join("D:\\PythonProject\\MMPOS", "test_image_pred_heatmap")
    output_csv_folder = 'path for predict csv'
    output_heatmap_folder 'path for heatmap'
    os.makedirs(output_csv_folder, exist_ok=True)
    os.makedirs(output_heatmap_folder, exist_ok=True)

    best_candidate = None
    best_accuracy = -1
    best_info = ""

    # For each model candidate and each candidate parameter, run prediction and evaluation.
    for model_info in model_candidates:
        model_name = model_info["model_name"]
        print(f"\nProcessing model: {model_name}")
        for idx, candidate_params in enumerate(candidate_params_list, start=1):
            candidate_str = candidate_to_str(candidate_params)
            print(f"\nRunning candidate {idx} for model {model_name}: {candidate_str}")
            candidate_csv = os.path.join(output_csv_folder, f"{model_name}_pred_{candidate_str}.csv")
            candidate_heatmap_folder = os.path.join(output_heatmap_folder, f"{model_name}_heatmap_{candidate_str}")
            os.makedirs(candidate_heatmap_folder, exist_ok=True)
            try:
                overall_acc = predict_and_save_results(
                    val_img_dir,
                    model_info["config_file"],
                    model_info["checkpoint_file"],
                    candidate_csv,
                    candidate_heatmap_folder,
                    candidate_params,
                    device,
                    gt_json,
                    restart_after=50
                )
                print(f"Candidate {candidate_params} for model {model_name} average accuracy: {overall_acc:.3f}")
                if overall_acc > best_accuracy:
                    best_accuracy = overall_acc
                    best_candidate = candidate_params
                    best_info = f"Model: {model_name}, Candidate: {candidate_params}"
            except Exception as e:
                with open("error_log.txt", "a", encoding="utf-8") as err_file:
                    err_file.write(f"Error for model {model_name} candidate {candidate_params}:\n")
                    err_file.write(traceback.format_exc())
                    err_file.write("\n\n")
                print(f"Error encountered for model {model_name} candidate {candidate_params}, check error_log.txt")
                continue

    best_model_file = os.path.join(output_csv_folder, "best_model.txt")
    with open(best_model_file, "w", encoding="utf-8") as f:
        f.write(f"Best Candidate:\n{best_info}\nAverage Accuracy: {best_accuracy:.3f}\n")
    print("All candidate predictions, CSV files, and heatmap visualizations have been saved.")
    print(f"Best candidate: {best_info}, Average Accuracy: {best_accuracy:.3f}")
    print(f"Best model info saved to: {best_model_file}")


if __name__ == '__main__':
    main()
