import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import yolov5
import json
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from trackers.ocsort_tracker.murine_ocsort import OCSort
from trackers.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return sorted(image_names)


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

class PredictorV5(object):
    def __init__(self, model_path, conf, iou):
        self.model_path = model_path
        self.model = yolov5.load(self.model_path)
        self.conf = conf
        self.iou = iou

    def inference(self, img, timer, frame_id):
        img_info = {"id": frame_id}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ratio"] = .999999
        #img, ratio = preprocess_v5(img, self.test_size, self.rgb_means, self.std)
        self.model.conf = self.conf
        results = self.model(img)
        timer.tic()
        predictions = results.pred[0]
        if len(predictions)==0:
            return [None], img_info
        else:
            boxes = predictions[:, :4].tolist()
            boxes_int = torch.tensor([[int(v) for v in box] for box in boxes])
            scores = torch.tensor(predictions[:,4].tolist())
            outputs = [torch.cat((boxes_int, torch.unsqueeze(scores, dim=1)), 1)]
            return outputs, img_info, img

class Predictor_fromJson(object):
    def __init__(self, results_path):
        self.results_path = results_path
    
    def inference(self, img_path, timer):
        self.frame_id = img_path.split("/")[-1]
        img_info = {"id": self.frame_id }
        if isinstance(img_path, str):
            img_info["file_name"] = img_path
            img = cv2.imread(img_path)
        else:
            img_info["file_name"] = None
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ratio"] = .999999
        ### read_json
        # image_id = img_path.replace(".jpg", ".json")
        with open(f'{self.results_path}/outs_{self.frame_id.replace(".jpg", ".json")}') as json_file:
            preds = json.load(json_file)
        return [torch.tensor(preds['bboxes'])[:,:5]], img_info, img

class PredictorFromDB(object):
    def __init__(self, model_path, conf, iou):
        self.model_path = model_path
        self.model = yolov5.load(self.model_path)
        self.conf = conf
        self.iou = iou

    def inference(self, img, timer, frame_id):
        img_info = {"id": frame_id}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ratio"] = .999999
        #img, ratio = preprocess_v5(img, self.test_size, self.rgb_means, self.std)
        self.model.conf = self.conf
        results = self.model(img)
        timer.tic()
        predictions = results.pred[0]
        if len(predictions)==0:
            return [None], img_info
        else:
            boxes = predictions[:, :4].tolist()
            boxes_int = torch.tensor([[int(v) for v in box] for box in boxes])
            scores = torch.tensor(predictions[:,4].tolist())
            outputs = [torch.cat((boxes_int, torch.unsqueeze(scores, dim=1)), 1)]
            return outputs, img_info, img

def image_demo(predictor, vis_folder, current_time, args):
    width = 1014
    height = 760
    vid_writer = cv2.VideoWriter(
        args.out_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (int(width), int(height))
    )
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info, img = predictor.inference(img_path, timer, frame_id)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, img_info, img)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB))
            vid_writer.write(cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB))

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo_type == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo_type == "video":
        save_path = args.out_path
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = OCSort(det_thresh=args.track_thresh,
     iou_threshold=args.iou_thresh,
      use_byte=args.use_byte)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info, img = predictor.inference(frame, timer, frame_id)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, img_info, img)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]  
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                cv2.imwrite(f"/home/ubuntu/trackers/OC_SORT/temp_test/{str(frame_id).zfill(5)}_frame.jpg", online_im)
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    #model = exp.get_model().to(args.device)
    #logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #model.eval()

    # if not args.trt:
    #     if args.ckpt is None:
    #         ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    #     else:
    #         ckpt_file = args.ckpt
    #     logger.info("loading checkpoint")
    #     ckpt = torch.load(ckpt_file, map_location="cpu")
    #     # load the model state dict
    #     model.load_state_dict(ckpt["model"])
    #     logger.info("loaded checkpoint done.")

    # if args.fuse:
    #     logger.info("\tFusing model...")
    #     model = fuse_model(model)

    # if args.fp16:
    #     model = model.half()  # to FP16

    # if args.trt:
    #     assert not args.fuse, "TensorRT model is not support model fusing!"
    #     trt_file = osp.join(output_dir, "model_trt.pth")
    #     assert osp.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    #     logger.info("Using TensorRT to inference")
    # else:
    #     trt_file = None
    #     decoder = None

    predictor = PredictorV5(model_path = "/home/ubuntu/yolov5/yolox-mousefinder/defaults_all_mousefinder_001/weights/best.pt",
                            conf =exp.test_conf,
                            iou = 0.4)
    print("args.path", args.path)
    predictor_imgs = Predictor_fromJson(results_path = args.path)
    current_time = time.localtime()

    if args.demo_type == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo_type == "video" or args.demo_type == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
