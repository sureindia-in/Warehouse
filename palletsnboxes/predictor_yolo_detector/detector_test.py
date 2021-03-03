import os
import shutil
import time
from pathlib import Path
import io
import logging

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image

from palletsnboxes.com_ineuron_utils.utils import encodeImageIntoBase64
from Pallet_Analysis import *



from MongoDB import mongodb
import sys
sys.path.insert(0, '../predictor_yolo_detector')

from palletsnboxes.predictor_yolo_detector.models.experimental import attempt_load
from palletsnboxes.predictor_yolo_detector.utils.datasets import LoadStreams, LoadImages
from palletsnboxes.predictor_yolo_detector.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from palletsnboxes.predictor_yolo_detector.utils.torch_utils import select_device, load_classifier, \
    time_synchronized



logging.basicConfig(filename="Prediction_Logs/Prediction_Log.txt",
                            format='%(asctime)s %(message)s',
                            filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

image_bytes = io.BytesIO()

class Detector():
    #def __init__(self, weights, conf, source, img_size, save_dir, save_txt, device, augment, agnostic_nms, conf_thres, ):
    def __init__(self, filename):
        import os

        path = os.getcwd()

        print(path)
        self.weights = "./palletsnboxes/predictor_yolo_detector/best.pt"
        self.conf = float(0.5)
        self.source = "./palletsnboxes/predictor_yolo_detector/inference/images/"
        self.img_size = int(416)
        self.save_dir = "./palletsnboxes/predictor_yolo_detector/inference/output"
        self.view_img = False
        self.save_txt = False
        self.device = 'cpu'
        self.augment = True
        self.agnostic_nms = True
        self.conf_thres = float(0.5)
        self.iou_thres = float(0.45)
        self.classes = 0
        self.save_conf = True
        self.update = True
        self.filename = filename
        self.connection = mongodb.CreateConnection()
        self.test1 = ' '
        self.test2 = ' '
        # self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        # self.log_writer = logger.App_logger()



    def detect(self, save_img=False):
        out, source, weights, view_img, save_txt, imgsz = \
            self.save_dir, self.source, self.weights, self.view_img, self.save_txt, self.img_size
        webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
        logger.info("Start of Prediction")
        logger.info("Intializing device")
        # Initialize
        set_logging()
        device = select_device(self.device)
        if os.path.exists(out):  # output dir
            shutil.rmtree(out)  # delete dir
        os.makedirs(out)  # make new dir
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        logger.info("Loading model")
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        logger.info("Loading classifier")
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        logger.info("Setting up DataLoader")
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        logger.info("Running inference on Image")
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            logger.info("Calculating inference time")
            t1 = time_synchronized()
            pred = model(img, augment=self.augment)[0]

            # Apply NMS
            logger.info("Applying NMS")
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            logger.info("Applying Classifier")
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            logger.info("Processing detections")
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                ##Added new print statements
                # print(s)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                i=0
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    ##Added new print statements
                    test1= s.split(' ')[1]
                    test2 = s.split(' ')[3]
                    # print(test1)
                    # print(test2)
                    # print(type(test1))
                    # print(type(test2))
                    # self.Pallets_label.setText(test1)
                    # self.Boxes_label.setText(test2)
                    # print(s.split(' ')[1])
                    # print(s.split(' ')[3])
                    # print(n)
                    # Write results
                    logger.info("Adding bbox and no. of detections on image")
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywhxywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, conf, *xywh) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line) + '\n') % line)

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            cv2.putText(im0, 'TotalDetections:' + str(s), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 0), 1, cv2.LINE_AA)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    logger.info("Total detections and image-size {}".format(str(s)))
                    logger.info("Saving Image to Good_Predictions output folder")
                    if save_img:
                        if dataset.mode == 'images':
                            im = Image.fromarray(im0)
                            im.save("Output_images/Good_Predictions/output.jpg")
                            im.save(image_bytes, format='JPEG')
                            self.connection.insert_good(image_bytes.getvalue(),str(s))
                # Print time (inference + NMS)
                #print('%sDone. (%.3fs)' % (s, t2 - t1))
                else:
                    i+=1

                    logger.info("Saving Image to Bad_Predictions output folder")
                    if save_img:
                        if dataset.mode == 'images':
                            im = Image.fromarray(im0)
                            im.save("Output_images/Bad_Predictions/output{}".format(i)+".jpg")
                            im.save(image_bytes, format='JPEG')
                            self.connection.insert_bad(image_bytes.getvalue())
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # Stream results
                # if view_img:
                #     cv2.imshow(p, im0)
                #     if cv2.waitKey(1) == ord('q'):  # q to quit
                #         raise StopIteration

                # Save results (image with detections)

                # if save_img:
                #     if dataset.mode == 'images':
                #         im = Image.fromarray(im0)
                #         im.save("Output_images/Good_Predictions/output.jpg")
                        #cv2.imwrite(save_path, im0)
                    # else:
                    #     print("Video Processing Needed")
                        # if vid_path != save_path:  # new video
                        #     vid_path = save_path
                        #     if isinstance(vid_writer, cv2.VideoWriter):
                        #         vid_writer.release()  # release previous video writer
                        #
                        #     fourcc = 'mp4v'  # output video codec
                        #     fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        #     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        #     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        #     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        # vid_writer.write(im0)

        #self.log_writer.log(self.file_object, 'End of Prediction')
        logger.info("End of prediction")
        if save_txt or save_img:
            print('Results saved to %s' % Path(out))

        print('Done. (%.3fs)' % (time.time() - t0))

        return test1,test2


    def detect_action(self):
        import os

        path = os.getcwd()

        print(path)
        with torch.no_grad():
            # if self.update:  # update all models (to fix SourceChangeWarning)
            #     for self.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            #         self.detect()
            #         strip_optimizer(self.weights)
            # else:
            self.detect()
        # /home/paul/PycharmProjects/factory_fire_&_smoke/your_file.jpg
        imagekeeper = []
        opencodedbase64 = encodeImageIntoBase64("output.jpg")
        imagekeeper.append({"image": opencodedbase64.decode('utf-8')})
        return imagekeeper



#firensmoke = Detector("inputimage.jpg")
#firensmoke.detect_action()
#browser.snapshot().save(str(counter) + '.png')