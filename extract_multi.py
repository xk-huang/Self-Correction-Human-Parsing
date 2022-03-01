from os.path import join
from os import chdir
import os
from glob import glob
import shutil
from termcolor import colored

current_dir = os.path.dirname(__file__)

move_root = lambda: chdir(current_dir)
move_mhp = lambda: chdir(join(current_dir, 'mhp_extension'))
move_tool = lambda: chdir(join(current_dir, 'mhp_extension', 'detectron2', 'tools'))

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))

def log(text):
    myprint(text, 'info')

def mywarn(text):
    myprint(text, 'warn')

def myerror(text):
    myprint(text, 'error')

def run_cmd(cmd, verbo=True):
    if verbo: myprint('[run] ' + cmd, 'run')
    os.system(cmd)
    return []

def check_and_run(outname, cmd):
    if (os.path.exists(outname) and os.path.isfile(outname)) or (os.path.isdir(outname) and len(os.listdir(outname)) >= 3):
        mywarn('Skip {}'.format(cmd))
    else:
        run_cmd(cmd)

def schp_pipeline(img_dir, ckpt_dir):
    tmp_dir = os.path.abspath(join('data', 'tmp_' + '_'.join(img_dir.split(os.sep)[-3:])))
    move_mhp()
    annotations = join(tmp_dir, 'Demo.json')
    cmd = f"python3 ./coco_style_annotation_creator/test_human2coco_format.py --dataset 'Demo' --json_save_dir {tmp_dir} --test_img_dir {img_dir}"
    check_and_run(annotations, cmd)

    move_tool()
    # 通过设置环境变量来控制
    os.environ['annotations'] = annotations
    os.environ['img_dir'] = img_dir
    cmd = f"python3 ./finetune_net.py --num-gpus 1 --config-file ../configs/Misc/demo.yaml --eval-only MODEL.WEIGHTS {join(ckpt_dir, 'detectron2_maskrcnn_cihp_finetune.pth')} TEST.AUG.ENABLED False DATALOADER.NUM_WORKERS 0 OUTPUT_DIR {join(tmp_dir, 'detectron2_prediction')}"
    check_and_run(join(tmp_dir, 'detectron2_prediction'), cmd)

    move_mhp()
    cmd = f"python3 make_crop_and_mask_w_mask_nms.py --img_dir {img_dir} --save_dir {tmp_dir} --img_list {annotations} --det_res {tmp_dir}/detectron2_prediction/inference/instances_predictions.pth"
    check_and_run(join(tmp_dir, 'crop_pic'), cmd)

    move_root()
    os.environ['PYTHONPATH'] = '{}:{}'.format(current_dir, os.environ.get('PYTHONPATH', ''))
    cmd = f"python3 mhp_extension/global_local_parsing/global_local_evaluate.py --data-dir {tmp_dir} --split-name crop_pic --model-restore {ckpt_dir}/exp_schp_multi_cihp_local.pth --log-dir {tmp_dir} --save-results"
    check_and_run(join(tmp_dir, 'crop_pic_parsing'), cmd)

    if not os.path.exists(join(tmp_dir, 'global_pic')):
        os.system('ln -s {} {}'.format(img_dir, join(tmp_dir, 'global_pic')))
    cmd = f"python mhp_extension/global_local_parsing/global_local_evaluate.py --data-dir {tmp_dir} --split-name global_pic --model-restore {ckpt_dir}/exp_schp_multi_cihp_global.pth --log-dir {tmp_dir} --save-results"
    check_and_run(join(tmp_dir, 'global_pic_parsing'), cmd)

    cmd = f"python mhp_extension/logits_fusion.py --test_json_path {tmp_dir}/crop.json --global_output_dir {tmp_dir}/global_pic_parsing --gt_output_dir {tmp_dir}/crop_pic_parsing --mask_output_dir {tmp_dir}/crop_mask --save_dir {tmp_dir}/mhp_fusion_parsing"
    run_cmd(cmd)
    # check the output
    out_dir = join(tmp_dir, 'mhp_fusion_parsing', 'global_tag')
    visnames = sorted(glob(join(out_dir, '*_vis.png')))
    imgnames = sorted(glob(join(img_dir, '*.jpg')))
    if len(visnames) == len(imgnames):
        log('[log] Finish extracting')
        log('[log] Copy results')
        resdir = join('data', img_dir.split(os.sep)[-3], img_dir.split(os.sep)[-1])
        os.makedirs(os.path.dirname(resdir), exist_ok=True)
        shutil.copytree(join(tmp_dir, 'mhp_fusion_parsing', 'schp'), resdir)
        for name in ['global_pic_parsing', 'crop_pic_parsing']:
            dirname = join(tmp_dir, name)
            if os.path.exists(dirname):
                log('[log] remove {}'.format(dirname))
                shutil.rmtree(dirname)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--ckpt_dir', type=str, default='/nas/share')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    for sub in sorted(os.listdir(join(args.path, 'images'))):
        schp_pipeline(join(args.path, 'images', sub), args.ckpt_dir)