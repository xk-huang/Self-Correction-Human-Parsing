from os.path import join
from os import chdir
import os
from glob import glob
import shutil
from termcolor import colored

current_dir = os.path.abspath(os.path.dirname(__file__))

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
    tmp_dir = os.path.abspath(join(args.tmp, 'tmp_' + '_'.join(img_dir.split(os.sep)[-3:])))
    seq = img_dir.split(os.sep)[-3]
    sub = img_dir.split(os.sep)[-1]

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
    check_and_run(join(tmp_dir, 'crop.json'), cmd)

    # check the output
    out_dir = join(tmp_dir, 'mhp_fusion_parsing', 'global_tag')
    visnames = sorted(glob(join(out_dir, '*_vis.png')))
    imgnames = sorted(glob(join(img_dir, '*.jpg'))+glob(join(img_dir, '*.png')))
    if len(visnames) == len(imgnames):
        log('[log] Already has results')
        log('[log] Copy results')
        for srcname, dstname in [('schp', 'mask-schp'), ('global_tag', 'mask-schp-instance'), ('global_parsing', 'mask-schp-parsing')]:
            dir_src = join(tmp_dir, 'mhp_fusion_parsing', srcname)
            dir_dst = join(args.tmp, seq, dstname, sub)
            if os.path.exists(dir_dst):
                if False:
                    log('[log] Remove results')
                    shutil.rmtree(dir_dst)
                else:
                    continue
            os.makedirs(os.path.dirname(dir_dst), exist_ok=True)
            os.system('cp -r {} {}'.format(dir_src, dir_dst))
        return 0
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
    imgnames = sorted(glob(join(img_dir, '*.jpg'))+glob(join(img_dir, '*.png')))
    if len(visnames) == len(imgnames):
        log('[log] Finish extracting')
        log('[log] Copy results')
        for srcname, dstname in [('schp', 'mask-schp'), ('global_tag', 'mask-schp-instance'), ('global_parsing', 'mask-schp-parsing')]:
            dir_src = join(tmp_dir, 'mhp_fusion_parsing', srcname)
            dir_dst = join(args.tmp, seq, dstname, sub)
            if os.path.exists(dir_dst):
                log('[log] Skip copy results')
                continue
            os.makedirs(os.path.dirname(dir_dst), exist_ok=True)
            run_cmd('cp -r {} {}'.format(dir_src, dir_dst))
        for name in ['global_pic_parsing', 'crop_pic_parsing']:
            dirname = join(tmp_dir, name)
            if os.path.exists(dirname):
                log('[log] remove {}'.format(dirname))
                shutil.rmtree(dirname)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, nargs='+')
    parser.add_argument('--subs', type=str, nargs='+', default=[])
    parser.add_argument('--gpus', type=str, nargs='+', default=[])
    parser.add_argument('--ckpt_dir', type=str, default='/nas/share')
    parser.add_argument('--tmp', type=str, default='data')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # check checkpoints
    for name in ['detectron2_maskrcnn_cihp_finetune.pth', 'exp_schp_multi_cihp_local.pth', 'exp_schp_multi_cihp_global.pth']:
        if not os.path.exists(join(args.ckpt_dir, name)):
            assert False, '{} does not exist'.format(join(args.ckpt_dir, name))
    if len(args.gpus) > 1:
        # 使用多卡来调
        assert len(args.path) >= 1, 'Only support 1 path for multiple GPU'
        from easymocap.mytools.debug_utils import run_cmd
        subs = sorted(os.listdir(join(args.path[0], 'images')))
        nproc = len(args.gpus)
        for i in range(len(args.gpus)):
            cmd = f'export CUDA_VISIBLE_DEVICES={args.gpus[i]} && python3 extract_multi.py {" ".join(args.path)} --subs {" ".join(subs[i::nproc])} --ckpt_dir {args.ckpt_dir} --tmp {args.tmp}'
            cmd += ' &'
            run_cmd(cmd)
    else:
        for path in args.path:
            if not os.path.isdir(path) or \
                not os.path.exists(join(path, 'images')):
                myerror('{} not exist!'.format(path))
                continue
            if len(args.subs) == 0:
                subs = sorted(os.listdir(join(path, 'images')))
            else:
                subs = args.subs 
            for sub in subs:
                schp_pipeline(join(path, 'images', sub), args.ckpt_dir)
