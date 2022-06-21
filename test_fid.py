"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import io
import os

import torch
from options.test_options import TestOptions
from models import create_model
from evaluations.fid_score import calculate_fid_given_paths
from util.translate_images import translate_images
from util.visualizer import WDVisualizer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.num_test = opt.num_test if opt.num_test > 0 else float("inf")
    visualizer = WDVisualizer(opt)
    logger = visualizer.logger

    # traverse all epoch for the evaluation
    files_list = os.listdir(opt.checkpoints_dir + '/' + opt.name)
    epoches = []
    for file in files_list:
        if 'net_G' in file and 'latest' not in file:
            name = file.split('_')
            epoches.append(name[0])

    # eval metrics
    dataroot = opt.dataroot
    fid_values = {}
    for epoch in epoches:
        opt.epoch = epoch
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)
        generator: torch.nn.Module = None
        for _, attr in enumerate(model.__dict__):
            if attr.startswith("netG"):
                if opt.direction == 'BtoA':
                    if attr.endswith('B'):
                        generator = getattr(model, attr)
                else:
                    generator = getattr(model, attr)
        
        if generator is None:
            raise "generator is no found"
        generator.eval()

        def A2B(image):
            ans = generator(image)
            if isinstance(ans, tuple):
                ans = ans[0]
            return ans

        src_img_dir = os.path.join(dataroot, "testA" if opt.direction == 'AtoB' else "testB")
        dst_img_dir = "%s_eval_%s" % (src_img_dir, epoch)
        real_img_dir = os.path.join(dataroot, "testB" if opt.direction == 'AtoB' else "testA")
        translate_images(A2B, src_img_dir, dst_img_dir, model.device, 10)
        fid_value = calculate_fid_given_paths([real_img_dir, dst_img_dir], 50, True, 2048)
        fid_values[int(epoch)] = fid_value

    print (fid_values)
    x = []
    y = []
    for key in sorted(fid_values.keys()):
        x.append(key)
        y.append(fid_values[key])
    plt.figure()
    plt.plot(x, y)
    for a, b in zip(x, y):
        plt.text(a, b, str(round(b, 2)))
    plt.xlabel('Epoch')
    plt.ylabel('FID on test set')
    plt.title(opt.name)
    img_buf = io.BytesIO()
    plt.savefig(img_buf)

    logger.sendBlobFile(img_buf, "%s_FID.jpg" % (opt.name), "/eval_metrics", "FID-%s" % (opt.name))
