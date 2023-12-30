import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model1 = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results

    # print("opt :",opt)
    # print("=" * 50)
    # print("data :",dataset)
    # print("=" * 50)

    # print("model :",model1.__dict__)
    # print("分割线".center(100,"-"))

    # opt.dataroot = "datasets/dh_512"
    # opt.name = "dh_512"
    # model2 = create_dataset(opt)
    # print("model :",model2.__dict__)


    # print("分割线".center(100,"-"))
    # opt.name = "dh_800"
    # opt.dataroot = "datasets/dh_800"
    # model3 = create_dataset(opt)
    # print("model :",model3.__dict__)


    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    print("瞎几把测试中..............")
  
    for i, data in enumerate(dataset):
        if i == 0:
            model1.setup(opt)
            model1.parallelize()


            opt.name = "dh_512"
            opt.dataroot = "datasets/dh_512"
            model2 = create_model(opt)
            model2.setup(opt)
            model2.parallelize()

            opt.name = "dh_800"
            opt.dataroot = "datasets/dh_800"
            model3 = create_model(opt)
            model3.setup(opt)
            model3.parallelize()
            if opt.eval:
                model1.eval()
                model2.eval()
                model3.eval()

        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model1.set_input(data)  # unpack data from data loader
        model2.set_input(data)  # unpack data from data loader
        model3.set_input(data)  # unpack data from data loader

        model1.test()           # run inference
        model2.test()
        model3.test()
        o1 = model1.get_current_visuals()
        o2 = model2.get_current_visuals()  # get image results
        o3 = model3.get_current_visuals()
        img_path = model1.get_image_paths()     # get image paths

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, o1, "256", width=opt.display_winsize)
        save_images(webpage, o2, "512", width=opt.display_winsize)
        save_images(webpage, o3, "800", width=opt.display_winsize)

    webpage.save()  # save the HTML

    



    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # for i, data in enumerate(dataset):
    #     if i == 0:
    #         model.setup(opt)               # regular setup: load and print networks; create schedulers
    #         model.parallelize()
    #         if opt.eval:
    #             model.eval()
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.test()           # run inference
    #     visuals = model.get_current_visuals()  # get image results
    #     img_path = model.get_image_paths()     # get image paths
    #     if i % 5 == 0:  # save images to an HTML file
    #         print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, width=opt.display_winsize)
    # webpage.save()  # save the HTML
