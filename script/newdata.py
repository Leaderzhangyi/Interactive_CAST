from PIL import Image
import os 
resolutions = [(256, 256), (512, 512), (800, 800)]


data_root = "../datasets/dh"
need_dir = ["../datasets/dh_256/trainA","../datasets/dh_256/trainB","../datasets/dh_512/trainA","../datasets/dh_512/trainB","../datasets/dh_800/trainA","../datasets/dh_800/trainB"]
for dir in need_dir:
    if os.path.exists(dir):
        print(f"{dir} is exists!")
        continue
    os.mkdir(dir)
    print(f"{dir} is creating...")




def resize_image(input_image, output_image, target_resolution):
    img = Image.open(input_image)
    resized_img = img.resize(target_resolution)
    resized_img.save(output_image)


for dir in os.listdir(data_root):
    if dir.startswith("test"):
        print("test dir, continue...")
        continue
    s_path = os.path.join(data_root,dir)
    for image in os.listdir(s_path):

        for resolution in resolutions:
            width, height = resolution
            print(f"{dir}/{image}_{width}x{height}.jpg")
            output_image = "../datasets/dh_" + str(resolution[0])+"/"+dir+"/"+image
            print(f"output_image = {output_image}")
            resize_image(os.path.join(s_path,image), output_image, resolution)










# input_image = r"C:\Users\ZinkCas\Desktop\121.jpg"

