from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

labels = [(val, key) for key, val in {
    0:"Aerosol", 
    1:"Alcohol", 
    2:"Awl",
    3:"Axe", 
    4:"Bat",
    5:"Battery", 
    6:"Bullet", 
    7:"Firecracker", 
    8:"Gun", 
    9:"GunParts",
    10:"Hammer", 
    11:"HandCuffs", 
    12:"HDD", 
    13:"Knife",
    14:"Laptop",
    15:"Lighter",
    16:"Liquid", 
    17:"Match",
    18:"MetalPipe", 
    19:"NailClippers", 
    20:"PrtableGas", 
    21:"Saw", 
    22:"Scissors", 
    23:"Screwdriver", 
    24:"SmartPhone", 
    25:"SolidFuel", 
    26:"Spanner", 
    27:"SSD",
    28:"SupplymentaryBattery", 
    29:"TabletPC",
    30:"Thinner", 
    31:"USB",
    32:"ZippoOil", 
    33:"Plier", 
    34:"Chisel", 
    35:"Electronic cigarettes", 
    36:"Electronic cigarettes(Liquid)", 
    37:"Throwing Knife", 
}.items()]

def reshape_transform(tensor, height=7, width=7):
    #cam(..., reshape_transform=reshape_transform)

    result = tensor.reshape(tensor.size(0),height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_fig(inputs, model,target_layers, target_name:str = None):
    '''
    model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]
    input_tensor = # Create an input tensor image for your model..
    cannot use in transformer(yet)
    '''
    if target_name:
        target_cat = labels[target_name]

    cam = ScoreCAM(model, target_layers, use_cuda=True, target_category=target_cat)
    # plt 기본 스타일 설정
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['font.size'] = 12

    fig, axes = plt.subplots(nrows=1, ncols=inputs.shape[0])
    for ax, input in zip(axes, inputs):
        img = show_cam_on_image(inputs, cam, use_rgb=True)
        ax.imshow(img)

    return fig / 