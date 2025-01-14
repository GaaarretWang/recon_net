import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from datetime import datetime,timedelta
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def millis(start_time):
   dt = datetime.now() - start_time
   ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
   return ms
def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            #if is_image_file(fname):
            path = os.path.join(root, fname)
            images.append(path)
    return images[:min(max_dataset_size, len(images))]

def initIndex(height,width):
    indeX = torch.zeros(1,height,width,2)

    h = torch.arange(height).to(device='cuda')
    h.unsqueeze_(0)
    h.unsqueeze_(2)
    h.unsqueeze_(3)
    h = h.repeat(1, 1, width, 1)

    w = torch.arange(width).to(device='cuda')
    w.unsqueeze_(0)
    w.unsqueeze_(1)
    w.unsqueeze_(3)
    w = w.repeat(1, height, 1, 1)
    #torch.cat((h,w),dim=3,out = indeX)
    return h,w

#返回current和warpPrev的插值结果lerpResult
def reProject(current,mv,h,w, height,width):
    #mv进来的时候是（1,2，1080，1920）
    #mv.squeeze_(0)
    #print(mv.size())
    #warpPrev = torch.zeros(1,3,1080,1920).to(device='cuda')
    #lerpResult = torch.zeros(1,3,1080,1920).to(device='cuda')
    prevPixelPos = torch.zeros(1, height, width, 2).to(device='cuda')

    #mv.squeeze_(0)
    #mv = 50*(mv-0.5)#先将mv恢复到正确的数值
    #mv = (mv-0.482352942)*115  # 先将mv恢复到正确的数值
    #mv = (mv - 0.482352942) * 96.5 #95.5不够大
    #print(mv)
    #mv = (mv-0.1981)*145.0
    #print(mv)
    #print(mv.size())
    #print(mv.size())
    mv = mv.permute(0,2,3,1)#将mv(1,2,1080,1920)变成（1，1080，1920，2）

    x = mv[:, :, :, :1].to(device='cuda')#mv最后一维的第一个值是x坐标
    y = mv[:, :, :, 1:2].to(device='cuda')#mv最后一维的第二个值是y坐标
    #torch.cat((y, x), dim=3, out=prevPixelPos)

    #print(x.shape)
    #print(h.shape)
    #print(y.shape)
    #print(w.shape)
    # x =  2*(int(h+1+x)/1016)-1
    #torch.add(x, 1, alpha=1, out=x)
    torch.add(w,x,  alpha=1, out=x)
    x=x.floor()
    #x=torch.where(x <= 0, 0, x)
    #x = torch.where(x >= width, width-1, x)
    torch.div(x,width-1,out=x)
    torch.mul(x,2,out=x)
    torch.sub(x,1,out=x)

    # y =  2*(int(w+1+y)/1919)-1
    #torch.add(y, 1, alpha=1, out=y)
    torch.add( h,y, alpha=1, out=y)
    y = y.floor()
    #y=torch.where(y <= 0, 0, y)
    #y = torch.where(y >= height, height - 1, y)
    torch.div(y, height - 1, out=y)
    torch.mul(y, 2, out=y)
    torch.sub(y, 1, out=y)

    torch.cat((x,y), dim=3, out=prevPixelPos)
    #return prevPixelPos
    #for h in range(1080):
    #    for w in range(1920):
    #        prevPixelPos[0, h, w, 1] = 2 * (int(h + 1 + mv[0, h, w]) / 1016) - 1
    #        prevPixelPos[0, h, w, 0] = 2 * (int(w + 1 + mv[1, h, w]) / 1919) - 1

    warpPrev = F.grid_sample(current, grid=prevPixelPos, mode='bilinear', align_corners=True)
    #warpPrevlerp =warpPrev
    #torch.mul(warpPrev, 0.8, out=warpPrevlerp)
    #currentlerp = current
    #torch.mul(current, 0.2, out=currentlerp)
    #lerpResult = torch.lerp(warpPrev, current, 0.2)
    #warpPrev = F.pad(warpPrev, (1, 0, 1, 0), mode='reflect').to(device='cuda')
    #warpPrev = warpPrev[:,:,:1080,:1920]
    #print(warpPrev.size())
    return warpPrev


def temporalReprojection(preResult: torch.Tensor, motion: torch.Tensor, padding_mode: str = "border", device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """ warped(x, y) = preResult(x + mv.x, y + mv.y)
    ### Parameters
    preResult (torch.Tensor): tensor need to be warped (N, C, H, W)
    motion (torch.Tensor): motion vector (N, H, W, 2)
    ### Returns
    warped (torch.Tensor): warped tensor (N, C, H, W)
    """
    w, h = (1920 + 2, 1080 + 2)
    mx, my = torch.split(motion, 1, dim=3)
    mx = F.pad(mx, [0, 0, 1, 1, 1, 1], mode="constant", value=0)
    my = F.pad(my, [0, 0, 1, 1, 1, 1], mode="constant", value=0)
    x = torch.linspace(0, w-1, steps=w)
    y = torch.linspace(0, h-1, steps=h)
    grid_y, grid_x = torch.meshgrid(y, x)

    grid_x = grid_x.unsqueeze(2)
    # temp_x = grid_x + mx
    # temp_x = torch.where(temp_x < 0, grid_x, temp_x)
    # grid_x = torch.where(temp_x >= w, grid_x, temp_x)
    grid_x = grid_x + mx

    grid_y = grid_y.unsqueeze(2)
    # temp_y = grid_y + my
    # temp_y = torch.where(temp_y < 0, grid_y, temp_y)
    # grid_y = torch.where(temp_y >= h, grid_y, temp_y)
    grid_y = grid_y + my

    grid = torch.cat((grid_x, grid_y), dim=3)
    xr = torch.ones(1, h, w, 1) * (2. / (w-1))
    yr = torch.ones(1, h, w, 1) * (2. / (h-1))
    xyr = torch.cat((xr, yr), dim=3)
    grid = (grid * xyr - 1).to(device=device)

    preResult = F.pad(preResult, [1, 1, 1, 1], mode="constant", value=1)

    warped = F.grid_sample(preResult, grid, align_corners=True,
                           padding_mode=padding_mode)
    warped = warped[..., 1:-1, 1:-1]
    return warped



def lerpResult(warpPrev,denoised,weight):
    lerpResult = torch.lerp(warpPrev, denoised, weight)
    return  lerpResult

class MyDatasetNew(Dataset):
    def __init__(self, data_dir, Number, isTrain=True):
        self.Number = Number
        self.isTrain =isTrain
        self.Width = 1280
        self.Height = 720
        self.data_dir = data_dir# F:/NeuralTemporalKP/

        self.noise_paths_dir = os.path.join(self.data_dir, 'noise/')
        self.noise_paths = make_dataset(self.noise_paths_dir, Number)#所有noise的到名称的地址的列表

        self.reference_paths_dir = os.path.join(self.data_dir, 'reference/')
        self.reference_paths = make_dataset(self.reference_paths_dir, Number)

        #G buffer
        self.albedo_paths_dir = os.path.join(self.data_dir, 'albedo/')
        self.albedo_paths = make_dataset(self.albedo_paths_dir, Number)
        self.depth_paths_dir = os.path.join(self.data_dir, 'depth/')
        self.depth_paths = make_dataset(self.depth_paths_dir, Number)
        self.normal_paths_dir = os.path.join(self.data_dir, 'normal/')
        self.normal_paths = make_dataset(self.normal_paths_dir, Number)
        self.motionVector_paths_dir = os.path.join(self.data_dir, 'zzMV/pt/')#用于reproject
        self.motionVector_paths = make_dataset(self.motionVector_paths_dir, Number)
        self.TmotionVector_paths_dir = os.path.join(self.data_dir, 'svgfMV/pt/')#传统mv
        self.TmotionVector_paths = make_dataset(self.TmotionVector_paths_dir, Number)

        self.visualMV_paths_dir = os.path.join(self.data_dir, 'svgfMV/png/')#用于input
        self.visualMV_paths = make_dataset(self.visualMV_paths_dir,Number)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.h, self.w = initIndex(self.Height, self.Width)

        #初始化第一帧的warpPrev为第一帧的noise
        self.warpPrev = Image.open(self.noise_paths[0]).convert('RGB')
        self.warpPrev_tensor=self.transform(self.warpPrev).to(device)

        #初始化第一帧的prevDenoise为第一帧的noise
        self.prevDenoise = Image.open(self.noise_paths[0]).convert('RGB')
        self.prevDenoise_tensor = self.transform(self.prevDenoise).to(device)

        self.grid_x,self.grid_y,self.w,self.h,self.xyr = self.get_grid()

    def __getitem__(self, index):
        noise_path = self.noise_paths[index]
        reference_path=self.reference_paths[index]
        albedo_path = self.albedo_paths[index]
        depth_path = self.depth_paths[index]
        normal_path = self.normal_paths[index]
        visualMV_path = self.visualMV_paths[index]

        if (index%10 == 9):
            NextmotionVector_path = self.motionVector_paths[index]#motionVector_paths
        else:
            NextmotionVector_path = self.motionVector_paths[index + 1]#motionVector_paths

        #if (index < self.Number-1):
        #    NextmotionVector_path = self.motionVector_paths[index + 1]  # motionVector_paths
        #else:
        #    NextmotionVector_path = self.motionVector_paths[index]
        if(index%10 ==0):
        #if (index == 0):
            traMV_path = self.TmotionVector_paths[index]
            preTraMV1_path = self.TmotionVector_paths[index]
            preTraMV2_path = self.TmotionVector_paths[index]
            occMV_path = self.motionVector_paths[index]
            preOccMV1_path = self.motionVector_paths[index]
            preOccMV2_path = self.motionVector_paths[index]
        if(index%10 ==1):
        #if (index == 1):
            traMV_path = self.TmotionVector_paths[index]
            preTraMV1_path = self.TmotionVector_paths[index-1]
            preTraMV2_path = self.TmotionVector_paths[index-1]
            occMV_path = self.motionVector_paths[index]
            preOccMV1_path = self.motionVector_paths[index-1]
            preOccMV2_path = self.motionVector_paths[index-1]

        traMV_path = self.TmotionVector_paths[index]
        preTraMV1_path = self.TmotionVector_paths[index-1]
        preTraMV2_path = self.TmotionVector_paths[index-2]
        occMV_path = self.motionVector_paths[index]
        preOccMV1_path = self.motionVector_paths[index-1]
        preOccMV2_path = self.motionVector_paths[index-2]

        noise = Image.open(noise_path).convert('RGB')
        reference = Image.open(reference_path).convert('RGB')

        albedo = Image.open(albedo_path).convert('RGB')
        depth = Image.open(depth_path).convert('RGB')
        normal = Image.open(normal_path).convert('RGB')

        visualMV = Image.open(visualMV_path).convert('RGB')

        noise_tensor = self.transform(noise).to(device)
        albedo_tensor = self.transform(albedo).to(device)
        depth_tensor = self.transform(depth).to(device)
        depth_tensor = depth_tensor[:1,:,:]#depth只需要一维就行了
        normal_tensor = self.transform(normal).to(device)

        visualMV_tensor = self.transform(visualMV).to(device)
        visualMV_tensor2 = visualMV_tensor[:2,:,:]

        traMV_tensor : torch.Tensor = torch.load(traMV_path)
        traMV_tensor.squeeze_(0)#2,720,1280
        traMV_tensor=traMV_tensor.to(device)

        preTraMV1_tensor :torch.Tensor = torch.load(preTraMV1_path)
        preTraMV1_tensor.squeeze_(0)
        preTraMV1_tensor=preTraMV1_tensor.to(device)

        preTraMV2_tensor: torch.Tensor = torch.load(preTraMV2_path)
        preTraMV2_tensor.squeeze_(0)
        preTraMV2_tensor = preTraMV2_tensor.to(device)

        occMV_tensor : torch.Tensor = torch.load(occMV_path)
        occMV_tensor.squeeze_(0)
        occMV_tensor= occMV_tensor.to(device)

        preOccMV1_tensor : torch.Tensor = torch.load(preOccMV1_path)
        preOccMV1_tensor.squeeze_(0)
        preOccMV1_tensor=preOccMV1_tensor.to(device)

        preOccMV2_tensor: torch.Tensor = torch.load(preOccMV2_path)
        preOccMV2_tensor.squeeze_(0)
        preOccMV2_tensor = preOccMV2_tensor.to(device)

        NextmotionVector_tensor : torch.Tensor = torch.load(NextmotionVector_path)
        NextmotionVector_tensor.squeeze_(0)
        NextmotionVector_tensor=NextmotionVector_tensor.to(device)

        #NextmotionVectorT_tensor: torch.Tensor = torch.load(NextmotionVectorT_path)
        #NextmotionVectorT_tensor.squeeze_(0)
        #NextmotionVectorT_tensor = NextmotionVectorT_tensor.to(device)

        prevDenoise_tensor = self.prevDenoise_tensor

        if (index%10 == 0):
        #if (index == 0):
            prevRef_tensor_path = self.reference_paths[index]
            prevDenoise_tensor_path = self.noise_paths[index]
            prevDenoise = Image.open(prevDenoise_tensor_path).convert('RGB')
            prevDenoise_tensor = self.transform(prevDenoise).to(device)
            self.warpPrev_tensor = prevDenoise_tensor
        else:
            prevRef_tensor_path = self.reference_paths[index - 1]


        prevRef = Image.open(prevRef_tensor_path).convert('RGB')
        prevRef_tensor = self.transform(prevRef).to(device)

        input_tensor = torch.cat((noise_tensor,albedo_tensor,
                   normal_tensor,depth_tensor,self.warpPrev_tensor),0)#,visualMV_tensor2)

        reference_tensor = self.transform(reference).to(device)

        return {'noise': noise_tensor, 'input': input_tensor, 'reference': reference_tensor,
                'traMV':traMV_tensor,'occMV':occMV_tensor,
                'preTraMV1':preTraMV1_tensor,'preOccMV1':preOccMV1_tensor,
                'preTraMV2':preTraMV2_tensor,'preOccMV2':preOccMV2_tensor,#'NmotionVectorT':NextmotionVectorT_tensor,
                'NmotionVector':NextmotionVector_tensor,'warpPrev':self.warpPrev_tensor,'prevRef':prevRef_tensor,'prevDenoise':prevDenoise_tensor}

    def __len__(self):
        if len(self.noise_paths) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.noise_paths)

    def get_grid(self):
        w, h = (1280, 720)
        x = torch.linspace(0, w - 1, steps=w)
        y = torch.linspace(0, h - 1, steps=h)
        grid_y, grid_x = torch.meshgrid(y, x)
        grid_x = grid_x.to(device='cuda')
        grid_y = grid_y.to(device='cuda')
        grid_x = grid_x.unsqueeze(0).unsqueeze(3)
        grid_y = grid_y.unsqueeze(0).unsqueeze(3)

        xr = torch.ones(1, h, w, 1) * (2. / (w - 1))
        yr = torch.ones(1, h, w, 1) * (2. / (h - 1))
        xyr = torch.cat((xr, yr), dim=3).to(device='cuda')

        return grid_x,grid_y,w,h,xyr

    def tp(self,preResult: torch.Tensor, motion: torch.Tensor, curInput: torch.Tensor,
           device: torch.device = torch.device("cuda:0")):
        # print(motion.size())
        start_warp_time = datetime.now()
        motion = motion.permute(0, 2, 3, 1)
        #w, h = (1280, 720)

        #x = torch.linspace(0, w - 1, steps=w)
        #y = torch.linspace(0, h - 1, steps=h)
        #grid_y, grid_x = torch.meshgrid(y, x)
        #grid_x = grid_x.to(device='cuda')
        #grid_y = grid_y.to(device='cuda')

        mx, my = torch.split(motion, 1, dim=3)
        #grid_x = grid_x.unsqueeze(0).unsqueeze(3)
        grid_mx = self.grid_x + mx
        #grid_y = grid_y.unsqueeze(0).unsqueeze(3)
        grid_my = self.grid_y + my
        grid = torch.cat((grid_mx, grid_my), dim=3)
        #xr = torch.ones(1, self.h, self.w, 1) * (2. / (self.w - 1))
        #yr = torch.ones(1, self.h, self.w, 1) * (2. / (self.h - 1))
        #xyr = torch.cat((xr, yr), dim=3).to(device='cuda')
        #grid = (grid * xyr - 1).to(device='cuda')
        #xyr = torch.cat((xr, yr), dim=3)
        grid = (grid * self.xyr - 1)


        warped = F.grid_sample(preResult, grid, align_corners=True)

        outOfBorderX: torch.Tensor = (grid_mx < 0) | (grid_mx >= self.w)
        outOfBorderY: torch.Tensor = (grid_my < 0) | (grid_my >= self.h)


        #outOfBorder = (outOfBorderX | outOfBorderY).permute(0, 3, 1, 2).to(device='cuda')
        outOfBorder = (outOfBorderX | outOfBorderY).permute(0, 3, 1, 2)

        warped = torch.where(outOfBorder, curInput, warped)
        warp_time = millis(start_warp_time)
        return warped, warp_time

