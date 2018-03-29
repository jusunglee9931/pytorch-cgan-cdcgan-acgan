import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import math


train_dataset = dsets.CIFAR10(root='./data', 
                            train=True, 
                            transform=transforms.Compose([
                                                          transforms.Resize(64)
                                                          ]),  
                            download=True)

class DataDistribution(object):
    def __init__(self,mu = 0,sigma = 1):
        self.mu =mu 
        self.sigma =sigma
    def sample(self, z_size,bs):
        samples = np.random.uniform(-1. , 1., (bs,z_size,1,1)).astype(np.float32)
       # samples.sort()
        return samples




seed = 11
np.random.seed(seed)
torch.manual_seed(seed)

'''
variables
'''
x_size =64
y_size =64
in_ch = 3
out_ch =3
z_size =100
label_size = 10
batch_size = 16
g_gd = np.random.uniform(-1. , 1.,(batch_size, z_size)).astype(np.float32)
g_gd = np.tile(g_gd, (label_size,1))
g_gd = torch.from_numpy(g_gd).view(-1,z_size).unsqueeze(-1).unsqueeze(-1)
g_gs = Variable(g_gd.cuda())


g_c  = np.linspace(0,label_size-1,label_size).astype(np.int)
g_c  = np.repeat(g_c, batch_size)
g_c = torch.LongTensor(g_c)

g_one_hot_label = torch.FloatTensor(batch_size*label_size,label_size).zero_()
g_one_hot_label.scatter_(1, g_c.view(-1,1), 1)
g_one_hot_label = g_one_hot_label.unsqueeze(-1).unsqueeze(-1)
g_v_label = Variable(g_one_hot_label.cuda())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=100,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='the batch size')
    parser.add_argument('--learning-rate', type=float, default = 0.0002,
                        help='learning rate')
    parser.add_argument('--eval', action='store_true',
                        help='eval mode')
    parser.add_argument('--save', action='store_true',
                        help='save on')
    return parser.parse_args()



'''
m  =model

'''
def drawlossplot( m,loss_g,loss_d,e):
    print(loss_g)
    g_x = np.linspace(0, len(loss_g), len(loss_g))
    f, ax = plt.subplots(1)
   
    plt.plot(g_x, loss_g, label='loss_g')
    plt.plot(g_x, loss_d, label='loss_d')
    ax.set_xlim(0, m.epoch)

    plt.title('Vanilla Generative Adversarial Network Loss Graph')
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.legend()
    plt.savefig("cifar_cdcgan_loss_epoch%d" %e)
    plt.close()

def convblocklayer(in_ch,out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size =4, stride = 2,padding = 1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU()
                         )
def deconvblocklayer(in_ch,out_ch,pad):
    return nn.Sequential(nn.ConvTranspose2d(in_ch,out_ch,kernel_size = 4, stride = 2,padding = pad),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU()
                         )

class generator(nn.Module):
      def __init__(self,z_channel,o_channel):
          super(generator, self).__init__()
          self.layer1= deconvblocklayer(z_channel,1024,0)
          self.layer2= deconvblocklayer(1024,512,1)
          self.layer3= deconvblocklayer(512,256,1)
          self.layer4= deconvblocklayer(256,128,1)
          self.conv5 = nn.ConvTranspose2d(128,o_channel, kernel_size =4,stride = 2, padding =1)
          self.sg    = nn.Sigmoid()
      def forward(self, x,y):
          out = torch.cat((x,y),1)
          out = self.layer1(out)
          out = self.layer2(out)
          out = self.layer3(out)
          out = self.layer4(out)
          out = self.conv5(out)
          out = self.sg(out)
          return out

class discriminator(nn.Module):
      def __init__(self, channel):
          super(discriminator,self).__init__()
          self.layer1 =convblocklayer(channel,128)
          self.layer2 =convblocklayer(128,256)
          self.layer3 =convblocklayer(256,512)
          self.layer4 =convblocklayer(512,1024)
          self.conv5 = nn.Conv2d(1024,1, kernel_size =4)
          self.sg = nn.Sigmoid()
      def forward(self,x,y):
          out = torch.cat((x,y),1) 
          out = self.layer1(out)
          out = self.layer2(out)
          out = self.layer3(out)
          out = self.layer4(out)
          out = self.conv5(out)
          out = self.sg(out)
          return out


class GAN(object):
      def __init__(self,params,in_ch,o_ch):
          self.g = generator(z_size+label_size,o_ch)
          self.d = discriminator(in_ch+label_size)
          self.g.cuda(0)
          self.d.cuda(0)
          self.batch_size = params.batch_size
          self.lr = params.learning_rate
          self.ct = nn.BCELoss()
          self.g_opt = torch.optim.Adam(self.g.parameters(),lr = self.lr)
          self.d_opt = torch.optim.Adam(self.d.parameters(),lr = self.lr)
          self.epoch = params.num_steps
      def save(self):
          torch.save(self.g,"g.pt")
          torch.save(self.d,"d.pt")
      def load(self):
          self.g = torch.load("g.pt")
          self.d = torch.load("d.pt")

def train(model,trl,gd):
    
    ones = Variable(torch.ones(model.batch_size,1,1,1).cuda())
    zeros = Variable(torch.zeros(model.batch_size,1,1,1).cuda())
    batch_img = torch.FloatTensor()
    trans = transforms.ToTensor()
    batch_label = []
    a_loss_g = []
    a_loss_d = []
    for i in range(model.epoch):
     e_loss_g = 0
     e_loss_d = 0
     
     print("epoch :%s" %i)
     for m,(images,label) in enumerate(trl):
       batch_img = torch.cat((batch_img,trans(images).unsqueeze(0)))
       batch_label.append(label)
       #print(batch_label)
       if (m+1) % model.batch_size == 0 :
        ds = Variable(batch_img.cuda())
        gs = Variable(torch.from_numpy(gd.sample(z_size,model.batch_size)).cuda())
        
        one_hot_label = torch.FloatTensor(model.batch_size,label_size).zero_()
        tensor_label = torch.LongTensor(batch_label)
        one_hot_label.scatter_(1,tensor_label.view(-1,1),1)
        one_hot_label = one_hot_label.unsqueeze(-1).unsqueeze(-1)
        v_label = Variable(one_hot_label.cuda())
       
        v_batch_label = np.repeat(batch_label, y_size*x_size) 
        v_batch_label = torch.LongTensor(v_batch_label)
        one_ch_label = torch.FloatTensor(model.batch_size,
                                                                            label_size,
                                                                            y_size,
                                                                            x_size
                                                                            ).zero_()
        one_ch_label.scatter_(1, v_batch_label.view(model.batch_size,
                                                                            1,
                                                                            y_size,
                                                                            x_size
                                                                            ), 1)
        v_d_label = Variable(one_ch_label.cuda())

        model.d_opt.zero_grad()

        d1 = model.d(ds,v_d_label)
        g = model.g(gs,v_label)
        d2 = model.d(g,v_d_label)
      
        loss_d1 = model.ct(d1,ones)
        loss_d2 = model.ct(d2,zeros)

        loss = loss_d1 + loss_d2
        loss.backward(retain_graph=True)
        model.d_opt.step()


        model.g_opt.zero_grad()
        
        loss_g = model.ct(d2,ones)
        loss_g.backward()
        model.g_opt.step()
		
        batch_img = torch.FloatTensor()
        batch_label = []
        e_loss_g += loss_g.data[0]
        e_loss_d += loss.data[0]
        #if (m+1) > 2*model.batch_size:
        # break
     a_loss_g.append(e_loss_g/trl.__len__())
     a_loss_d.append(e_loss_d/trl.__len__())
     drawlossplot(model,a_loss_g,a_loss_d,i)
     generateimage(model,gd,i,0)  

  

def generateimage(m,gd,e,new=False):
       m.g.eval()
       if new :
        gs = Variable(torch.from_numpy(gd.sample(z_size,m.batch_size)).cuda())
        g = m.g(gs,g_v_label)
       else :
        g = m.g(g_gs,g_v_label)
       g = g.data.cpu().numpy()
       g = g.reshape(-1,out_ch,y_size,x_size)
       #print(g)
       fig = plt.figure(figsize=(y_size,x_size),tight_layout=True)
       grid = label_size
       for i in range(grid*grid):
        ax = fig.add_subplot(grid,grid,i+1)
        ax.set_axis_off()
        idx = int(i/label_size)*m.batch_size + i % label_size
        tp = np.moveaxis(g[idx],0,-1)
        plt.imshow(tp,shape=(y_size,x_size))
       plt.savefig("cdcgan_figure_epoch%s" %e)
       plt.close()
       m.g.train()




def main(args):
   
    model = GAN(args,in_ch,out_ch)
    dd = DataDistribution(0,1)
    if args.eval:
     for i in range(10):
      generateimage(model,dd,i)
    else: 
     train(model, train_dataset, dd)    
   
    
    if args.save:
     model.save()

if __name__ == '__main__':
    main(parse_args())










