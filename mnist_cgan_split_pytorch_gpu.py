import torch 
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import math

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)




seed = 11
np.random.seed(seed)
torch.manual_seed(seed)
'''
model parameter
'''
g_seperate_input_size = 100
g_seperate_output_size = 10
d_seperate_input_size = 600
d_seperate_output_size = 10
label_size = 10

class DataDistribution(object):
    def __init__(self,mu = 0,sigma = 1):
        self.mu = mu
        self.sigma = sigma

    def sample(self, in_s,bs):
        samples = np.random.uniform(-1. , 1.,(bs,in_s)).astype(np.float32)
        return samples



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=100,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='the batch size')
    parser.add_argument('--learning-rate', type=float, default = 0.0002,
                        help='learning rate')
    parser.add_argument('--z-size', type=int, default=100,
                        help='the z size')
    parser.add_argument('--eval', action='store_true',
                        help='eval mode')
    parser.add_argument('--save', action='store_true',
                        help='save on')
    return parser.parse_args()


'''
m  =model
e  =epoch
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
    plt.savefig("mnist_gan_loss_epoch%d" %e)
    plt.close()



class generator(nn.Module):
      def __init__(self,input_size, hidden_size, output_size):
          super(generator, self).__init__()
          self.fc1 = nn.Linear(input_size  , g_seperate_input_size)
          self.fc4 = nn.Linear(label_size, g_seperate_output_size)
          self.fc5 = nn.Linear(g_seperate_input_size+g_seperate_output_size,hidden_size)
          self.stp = nn.ReLU()
          self.fc2  = nn.Linear(hidden_size,2* hidden_size)
          self.fc3 = nn.Linear(2*hidden_size,output_size)
          self.do = nn.Dropout()
          self.sg  = nn.Sigmoid()
  
      def forward(self, x,y):
          out_1 = self.fc1(x)
          out_2 = self.fc4(y)
          out = torch.cat((out_1,out_2),1)
          out = self.fc5(out)
          out = self.stp(out)
          out = self.do(out)
          out = self.fc2(out)
          out = self.stp(out)
          out = self.do(out)
          out = self.fc3(out)
          out = self.sg(out)
          return out

class discriminator(nn.Module):
      def __init__(self, input_size, hidden_size):
          super(discriminator,self).__init__()
          self.fc1 = nn.Linear(input_size  , d_seperate_input_size)
          self.fc4 = nn.Linear(label_size, d_seperate_output_size)
          self.fc5 = nn.Linear(d_seperate_input_size+d_seperate_output_size,hidden_size)
          self.relu= nn.ReLU()
          self.bn1= nn.BatchNorm1d(hidden_size)
          self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
          self.bn2= nn.BatchNorm1d( int(hidden_size/2))
          self.fc3 = nn.Linear( int(hidden_size/2),1)
          self.sg  = nn.Sigmoid()
          self.do = nn.Dropout()
      def forward(self,x,y):
          out_1 = self.fc1(x)
          out_2 = self.fc4(y)
          out = torch.cat((out_1,out_2),1)
          out = self.fc5(out)
          out = self.relu(out)
          out = self.do(out)
          out = self.fc2(out)
          out = self.relu(out)
          out = self.do(out)
          out = self.fc3(out)
          out = self.sg(out)
          return out

class GAN(object):
      def __init__(self,params,in_s,in_g):
          self.in_g = in_g
          self.g = generator(in_g,params.hidden_size,in_s)
          self.d = discriminator(in_s,params.hidden_size)
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
    
    ones = Variable(torch.ones(model.batch_size,1).cuda())
    zeros = Variable(torch.zeros(model.batch_size,1).cuda())
    a_loss_g = []
    a_loss_d = []
	
	
    for i in range(model.epoch):
     e_loss_g = 0
     e_loss_d = 0
     print("epoch :%s" %i)
     
     for m,(image,label) in enumerate(trl):
       ds = Variable(image.view(-1,28*28).cuda())
       gs = Variable(torch.from_numpy(gd.sample(model.in_g,model.batch_size)).cuda())
       one_hot_label = torch.FloatTensor(model.batch_size,label_size).zero_()
       one_hot_label.scatter_(1,label.view(-1,1),1)
       v_label = Variable(one_hot_label.cuda())
      #print(one_hot_label) 
       model.d_opt.zero_grad()

       d1 = model.d(ds,v_label)
       g = model.g(gs,v_label)
       d2 = model.d(g,v_label)
      
       loss_d1 = model.ct(d1,ones)
       loss_d2 = model.ct(d2,zeros)

       loss = loss_d1 + loss_d2
       loss.backward(retain_graph=True)
       model.d_opt.step()


       model.g_opt.zero_grad()
        
       loss_g = model.ct(d2,ones)
       loss_g.backward()
       model.g_opt.step()

       e_loss_g += loss_g.data[0]
       e_loss_d += loss.data[0]
       
     a_loss_g.append(e_loss_g/trl.__len__())
     a_loss_d.append(e_loss_d/trl.__len__())
     drawlossplot(model,a_loss_g,a_loss_d,i)
     generateimage(model,gd,i,0)


def generateimage(m, gd, e, new):
    m.g.eval()
    if new:
        gs = Variable(torch.from_numpy(gd.sample(m.in_g, m.batch_size)).cuda())
        g = m.g(gs, g_v_label)
    else:
        g = m.g(g_gs, g_v_label)
    g = g.data.cpu().numpy()
    g = g.reshape(-1, 28, 28)
    fig = plt.figure(figsize=(28, 28), tight_layout=True)
    grid = label_size
    for i in range(grid * grid):
        ax = fig.add_subplot(grid, grid, i + 1)
        ax.set_axis_off()
        idx = int(i / label_size) * m.batch_size + i % label_size
        plt.imshow(g[idx], shape=(28, 28), cmap='Greys_r')
    title = 'Epoch {0}'.format(e + 1)
    fig.text(0.5, 0.04, title, ha='center')
    plt.savefig("mnist_gan_epoch%s" % e)
    plt.close()
    m.g.train()


'''
global variable for draw....
'''
g_arg = parse_args()
g_gd = np.random.uniform(-1. , 1.,(g_arg.batch_size,g_arg.z_size)).astype(np.float32)
g_gd = np.tile(g_gd, (label_size,1))
g_gs = Variable(torch.from_numpy(g_gd).view(-1,g_arg.z_size).cuda())

g_c  = np.linspace(0,label_size-1,label_size).astype(np.int)
g_c  = np.repeat(g_c, g_arg.batch_size)
g_c = torch.LongTensor(g_c)

g_one_hot_label = torch.FloatTensor(label_size*g_arg.batch_size,label_size).zero_()
g_one_hot_label.scatter_(1, g_c.view(-1,1), 1)

g_v_label = Variable(g_one_hot_label.cuda())

def main(args):
    trl = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=args.batch_size, 
                                           shuffle=True)



    model = GAN(args,28*28,args.z_size)
    dd = DataDistribution(0,1)
    
    if args.eval:
     for i in range(10):
      generateimage(model,dd,i)
    else: 
     train(model, trl, dd)    
   
    
    if args.save:
     model.save()

if __name__ == '__main__':
    main(parse_args())















