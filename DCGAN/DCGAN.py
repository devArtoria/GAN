#INIT

'''
batch size 100

learning rate 0.0002

epoch 20

leakyReLU 0.2

Adam
'''


'''
Net config

Discriminator

conv 1
64*64*1
4*4 filter 2 stride

conv 2
32*32*128
4*4 filter 2 stride

leaky_relu

conv 3
16*16*256
4*4 filter 2 stride

bnorm

leaky_relu

conv 4
8*8*512
4*4 filter 2 stride

bnorm

leaky_relu

conv 5
4*4*1024
4*4 filter 2 stride

bnorm

leaky_relu

sigmoid(1*1 output)




Generator

input 100

dconv1
4*4*1024
4*4 filter 2 stride

dconv2
8*8*512
4*4 filter 2 stride

bnorm

leaky_relu

dconv3
16*16*256
4*4 filter 2 stride

bnorm

leaky_relu

dconv4 
32*32*128
4*4 filter 2 stride

bnorm

leaky_relu

dconv5
64*64*1

TanH
'''