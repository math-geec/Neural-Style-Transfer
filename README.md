# Neural-Style-Transfer

This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content  of an image with the style of another image. 

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). 

# Demo

Input image:
![content](/images/content/image1.jpg)

Style image:
![style](/images/style/rain_night.jpg)

Output image:
![output](/images/output/out1.jpg)

*Known issue*:



- cuda = 0 kills all the RAM



- cuda = 1 runs out of drive space



*Possible solution*: reduce content image size 
