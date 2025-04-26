# cs7643-assignment-3-deep-learning-solved
**TO GET THIS SOLUTION VISIT:** [CS7643: Assignment 3 Deep Learning Solved](https://www.ankitcodinghub.com/product/cs7643-deep-learning-solved-3/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;105502&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;4&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (4 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS7643: Assignment 3 Deep Learning Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (4 votes)    </div>
    </div>
• Discussion is encouraged, but each student must write his/her own answers and explicitly mention any collaborators.

• It is your responsibility to make sure that all code and other deliverables are in the correct format and that your submission compiles and runs. We will not manually check your code (this is not feasible given the class size). Thus, non-runnable code in our test environment will directly lead to a score of 0. Also, your entire programming parts will NOT be graded and given 0 score if your code prints out anything that is not asked in each question.

Python and dependencies

In this assignment, we will work with Python 3. If you do not have a python distribution installed yet, we recommend installing Anaconda (or miniconda) with Python 3. We provide environment.yaml which contains a list of libraries needed to set the environment for this assignment. You can use it to create a copy of conda environment. Refer to the users’ manual for more details.

$ conda env create -f environment.yaml

We recommend using PyTorch 1.9.1 and torchvision 0.2.2 to finish the problems in this assignment.

If you already have your own Python development environment, please refer to this file to find necessary libraries, which are used to set the same coding/grading environment.

$ pip install future

$ pip install scipy

$ pip install torchvision

Additionally, you will work with Captum in this assignment. Make sure you follow the instruction in the official document of Captum to install it in your environment. You can use the following command to install captum $ conda install captum -c pytorch

1 Network Visualization

In the first part we will explore the use of different type of attribution algorithms – both gradient and perturbation – for images, and understand their differences using the Captum model interpretability tool for PyTorch. As an exercise you’ll be also asked to implement Saliency Maps from scratch.

1. Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. ”Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps”, ICLR Workshop 2014.

2. Mukund Sundararajan, Ankur Taly, Qiqi Yan, ”Axiomatic Attribution for Deep Networks”, ICML, 2017

3. Matthew D Zeiler, Rob Fergus, ”Visualizing and Understanding Convolutional Networks”, Visualizing and Understanding Convolutional Networks, 2013.

4. Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, 2016 In the second and third parts we will focus on generating new images, by studying and implementing key components in two papers:

1. Szegedy et al, ”Intriguing properties of neural networks”, ICLR 2014

2. Yosinski et al, ”Understanding Neural Networks Through Deep Visualization”, ICML 2015 Deep Learning Workshop

You will need to first read the papers, and then we will guide you to understand them deeper with some problems.

When training a model, we define a loss function which measures our current unhappiness with the model’s performance; we then use backpropagation to compute the gradient of the loss with respect to the model parameters, and perform gradient descent on the model parameters to minimize the loss.

In this homework, we will do something slightly different. We will start from a convolutional neural network model which has been pretrained to perform image classification on the ImageNet dataset. We will use this model to define a loss function which quantifies our current unhappiness with our image, then use backpropagation to compute the gradient of this loss with respect to the pixels of the image. We will then keep the model fixed, and perform gradient descent on the image to synthesize a new image which minimizes the loss.

We will explore four different techniques:

1. Saliency Maps: Saliency maps are a quick way to tell which part of the image influenced the classification decision made by the network.

2. GradCAM: GradCAM is a way to show the focus area on an image for a given label.

3. Fooling Images: We can perturb an input image so that it appears the same to humans, but will be misclassified by the pretrained network.

4. Class Visualization: We can synthesize an image to maximize the classification score of a particular class; this can give us some sense of what the network is looking for when it classifies images of that class.

1.1 Saliency Map

Using this pretrained model, we will compute class saliency maps as described in the paper:

[1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. ”Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps”, ICLR Workshop 2014.

A saliency map tells us the degree to which each pixel in the image affects the classification score for that image. To compute it, we compute the gradient of the unnormalized score corresponding to the correct class (which is a scalar) with respect to the pixels of the image. If the image has shape (3, H, W) then this gradient will also have shape (3, H, W); for each pixel in the image, this gradient tells us the amount by which the classification score will change if the pixel changes by a small amount. To compute the saliency map, we take the absolute value of this gradient, then take the maximum value over the 3 input channels; the final saliency map thus has shape (H, W) and all entries are nonnegative. Your tasks are as follows:

1. Follow instructions and implement functions in visualizers/saliency_map.py, which manually computes the saliency map

2. Follow instructions and implement Saliency Map with Captum in root/saliency_map.py

As the final step, you should run the python script root/saliency_map.py to generate plots for visualization.

1.2 GradCam

GradCAM (which stands for Gradient Class Activation Mapping) is a technique that tells us where a convolutional network is looking when it is making a decision on a given input image. There are three main stages to it:

1. Guided Backprop (Changing ReLU Backprop Layer, Link)

2. GradCAM (Manipulating gradients at the last convolutional layer, Link)

3. Guided GradCAM (Pointwise multiplication of above stages)

In this section, you will be implementing these three stages to recreate the full GradCAM pipeline. Your tasks are as follows:

1. Follow instructions and implement functions in visualizers/gradcam.py, which manually computes guided backprop and GradCam

2. Follow instructions and implement GradCam with Captum in root/gradcam.py

As the final step, you should run the python script root/gradcam.py to generate plots for visualization.

1.3 Fooling Image

We can also use the similar concept of image gradients to study the stability of the network. Consider a state-of-the-art deep neural network that generalizes well on an object recognition task. We expect such network to be robust to small perturbations of its input, because small perturbation cannot change the object category of an image. However, [2] find that applying an imperceptible non-random perturbation to a test image, it is possible to arbitrarily change the network’s prediction.

[2] Szegedy et al, ”Intriguing properties of neural networks”, ICLR 2014

Given an image and a target class, we can perform gradient ascent over the image to maximize the target class, stopping when the network classifies the image as the target class. We term the so perturbed examples “adversarial examples”.

Read the paper, and then implement the following function to generate fooling images. Your tasks are as follows:

1. Follow instructions and implement functions in visualizers/fooling_image.py, which manually computes the fooling image

As the final step, you should run the python script root/fooling_image.py to generate fooling images.

1.4 Class Visualization

By starting with a random noise image and performing gradient ascent on a target class, we can generate an image that the network will recognize as the target class. This idea was first presented in [1]; [3] extended this idea by suggesting several regularization techniques that can improve the quality of the generated image.

Concretely, let I be an image and let y be a target class. Let sy(I) be the score that a convolutional network assigns to the image I for class y; note that these are raw unnormalized scores, not class probabilities. We wish to generate an image I∗ that achieves a high score for the class y by solving the problem

I∗ = argmaxsy(I)− R(I)

I

where R is a (possibly implicit) regularizer (note the sign of R(I) in the argmax: we want to minimize this regularization term). We can solve this optimization problem using gradient ascent, computing gradients with respect to the generated image. We will use (explicit) L2 regularization of the form

and implicit regularization as suggested by [3] by periodically blurring the generated image. We can solve this problem using gradient ascent on the generated image.

[1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. ”Deep Inside Convolutional Networks: Visualising Image Classification Models and

Saliency Maps”, ICLR Workshop 2014(https://arxiv.org/abs/1312.6034)

[3] Yosinski et al, ”Understanding Neural Networks Through Deep Visualization”, ICML 2015 Deep Learning Workshop Your tasks are as follows:

1. Follow instructions and implement functions in visualizers/class_visualization.py, which manually computes the class visualization As the final step, you should run the python script root/class_visualization.py to generate fooling images.

2 Style Transfer

Another task closely related to image gradient is style transfer. This has become a cool application in deep learning with computer vision. In this section we will study and implement the style transfer technique from:

”Image Style Transfer Using Convolutional Neural Networks” (Gatys et al., CVPR 2015).

The general idea is to take two images (a content image and a style image), and produce a new image that reflects the content of one but the artistic ”style” of the other. We will do this by first formulating a loss function that matches the content and style of each respective image in the feature space of a deep network, and then performing gradient descent on the pixels of the image itself.

2.1 Content Loss

We can generate an image that reflects the content of one image and the style of another by incorporating both in our loss function. We want to penalize deviations from the content of the content image and deviations from the style of the style image. We can then use this hybrid loss function to perform gradient descent not on the parameters of the model, but instead on the pixel values of our original image.

Let’s first write the content loss function. Content loss measures how much the feature map of the generated image differs from the feature map of the source image. We only care about the content representation of one layer of the network (say, layer `), that has feature maps A` ∈R1×C`×H`×W`. C` is the number of channels in layer `, H` and W` are the height and width. We will work with reshaped versions of these feature maps that combine all spatial positions into one dimension. Let F` ∈RN`×M` be the feature map for the current image and P` ∈RN`×M` be the feature map for the content source image where M` = H` × W` is the number of elements in each feature map. Each row of F` or P` represents the vectorized activations of a particular filter, convolved over all positions of the image. Finally, let wc be the weight of the content loss term in the loss function.

Then the content loss is given by:

Lc = wc ×Pi,j(Fij` − Pij` )2

1. Implement Content Loss in style_modules/content_loss.py

You can check your implementation by running the ’Test content loss’ function. The expected error should be 0.0

2.2 Style Loss

Now we can tackle the style loss. For a given layer `, the style loss is defined as follows:

First, compute the Gram matrix G which represents the correlations between the responses of each filter, where F is as above. The Gram matrix is an approximation to the covariance matrix – we want the activation statistics of our generated image to match the activation statistics of our style image, and matching the (approximate) covariance is one way to do that. There are a variety of ways you could do this, but the Gram matrix is nice because it’s easy to compute and in practice shows good results.

Given a feature map F` of shape (1,C`,M`), the Gram matrix has shape (1,C`,C`) and its elements are given by:

G`ij = XFik` Fjk`

k

Assuming G` is the Gram matrix from the feature map of the current image, A` is the Gram Matrix from the feature map of the source style image, and w` a scalar weight term, then the style loss for the layer ` is simply the weighted Euclidean distance between the two Gram matrices:

In practice we usually compute the style loss at a set of layers L rather than just a single layer `; then the total style loss is the sum of style losses at each layer:

Ls = XL`s

`∈L

1. Implement Style Loss in style_modules/style_loss.py

You can check your implementation by running the ’Test style loss’ function. The expected error should be 0.0

2.3 Total Variation Loss

It turns out that it’s helpful to also encourage smoothness in the image. We can do this by adding another term to our loss that penalizes wiggles or **total variation** in the pixel values. This concept is widely used in many computer vision task as a regularization term.

You can compute the ”total variation” as the sum of the squares of differences in the pixel values for all pairs of pixels that are next to each other (horizontally or vertically). Here we sum the total-variation regualarization for each of the 3 input channels (RGB), and weight the total summed loss by the total variation weight,

You should try to provide an efficient vectorized implementation.

1. Implement Style Loss in style_modules/tv_loss.py

You can check your implementation by running ’Test total variation loss’ function. The expected error should be 0.0

2.4 Style Transfer

You have implemented all the loss functions in the paper. Now we’re ready to string it all together. Please read the entire function: figure out what are all the parameters, inputs, solvers, etc. The update rule in function style_transfer of style_utils.py is held out for you to finish.

As a final step, run the script style_transfer.py to generate stylized images.

3 Sample Outputs

We provide some sample outputs for your reference to verify the correctness of your code:

Figure 1: Example Outputs Part 1 in the following order from top to bottom – Original images, Saliency maps, Guided backprop, Gradcam and Guided Gradcam

Figure 2: Example Outputs Part 2

Figure 3: Example Outputs Part 3

Figure 4: Class visualization

4 Deliverables

4.1 Coding

To submit your code to Gradescope, you will need to submit a zip file containing all your codes in structure. For your convenience, we provide a handy script for you.

Simply run

$ bash collect_submission . sh or if running Microsoft Windows 10

C:assignmentfolder&gt;collect_submission . bat then upload assignment_3_submission.zip to Gradescope Assignment 3 Code.

4.2 Writeup

You will need to follow the guidance and fill in the report template. Scripts in the root directory save your visualization and stylized images into specific directories. You will need to include them in your report. You need to upload the report to Gradescope in Assignment 3 Writeup.
