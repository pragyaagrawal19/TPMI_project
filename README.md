# TPMI_project: Study and Analysis of GAN and VAE in continual learning
# Abstract:
Lifelong learners are known to retain and reuse learned behavior acquired over past tasks and aim to
maximize performance across all tasks. Neural networks face the problem of completely forgetting
previously learned information upon learning new information, aka catastrophic forgetting. This
problem of catastrophic forgetting is a central obstacle that needs to be resolved to build an effective
neural lifelong learner. Existing lifelong learning methods have been roughly grouped into three
categories, i.e., regularization-based, dynamic-model-based, and generative-replay-based methods.
Generative replay is believed to be an effective and general strategy for lifelong learning problems,
although most existing methods of this type often have blurry/distorted generation (for images)
or scalability issues. The use of Generative Adversarial Networks and Variational Auto-encoders
have shown to produce commendable results, maintaining the principle of continual learning and
generating good images. Although it has been experimentally confirmed that GANs can produce
higher quality images than VAEs, we would like to study and discover the pros and cons of each
approach which may allow us to further integrate the best of both approaches.
To do so, we would perform the following analysis,
* Reproduce experimental results and gain a measure of how realistic the generated images
are across the two approaches.
* Measure the forgetfulness of the two approaches, find the relationship between number of
tasks and forgetfulness and compare the performance of the two approaches in continual
learning.
* Generate images over a domain perceptually-distant from the domain the models have been
trained over with varying sample sizes (few-shot learning).
Here is report of project work - [report.pdf](https://github.com/pragyaagrawal19/TPMI_project/files/6562755/report.pdf)
# Few shot learning on VAE:
For performing few-shot learning the model was trained on FASHION-MNIST for 60000 sample
and 256 batch-size for 60 epochs. Below graph is for FID-Score plotted for pre-trained model:
![fashion_fid](https://user-images.githubusercontent.com/48952693/120027803-78e23600-c011-11eb-9daa-510fd75fe3de.png)

After pre-training few short learning was performed on MNIST data-set. For performing the
few shot learning a student model was initialized with MNIST data set and model obtained from
pre-trained model was copied into teacher of this new MNIST data-set model. After that it was run
for 300 epochs with sample size of 2, 4, 8, 16, 32, 64, 128, 512, 1024. Here sample-size means that
we have randomly taken that many examples from train and test set both for performing few shot
learning. FID score was computer on test-set.
![mnist1](https://user-images.githubusercontent.com/48952693/120027900-97483180-c011-11eb-9602-85383a3f9d0e.png)
![mnist2](https://user-images.githubusercontent.com/48952693/120027934-a4652080-c011-11eb-9194-eeb9c0c554da.png)

Above graphs show FID-score calculate for each sample-size of 2 to 16 and sample-size of 32 to 1024 respectively. From the graph it can be
seen that, for less number of sample the FID score has zig-zag pattern because of the possibility
of overshooting in model weights. And on other hand when we increase the sample size, graph
stabilizes (we have noticed this trend in sample size 32 and larger sized samples). Also it is obvious
that when we increase the sample size, the model learns well and so the FID-score is lower than the
FID-scores of lesser samples in training.
Generated images for different number of data samples 8, 16, 32, 64, 128, 512, 1024:
![mnist_generation_page-0001](https://user-images.githubusercontent.com/48952693/120028433-4edd4380-c012-11eb-85e0-e41ad9c32f33.jpg)

We can see from generated images that from sample
size 32 on-words model has generated quite good quality images from epoch 100 and it further
improves with more number of epochs.
