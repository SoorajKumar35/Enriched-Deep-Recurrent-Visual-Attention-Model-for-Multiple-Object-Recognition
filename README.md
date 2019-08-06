# Enriched-Deep-Recurrent-Visual-Attention-Model-for-Multiple-Object-Recognition

Two common and related problems in computer vision are object detection and object classification. Enriched Deep Recurrent Visual Attention for Multiple Object Recognition, abbreviated as EDRAM, proposes an approach for efficiently and accurately solving both of these tasks in the MNIST cluttered and SVHN datasets. 

Convolutional neural networks work well for object classification, but have problems with
object detection and localization, and computation cost scales linearly with the number of image pixels. These were
followed by attempts at visual attention models, based loosely on human vision patterns, in which the network can
glimpse parts of the image, decide where to look next, and store information of past glimpses in a recurrent module
to assist in classification. In Recurrent Models of Visual Attention, DeepMind sought to solve the problem with a
recurrent visual attention model where the computational complexity is independent of image size. Weaknesses of
this approach included its non-differentiability and reliance on reinforcement learning. EDRAM is a refinement of
these visual attention models, and achieved state-of-the-art results at time of its publication. In addition, EDRAM is
end-to-end differentiable, allowing for training by stochastic gradient descent.
