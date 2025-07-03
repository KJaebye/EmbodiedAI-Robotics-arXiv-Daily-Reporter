# SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space 

**Title (ZH)**: SE(3)对称扩散政策在球面傅里叶空间中 

**Authors**: Xupeng Zhu, Fan Wang, Robin Walters, Jane Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01723)  

**Abstract**: Diffusion Policies are effective at learning closed-loop manipulation policies from human demonstrations but generalize poorly to novel arrangements of objects in 3D space, hurting real-world performance. To address this issue, we propose Spherical Diffusion Policy (SDP), an SE(3) equivariant diffusion policy that adapts trajectories according to 3D transformations of the scene. Such equivariance is achieved by embedding the states, actions, and the denoising process in spherical Fourier space. Additionally, we employ novel spherical FiLM layers to condition the action denoising process equivariantly on the scene embeddings. Lastly, we propose a spherical denoising temporal U-net that achieves spatiotemporal equivariance with computational efficiency. In the end, SDP is end-to-end SE(3) equivariant, allowing robust generalization across transformed 3D scenes. SDP demonstrates a large performance improvement over strong baselines in 20 simulation tasks and 5 physical robot tasks including single-arm and bi-manual embodiments. Code is available at this https URL. 

**Abstract (ZH)**: Spherical Diffusion Policy在学习基于人类演示的闭环操作策略方面效果显著，但在3D空间中对新颖物体排列的泛化能力差，影响实际性能。为解决这一问题，我们提出了一种SE(3)等变扩散策略Spherical Diffusion Policy (SDP)，该策略根据场景的3D变换调整轨迹。这种等变性通过将状态、动作和去噪过程嵌入球面傅里叶空间实现。此外，我们采用了新的球面FiLM层，使动作去噪过程在场景嵌入的基础上等变地受条件制约。最后，我们提出了一个球面去噪时序U-网，实现了时空等变性并保持了计算效率。最终，SDP 是端到端的SE(3)等变策略，允许在变换的3D场景中稳健泛化。SDP 在20个模拟任务和5个物理机器人任务中（包括单臂和双臂操作）展示了显著的性能提升。代码可在以下链接获取。 

---
# Geometry-aware 4D Video Generation for Robot Manipulation 

**Title (ZH)**: 面向机器人的几何感知4D视频生成 

**Authors**: Zeyi Liu, Shuang Li, Eric Cousineau, Siyuan Feng, Benjamin Burchfiel, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.01099)  

**Abstract**: Understanding and predicting the dynamics of the physical world can enhance a robot's ability to plan and interact effectively in complex environments. While recent video generation models have shown strong potential in modeling dynamic scenes, generating videos that are both temporally coherent and geometrically consistent across camera views remains a significant challenge. To address this, we propose a 4D video generation model that enforces multi-view 3D consistency of videos by supervising the model with cross-view pointmap alignment during training. This geometric supervision enables the model to learn a shared 3D representation of the scene, allowing it to predict future video sequences from novel viewpoints based solely on the given RGB-D observations, without requiring camera poses as inputs. Compared to existing baselines, our method produces more visually stable and spatially aligned predictions across multiple simulated and real-world robotic datasets. We further show that the predicted 4D videos can be used to recover robot end-effector trajectories using an off-the-shelf 6DoF pose tracker, supporting robust robot manipulation and generalization to novel camera viewpoints. 

**Abstract (ZH)**: 理解并预测物理世界的动态可以增强机器人在复杂环境中的规划和交互能力。尽管近期的视频生成模型在建模动态场景方面显示出强大的潜力，但在训练过程中通过跨视图点图对齐监督模型，以确保视频的多视图三维一致性，生成在时间和几何上均一致的视频仍然是一项重大挑战。为此，我们提出了一种4D视频生成模型，通过在训练过程中监督模型的跨视图点图对齐，确保视频的多视图三维一致性。这种几何监督使模型能够学习场景的共享三维表示，从而仅根据给定的RGB-D观测值预测新视角下的未来视频序列，而无需将相机姿态作为输入。与现有基线方法相比，我们的方法在多个模拟和真实世界机器人数据集上生成的预测结果更为视觉稳定且在空间上更一致。我们进一步表明，预测的4D视频可以使用商业6自由度姿态跟踪器来恢复机器人末端执行器轨迹，从而支持鲁棒的机器人操作并能够泛化到新的相机视角。 

---
# Automated Vehicles Should be Connected with Natural Language 

**Title (ZH)**: 自动驾驶车辆应当与自然语言连接。 

**Authors**: Xiangbo Gao, Keshu Wu, Hao Zhang, Kexin Tian, Yang Zhou, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01059)  

**Abstract**: Multi-agent collaborative driving promises improvements in traffic safety and efficiency through collective perception and decision making. However, existing communication media -- including raw sensor data, neural network features, and perception results -- suffer limitations in bandwidth efficiency, information completeness, and agent interoperability. Moreover, traditional approaches have largely ignored decision-level fusion, neglecting critical dimensions of collaborative driving. In this paper we argue that addressing these challenges requires a transition from purely perception-oriented data exchanges to explicit intent and reasoning communication using natural language. Natural language balances semantic density and communication bandwidth, adapts flexibly to real-time conditions, and bridges heterogeneous agent platforms. By enabling the direct communication of intentions, rationales, and decisions, it transforms collaborative driving from reactive perception-data sharing into proactive coordination, advancing safety, efficiency, and transparency in intelligent transportation systems. 

**Abstract (ZH)**: 多智能体协作驾驶通过集体感知和决策有望在交通安全性与效率方面取得改进。然而，现有的通信媒介——包括原始传感器数据、神经网络特征以及感知结果——在带宽效率、信息完整性及智能体互操作性方面存在局限。此外，传统方法在决策级融合方面基本忽略，忽略了协作驾驶的关键维度。本文认为，应对这些挑战需要从纯粹以感知为导向的数据交换转向使用自然语言进行明确的意图和推理通信。自然语言平衡了语义密度和通信带宽，能够灵活适应实时条件，并连接异构智能体平台。通过直接通信意图、推理和决策，它将协作驾驶从被动的感知数据共享转变为积极的协调，从而在智能交通系统中促进安全、效率和透明度。 

---
# Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation 

**Title (ZH)**: 面向局部性的并行解码以实现高效的自回归图像生成 

**Authors**: Zhuoyang Zhang, Luke J. Huang, Chengyue Wu, Shang Yang, Kelly Peng, Yao Lu, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.01957)  

**Abstract**: We present Locality-aware Parallel Decoding (LPD) to accelerate autoregressive image generation. Traditional autoregressive image generation relies on next-patch prediction, a memory-bound process that leads to high latency. Existing works have tried to parallelize next-patch prediction by shifting to multi-patch prediction to accelerate the process, but only achieved limited parallelization. To achieve high parallelization while maintaining generation quality, we introduce two key techniques: (1) Flexible Parallelized Autoregressive Modeling, a novel architecture that enables arbitrary generation ordering and degrees of parallelization. It uses learnable position query tokens to guide generation at target positions while ensuring mutual visibility among concurrently generated tokens for consistent parallel decoding. (2) Locality-aware Generation Ordering, a novel schedule that forms groups to minimize intra-group dependencies and maximize contextual support, enhancing generation quality. With these designs, we reduce the generation steps from 256 to 20 (256$\times$256 res.) and 1024 to 48 (512$\times$512 res.) without compromising quality on the ImageNet class-conditional generation, and achieving at least 3.4$\times$ lower latency than previous parallelized autoregressive models. 

**Abstract (ZH)**: 局部意识并行解码加速自回归图像生成 

---
# How Well Does GPT-4o Understand Vision? Evaluating Multimodal Foundation Models on Standard Computer Vision Tasks 

**Title (ZH)**: GPT-4o在视觉理解方面的能力如何？多模态基础模型在标准计算机视觉任务上的评估 

**Authors**: Rahul Ramachandran, Ali Garjani, Roman Bachmann, Andrei Atanov, Oğuzhan Fatih Kar, Amir Zamir  

**Link**: [PDF](https://arxiv.org/pdf/2507.01955)  

**Abstract**: Multimodal foundation models, such as GPT-4o, have recently made remarkable progress, but it is not clear where exactly these models stand in terms of understanding vision. In this paper, we benchmark the performance of popular multimodal foundation models (GPT-4o, o4-mini, Gemini 1.5 Pro and Gemini 2.0 Flash, Claude 3.5 Sonnet, Qwen2-VL, Llama 3.2) on standard computer vision tasks (semantic segmentation, object detection, image classification, depth and surface normal prediction) using established datasets (e.g., COCO, ImageNet and its variants, etc).
The main challenges to performing this are: 1) most models are trained to output text and cannot natively express versatile domains, such as segments or 3D geometry, and 2) many leading models are proprietary and accessible only at an API level, i.e., there is no weight access to adapt them. We address these challenges by translating standard vision tasks into equivalent text-promptable and API-compatible tasks via prompt chaining to create a standardized benchmarking framework.
We observe that 1) the models are not close to the state-of-the-art specialist models at any task. However, 2) they are respectable generalists; this is remarkable as they are presumably trained on primarily image-text-based tasks. 3) They perform semantic tasks notably better than geometric ones. 4) While the prompt-chaining techniques affect performance, better models exhibit less sensitivity to prompt variations. 5) GPT-4o performs the best among non-reasoning models, securing the top position in 4 out of 6 tasks, 6) reasoning models, e.g. o3, show improvements in geometric tasks, and 7) a preliminary analysis of models with native image generation, like the latest GPT-4o, shows they exhibit quirks like hallucinations and spatial misalignments. 

**Abstract (ZH)**: 多模态基础模型，如GPT-4o，在视觉理解方面取得了显著进展，但这些模型在理解视觉方面的具体水平尚不明确。本文在标准计算机视觉任务（语义分割、物体检测、图像分类、深度和表面法线预测）上，使用标准数据集（如COCO、ImageNet及其变体等）评估了流行多模态基础模型（GPT-4o、o4-mini、Gemini 1.5 Pro、Gemini 2.0 Flash、Claude 3.5 Sonnet、Qwen2-VL、Llama 3.2）的表现。 

---
# Are Vision Transformer Representations Semantically Meaningful? A Case Study in Medical Imaging 

**Title (ZH)**: 视觉变换器表示具有语义意义吗？以医学成像为例 

**Authors**: Montasir Shams, Chashi Mahiul Islam, Shaeke Salman, Phat Tran, Xiuwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01788)  

**Abstract**: Vision transformers (ViTs) have rapidly gained prominence in medical imaging tasks such as disease classification, segmentation, and detection due to their superior accuracy compared to conventional deep learning models. However, due to their size and complex interactions via the self-attention mechanism, they are not well understood. In particular, it is unclear whether the representations produced by such models are semantically meaningful. In this paper, using a projected gradient-based algorithm, we show that their representations are not semantically meaningful and they are inherently vulnerable to small changes. Images with imperceptible differences can have very different representations; on the other hand, images that should belong to different semantic classes can have nearly identical representations. Such vulnerability can lead to unreliable classification results; for example, unnoticeable changes cause the classification accuracy to be reduced by over 60\%. %. To the best of our knowledge, this is the first work to systematically demonstrate this fundamental lack of semantic meaningfulness in ViT representations for medical image classification, revealing a critical challenge for their deployment in safety-critical systems. 

**Abstract (ZH)**: Vision变压器（ViTs）在疾病分类、分割和检测等医学影像任务中由于其优越的准确率已迅速获得了重要地位，但由于其规模庞大和通过自注意力机制实现的复杂交互，它们并不容易理解。特别是，目前尚不清楚此类模型生成的表示是否具有语义意义。在本文中，我们使用投影梯度基算法展示了它们的表示并不是语义上有意义的，并且本质上对细微变化非常敏感。不可感知差异的图像可以具有非常不同的表示；另一方面，应该属于不同语义类别的图像可以具有几乎相同的表示。这种脆弱性可能导致分类结果不可靠；例如，细微变化会导致分类准确率下降超过60%。据我们所知，这是首次系统地证明ViT表示在医学图像分类中的根本缺乏语义意义的工作，揭示了其在安全关键系统部署中的一个关键挑战。 

---
# Autoregressive Image Generation with Linear Complexity: A Spatial-Aware Decay Perspective 

**Title (ZH)**: 具有线性复杂度的自回归图像生成：一种空间aware衰减视角 

**Authors**: Yuxin Mao, Zhen Qin, Jinxing Zhou, Hui Deng, Xuyang Shen, Bin Fan, Jing Zhang, Yiran Zhong, Yuchao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2507.01652)  

**Abstract**: Autoregressive (AR) models have garnered significant attention in image generation for their ability to effectively capture both local and global structures within visual data. However, prevalent AR models predominantly rely on the transformer architectures, which are beset by quadratic computational complexity concerning input sequence length and substantial memory overhead due to the necessity of maintaining key-value caches. Although linear attention mechanisms have successfully reduced this burden in language models, our initial experiments reveal that they significantly degrade image generation quality because of their inability to capture critical long-range dependencies in visual data. We propose Linear Attention with Spatial-Aware Decay (LASAD), a novel attention mechanism that explicitly preserves genuine 2D spatial relationships within the flattened image sequences by computing position-dependent decay factors based on true 2D spatial location rather than 1D sequence positions. Based on this mechanism, we present LASADGen, an autoregressive image generator that enables selective attention to relevant spatial contexts with linear complexity. Experiments on ImageNet show LASADGen achieves state-of-the-art image generation performance and computational efficiency, bridging the gap between linear attention's efficiency and spatial understanding needed for high-quality generation. 

**Abstract (ZH)**: 基于空间意识衰减的线性注意力机制在图像生成中的应用：LASADGen 

---
# Depth Anything at Any Condition 

**Title (ZH)**: 任意条件下的深度估计 

**Authors**: Boyuan Sun, Modi Jin, Bowen Yin, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2507.01634)  

**Abstract**: We present Depth Anything at Any Condition (DepthAnything-AC), a foundation monocular depth estimation (MDE) model capable of handling diverse environmental conditions. Previous foundation MDE models achieve impressive performance across general scenes but not perform well in complex open-world environments that involve challenging conditions, such as illumination variations, adverse weather, and sensor-induced distortions. To overcome the challenges of data scarcity and the inability of generating high-quality pseudo-labels from corrupted images, we propose an unsupervised consistency regularization finetuning paradigm that requires only a relatively small amount of unlabeled data. Furthermore, we propose the Spatial Distance Constraint to explicitly enforce the model to learn patch-level relative relationships, resulting in clearer semantic boundaries and more accurate details. Experimental results demonstrate the zero-shot capabilities of DepthAnything-AC across diverse benchmarks, including real-world adverse weather benchmarks, synthetic corruption benchmarks, and general benchmarks.
Project Page: this https URL
Code: this https URL 

**Abstract (ZH)**: Depth Anything at Any Condition (DepthAnything-AC): 一种能够在各种条件下的单目深度估计基础模型 

---
# Tile and Slide : A New Framework for Scaling NeRF from Local to Global 3D Earth Observation 

**Title (ZH)**: Tile and Slide：一种从局部到全局3D地球观测扩展NeRF的新框架 

**Authors**: Camille Billouard, Dawa Derksen, Alexandre Constantin, Bruno Vallet  

**Link**: [PDF](https://arxiv.org/pdf/2507.01631)  

**Abstract**: Neural Radiance Fields (NeRF) have recently emerged as a paradigm for 3D reconstruction from multiview satellite imagery. However, state-of-the-art NeRF methods are typically constrained to small scenes due to the memory footprint during training, which we study in this paper. Previous work on large-scale NeRFs palliate this by dividing the scene into NeRFs. This paper introduces Snake-NeRF, a framework that scales to large scenes. Our out-of-core method eliminates the need to load all images and networks simultaneously, and operates on a single device. We achieve this by dividing the region of interest into NeRFs that 3D tile without overlap. Importantly, we crop the images with overlap to ensure each NeRFs is trained with all the necessary pixels. We introduce a novel $2\times 2$ 3D tile progression strategy and segmented sampler, which together prevent 3D reconstruction errors along the tile edges. Our experiments conclude that large satellite images can effectively be processed with linear time complexity, on a single GPU, and without compromise in quality. 

**Abstract (ZH)**: 基于神经辐射场的大型场景三维重建：Snake-NeRF 

---
# Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss 

**Title (ZH)**: 区域联合损失导向的提示引导与人类proximal感知的HOT预测 

**Authors**: Yuxiao Wang, Yu Lei, Zhenao Wei, Weiying Xue, Xinyu Jiang, Nan Zhuang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01630)  

**Abstract**: The task of Human-Object conTact (HOT) detection involves identifying the specific areas of the human body that are touching objects. Nevertheless, current models are restricted to just one type of image, often leading to too much segmentation in areas with little interaction, and struggling to maintain category consistency within specific regions. To tackle this issue, a HOT framework, termed \textbf{P3HOT}, is proposed, which blends \textbf{P}rompt guidance and human \textbf{P}roximal \textbf{P}erception. To begin with, we utilize a semantic-driven prompt mechanism to direct the network's attention towards the relevant regions based on the correlation between image and text. Then a human proximal perception mechanism is employed to dynamically perceive key depth range around the human, using learnable parameters to effectively eliminate regions where interactions are not expected. Calculating depth resolves the uncertainty of the overlap between humans and objects in a 2D perspective, providing a quasi-3D viewpoint. Moreover, a Regional Joint Loss (RJLoss) has been created as a new loss to inhibit abnormal categories in the same area. A new evaluation metric called ``AD-Acc.'' is introduced to address the shortcomings of existing methods in addressing negative samples. Comprehensive experimental results demonstrate that our approach achieves state-of-the-art performance in four metrics across two benchmark datasets. Specifically, our model achieves an improvement of \textbf{0.7}$\uparrow$, \textbf{2.0}$\uparrow$, \textbf{1.6}$\uparrow$, and \textbf{11.0}$\uparrow$ in SC-Acc., mIoU, wIoU, and AD-Acc. metrics, respectively, on the HOT-Annotated dataset. Code is available at this https URL. 

**Abstract (ZH)**: Human-Object Contact (HOT) 检测中的 P3HOT 框架：结合提示引导和人体临近感知 

---
# Integrating Traditional and Deep Learning Methods to Detect Tree Crowns in Satellite Images 

**Title (ZH)**: 融合传统方法和深度学习方法检测卫星图像中的树冠 

**Authors**: Ozan Durgut, Beril Kallfelz-Sirmacek, Cem Unsalan  

**Link**: [PDF](https://arxiv.org/pdf/2507.01502)  

**Abstract**: Global warming, loss of biodiversity, and air pollution are among the most significant problems facing Earth. One of the primary challenges in addressing these issues is the lack of monitoring forests to protect them. To tackle this problem, it is important to leverage remote sensing and computer vision methods to automate monitoring applications. Hence, automatic tree crown detection algorithms emerged based on traditional and deep learning methods. In this study, we first introduce two different tree crown detection methods based on these approaches. Then, we form a novel rule-based approach that integrates these two methods to enhance robustness and accuracy of tree crown detection results. While traditional methods are employed for feature extraction and segmentation of forested areas, deep learning methods are used to detect tree crowns in our method. With the proposed rule-based approach, we post-process these results, aiming to increase the number of detected tree crowns through neighboring trees and localized operations. We compare the obtained results with the proposed method in terms of the number of detected tree crowns and report the advantages, disadvantages, and areas for improvement of the obtained outcomes. 

**Abstract (ZH)**: 全球变暖、生物多样性的丧失和空气污染是地球面临的最重大问题之一。在应对这些问题时，监测森林以保护它们是一个主要挑战。因此，需要利用遥感和计算机视觉方法来自动化监测应用，从而基于传统和深度学习方法提出了自动树冠检测算法。在本研究中，我们首先介绍了基于这两种方法的两种不同的树冠检测方法，然后提出了一种新的基于规则的方法，将这两种方法结合起来以增强树冠检测结果的可靠性和准确性。在我们的方法中，传统的技术用于提取森林区域的特征和分割，而深度学习技术则用于检测树冠。通过提出的基于规则的方法，我们对这些结果进行后处理，旨在通过邻近树木和局部操作增加检测到的树冠数量。我们从检测到的树冠数量的角度比较了提出的算法与其他方法的结果，并报告了所得结果的优点、缺点和改进空间。 

---
# Crop Pest Classification Using Deep Learning Techniques: A Review 

**Title (ZH)**: 基于深度学习技术的农作物害虫分类：一项综述 

**Authors**: Muhammad Hassam Ejaz, Muhammad Bilal, Usman Habib  

**Link**: [PDF](https://arxiv.org/pdf/2507.01494)  

**Abstract**: Insect pests continue to bring a serious threat to crop yields around the world, and traditional methods for monitoring them are often slow, manual, and difficult to scale. In recent years, deep learning has emerged as a powerful solution, with techniques like convolutional neural networks (CNNs), vision transformers (ViTs), and hybrid models gaining popularity for automating pest detection. This review looks at 37 carefully selected studies published between 2018 and 2025, all focused on AI-based pest classification. The selected research is organized by crop type, pest species, model architecture, dataset usage, and key technical challenges. The early studies relied heavily on CNNs but latest work is shifting toward hybrid and transformer-based models that deliver higher accuracy and better contextual understanding. Still, challenges like imbalanced datasets, difficulty in detecting small pests, limited generalizability, and deployment on edge devices remain significant hurdles. Overall, this review offers a structured overview of the field, highlights useful datasets, and outlines the key challenges and future directions for AI-based pest monitoring systems. 

**Abstract (ZH)**: 昆虫害虫继续对全球作物产量构成严重威胁，传统的监测方法往往速度慢、手工操作且难以扩展。近年来，深度学习 emerges as a powerful solution，卷积神经网络（CNNs）、视觉变换器（ViTs）和混合模型等技术被广泛用于自动化害虫检测。这篇综述涵盖了2018年至2025年间发表的37篇精心挑选的研究，所有研究都集中在基于AI的害虫分类上。所选研究按照作物类型、害虫种类、模型架构、数据集使用情况以及关键技术挑战进行组织。早期的研究主要依赖于CNNs，但最新的工作转向了混合模型和基于变换器的模型，这些模型能提供更高的准确性和更好的上下文理解。然而，如数据集不平衡、难以检测小型害虫、泛化能力有限以及在边缘设备上的部署等挑战仍然是重要的障碍。总体而言，这篇综述提供了该领域的结构化概述，强调了有用的数据库集，并概述了基于AI的害虫监测系统的关键挑战和未来方向。 

---
# NOCTIS: Novel Object Cyclic Threshold based Instance Segmentation 

**Title (ZH)**: NOCTIS: 新颖对象循环阈值实例分割 

**Authors**: Max Gandyra, Alessandro Santonicola, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.01463)  

**Abstract**: Instance segmentation of novel objects instances in RGB images, given some example images for each object, is a well known problem in computer vision. Designing a model general enough to be employed, for all kinds of novel objects, without (re-) training, has proven to be a difficult task. To handle this, we propose a simple, yet powerful, framework, called: Novel Object Cyclic Threshold based Instance Segmentation (NOCTIS). This work stems from and improves upon previous ones like CNOS, SAM-6D and NIDS-Net; thus, it also leverages on recent vision foundation models, namely: Grounded-SAM 2 and DINOv2. It utilises Grounded-SAM 2 to obtain object proposals with precise bounding boxes and their corresponding segmentation masks; while DINOv2's zero-shot capabilities are employed to generate the image embeddings. The quality of those masks, together with their embeddings, is of vital importance to our approach; as the proposal-object matching is realized by determining an object matching score based on the similarity of the class embeddings and the average maximum similarity of the patch embeddings. Differently to SAM-6D, calculating the latter involves a prior patch filtering based on the distance between each patch and its corresponding cyclic/roundtrip patch in the image grid. Furthermore, the average confidence of the proposals' bounding box and mask is used as an additional weighting factor for the object matching score. We empirically show that NOCTIS, without further training/fine tuning, outperforms the best RGB and RGB-D methods on the seven core datasets of the BOP 2023 challenge for the "Model-based 2D segmentation of unseen objects" task. 

**Abstract (ZH)**: 基于循环阈值的新型物体实例分割（NOCTIS）：RGB图像中给定示例图像的新颖物体实例分割 

---
# DocShaDiffusion: Diffusion Model in Latent Space for Document Image Shadow Removal 

**Title (ZH)**: DocShaDiffusion：文档图像阴影去除的潜空间扩散模型 

**Authors**: Wenjie Liu, Bingshu Wang, Ze Wang, C.L. Philip Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.01422)  

**Abstract**: Document shadow removal is a crucial task in the field of document image enhancement. However, existing methods tend to remove shadows with constant color background and ignore color shadows. In this paper, we first design a diffusion model in latent space for document image shadow removal, called DocShaDiffusion. It translates shadow images from pixel space to latent space, enabling the model to more easily capture essential features. To address the issue of color shadows, we design a shadow soft-mask generation module (SSGM). It is able to produce accurate shadow mask and add noise into shadow regions specially. Guided by the shadow mask, a shadow mask-aware guided diffusion module (SMGDM) is proposed to remove shadows from document images by supervising the diffusion and denoising process. We also propose a shadow-robust perceptual feature loss to preserve details and structures in document images. Moreover, we develop a large-scale synthetic document color shadow removal dataset (SDCSRD). It simulates the distribution of realistic color shadows and provides powerful supports for the training of models. Experiments on three public datasets validate the proposed method's superiority over state-of-the-art. Our code and dataset will be publicly available. 

**Abstract (ZH)**: 文档阴影去除是文档图像增强领域的关键任务。然而，现有的方法倾向于移除具有恒定颜色背景的阴影，而忽视彩色阴影。本文首先在潜空间中设计了一种扩散模型，称为DocShaDiffusion，用于文档图像阴影去除，它将阴影图像从像素空间转换到潜空间，使模型更易于捕捉到重要的特征。为了解决彩色阴影的问题，我们设计了一个阴影软掩模生成模块（SSGM），能够生成准确的阴影掩模并在阴影区域添加噪声。在阴影掩模的引导下，我们提出了一种阴影掩模感知引导扩散模块（SMGDM），通过监督扩散和去噪过程从文档图像中移除阴影。此外，我们提出了阴影鲁棒感知特征损失，以保留文档图像中的细节和结构。此外，我们还开发了一个大规模合成文档彩色阴影去除数据集（SDCSRD），模拟了现实彩色阴影的分布，并为模型训练提供了强有力的支持。在三个公开数据集上的实验验证了所提出方法优于现有方法。我们的代码和数据集将公开提供。 

---
# Prompt Mechanisms in Medical Imaging: A Comprehensive Survey 

**Title (ZH)**: 医学影像中的提示机制：一项全面综述 

**Authors**: Hao Yang, Xinlong Liang, Zhang Li, Yue Sun, Zheyu Hu, Xinghe Xie, Behdad Dashtbozorg, Jincheng Huang, Shiwei Zhu, Luyi Han, Jiong Zhang, Shanshan Wang, Ritse Mann, Qifeng Yu, Tao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.01055)  

**Abstract**: Deep learning offers transformative potential in medical imaging, yet its clinical adoption is frequently hampered by challenges such as data scarcity, distribution shifts, and the need for robust task generalization. Prompt-based methodologies have emerged as a pivotal strategy to guide deep learning models, providing flexible, domain-specific adaptations that significantly enhance model performance and adaptability without extensive retraining. This systematic review critically examines the burgeoning landscape of prompt engineering in medical imaging. We dissect diverse prompt modalities, including textual instructions, visual prompts, and learnable embeddings, and analyze their integration for core tasks such as image generation, segmentation, and classification. Our synthesis reveals how these mechanisms improve task-specific outcomes by enhancing accuracy, robustness, and data efficiency and reducing reliance on manual feature engineering while fostering greater model interpretability by making the model's guidance explicit. Despite substantial advancements, we identify persistent challenges, particularly in prompt design optimization, data heterogeneity, and ensuring scalability for clinical deployment. Finally, this review outlines promising future trajectories, including advanced multimodal prompting and robust clinical integration, underscoring the critical role of prompt-driven AI in accelerating the revolution of diagnostics and personalized treatment planning in medicine. 

**Abstract (ZH)**: 深度学习在医学影像领域的应用具有变革性潜力，但在临床应用中常因数据稀缺、分布偏移以及任务泛化能力不足等问题受到制约。基于提示的方法已成为引导深度学习模型的关键策略，提供灵活的、特定领域的适应性，显著提升模型性能和适应性，同时减少重新训练的需求。本文系统性地审视了提示工程在医学影像领域的新兴景观。我们剖析了各种提示模态，包括文本指令、视觉提示和可学习嵌入，并分析了它们在核心任务（如图像生成、分割和分类）中的整合应用。我们的综合分析表明，这些机制通过提高准确性、稳健性和数据效率，减少了对人工特征工程的依赖，同时使模型的指导更加明确，从而增强模型的可解释性。尽管取得了显著进展，但提示设计优化、数据异质性和确保临床部署的可扩展性等持续挑战仍然存在。最后，本文概述了提示驱动AI的有希望的未来发展方向，包括先进的多模态提示和稳健的临床集成，强调了提示驱动AI在加速医学诊断和个性化治疗规划革命中的关键作用。 

---
