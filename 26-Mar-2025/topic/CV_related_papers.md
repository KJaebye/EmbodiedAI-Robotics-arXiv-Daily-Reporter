# Semi-SD: Semi-Supervised Metric Depth Estimation via Surrounding Cameras for Autonomous Driving 

**Title (ZH)**: 半监督SD：基于周围摄像头的半监督度量深度估计在自动驾驶中的应用 

**Authors**: Yusen Xie, Zhengmin Huang, Shaojie Shen, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.19713)  

**Abstract**: In this paper, we introduce Semi-SD, a novel metric depth estimation framework tailored for surrounding cameras equipment in autonomous driving. In this work, the input data consists of adjacent surrounding frames and camera parameters. We propose a unified spatial-temporal-semantic fusion module to construct the visual fused features. Cross-attention components for surrounding cameras and adjacent frames are utilized to focus on metric scale information refinement and temporal feature matching. Building on this, we propose a pose estimation framework using surrounding cameras, their corresponding estimated depths, and extrinsic parameters, which effectively address the scale ambiguity in multi-camera setups. Moreover, semantic world model and monocular depth estimation world model are integrated to supervised the depth estimation, which improve the quality of depth estimation. We evaluate our algorithm on DDAD and nuScenes datasets, and the results demonstrate that our method achieves state-of-the-art performance in terms of surrounding camera based depth estimation quality. The source code will be available on this https URL. 

**Abstract (ZH)**: 本文引入了Semi-SD，这是一种针对自动驾驶周围摄像头设备的新颖度量depth估计框架。本文的输入数据包括相邻的周围帧和摄像头参数。我们提出了一种统一的空间-时间-语义融合模块来构建视觉融合特征。利用周围摄像头和相邻帧的交叉注意力组件，专注于度量尺度信息的细化和时间特征匹配。在此基础上，我们提出了一种使用周围摄像头、它们对应的估计depth以及外参的姿态估计框架，有效解决了多摄像头配置中的尺度模糊问题。此外，语义世界模型和单目depth估计世界模型被集成以监督depth估计，从而提高depth估计的质量。我们在DDAD和nuScenes数据集上评估了该算法，结果表明，我们的方法在基于周围摄像头的depth估计质量方面达到了最先进的性能。源代码将在此处提供。 

---
# G-DexGrasp: Generalizable Dexterous Grasping Synthesis Via Part-Aware Prior Retrieval and Prior-Assisted Generation 

**Title (ZH)**: G-DexGrasp: 基于部件意识先验检索与先验辅助生成的可泛化灵巧抓取合成 

**Authors**: Juntao Jian, Xiuping Liu, Zixuan Chen, Manyi Li, Jian Liu, Ruizhen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19457)  

**Abstract**: Recent advances in dexterous grasping synthesis have demonstrated significant progress in producing reasonable and plausible grasps for many task purposes. But it remains challenging to generalize to unseen object categories and diverse task instructions. In this paper, we propose G-DexGrasp, a retrieval-augmented generation approach that can produce high-quality dexterous hand configurations for unseen object categories and language-based task instructions. The key is to retrieve generalizable grasping priors, including the fine-grained contact part and the affordance-related distribution of relevant grasping instances, for the following synthesis pipeline. Specifically, the fine-grained contact part and affordance act as generalizable guidance to infer reasonable grasping configurations for unseen objects with a generative model, while the relevant grasping distribution plays as regularization to guarantee the plausibility of synthesized grasps during the subsequent refinement optimization. Our comparison experiments validate the effectiveness of our key designs for generalization and demonstrate the remarkable performance against the existing approaches. Project page: this https URL 

**Abstract (ZH)**: 最近在灵巧抓取合成方面的重要进展已经在许多任务目标中展示了生成合理且可信抓取方式的显著进步。但将其泛化到未见物体类别和多样化的任务指令仍然具有挑战性。本文提出了一种检索增强生成方法G-DexGrasp，能够在未见物体类别和基于语言的任务指令下生成高质量的灵巧手部配置。关键在于检索可泛化的抓取先验，包括精细的接触部分和与相关抓取实例相关的功能性分布，用于后续合成管道。具体而言，精细的接触部分和功能性作为可泛化的指导，通过生成模型推断未见物体的合理抓取配置，而相关抓取分布则作为正则化手段，确保生成抓取的可信性。我们的比较实验验证了我们关键设计的有效性，并展示了其相对于现有方法的出色性能。项目页面：this https URL 

---
# MATT-GS: Masked Attention-based 3DGS for Robot Perception and Object Detection 

**Title (ZH)**: MATT-GS：基于掩码注意力的3DGS机器人感知与物体检测 

**Authors**: Jee Won Lee, Hansol Lim, SooYeun Yang, Jongseong Brad Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.19330)  

**Abstract**: This paper presents a novel masked attention-based 3D Gaussian Splatting (3DGS) approach to enhance robotic perception and object detection in industrial and smart factory environments. U2-Net is employed for background removal to isolate target objects from raw images, thereby minimizing clutter and ensuring that the model processes only relevant data. Additionally, a Sobel filter-based attention mechanism is integrated into the 3DGS framework to enhance fine details - capturing critical features such as screws, wires, and intricate textures essential for high-precision tasks. We validate our approach using quantitative metrics, including L1 loss, SSIM, PSNR, comparing the performance of the background-removed and attention-incorporated 3DGS model against the ground truth images and the original 3DGS training baseline. The results demonstrate significant improves in visual fidelity and detail preservation, highlighting the effectiveness of our method in enhancing robotic vision for object recognition and manipulation in complex industrial settings. 

**Abstract (ZH)**: 本文提出了一种新颖的掩码注意力机制下的3D高斯点云（3DGS）方法，以增强工业和智能工厂环境中的机器人感知和物体检测。U2-Net用于背景去除，以隔离目标物体，从而减少杂乱并确保模型仅处理相关信息。此外，基于Sobel滤波器的注意力机制被集成到3DGS框架中，以增强细节点的提取，捕获诸如螺钉、电线和复杂纹理等关键特征，这对于高精度任务至关重要。我们通过使用L1损失、SSIM和PSNR等定量指标来验证该方法，并将去背景处理和注意力机制结合的3DGS模型的性能与真实图像和原始3DGS训练基线进行比较。结果表明，该方法在视觉保真度和细节保留方面有显著改善，突显了其在复杂工业环境中增强机器人视觉以进行物体识别和操纵的有效性。 

---
# A Survey on Event-driven 3D Reconstruction: Development under Different Categories 

**Title (ZH)**: 事件驱动的3D重建综述：不同类别下的发展 

**Authors**: Chuanzhi Xu, Haoxian Zhou, Haodong Chen, Vera Chung, Qiang Qu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19753)  

**Abstract**: Event cameras have gained increasing attention for 3D reconstruction due to their high temporal resolution, low latency, and high dynamic range. They capture per-pixel brightness changes asynchronously, allowing accurate reconstruction under fast motion and challenging lighting conditions. In this survey, we provide a comprehensive review of event-driven 3D reconstruction methods, including stereo, monocular, and multimodal systems. We further categorize recent developments based on geometric, learning-based, and hybrid approaches. Emerging trends, such as neural radiance fields and 3D Gaussian splatting with event data, are also covered. The related works are structured chronologically to illustrate the innovations and progression within the field. To support future research, we also highlight key research gaps and future research directions in dataset, experiment, evaluation, event representation, etc. 

**Abstract (ZH)**: 事件相机由于其高时间分辨率、低延迟和高动态范围，逐渐在三维重建领域获得广泛关注。它们以异步方式捕获每个像素的亮度变化，使得在快速运动和具有挑战性的光照条件下进行精确重建成为可能。在本文综述中，我们提供了事件驱动三维重建方法的综述，包括立体视觉、单目视觉和多模态系统。我们进一步根据几何方法、学习方法和混合方法对近期发展进行了分类。此外，还涵盖了神经辐射场和带有事件数据的3D高斯Splatting等新兴趋势。相关工作的结构化编年史展示了该领域内的创新和发展。为支持未来研究，我们还指出了数据集、实验、评估、事件表示等方面的潜在研究空白和未来研究方向。 

---
# CamSAM2: Segment Anything Accurately in Camouflaged Videos 

**Title (ZH)**: CamSAM2: 在伪装视频中准确分割Anything 

**Authors**: Yuli Zhou, Guolei Sun, Yawei Li, Yuqian Fu, Luca Benini, Ender Konukoglu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19730)  

**Abstract**: Video camouflaged object segmentation (VCOS), aiming at segmenting camouflaged objects that seamlessly blend into their environment, is a fundamental vision task with various real-world applications. With the release of SAM2, video segmentation has witnessed significant progress. However, SAM2's capability of segmenting camouflaged videos is suboptimal, especially when given simple prompts such as point and box. To address the problem, we propose Camouflaged SAM2 (CamSAM2), which enhances SAM2's ability to handle camouflaged scenes without modifying SAM2's parameters. Specifically, we introduce a decamouflaged token to provide the flexibility of feature adjustment for VCOS. To make full use of fine-grained and high-resolution features from the current frame and previous frames, we propose implicit object-aware fusion (IOF) and explicit object-aware fusion (EOF) modules, respectively. Object prototype generation (OPG) is introduced to abstract and memorize object prototypes with informative details using high-quality features from previous frames. Extensive experiments are conducted to validate the effectiveness of our approach. While CamSAM2 only adds negligible learnable parameters to SAM2, it substantially outperforms SAM2 on three VCOS datasets, especially achieving 12.2 mDice gains with click prompt on MoCA-Mask and 19.6 mDice gains with mask prompt on SUN-SEG-Hard, with Hiera-T as the backbone. The code will be available at \href{this https URL}{this http URL}. 

**Abstract (ZH)**: 掩蔽视频对象分割 (VCOS): 针对无缝融合环境的掩蔽对象分割，是视觉领域的基础任务，具有多种实际应用。随着SAM2的发布，视频分割取得了显著进展。然而，SAM2在分割掩蔽视频方面的能力不尽如人意，特别是对于简单的提示如点和框。为了解决这一问题，我们提出了一种增强SAM2处理掩蔽场景能力的方法——Camouflaged SAM2 (CamSAM2)，该方法在不修改SAM2参数的情况下提升了其性能。具体而言，我们引入了一种去掩蔽标记以提供VCOS中特征调整的灵活性。为充分利用当前帧和先前帧的细粒度和高分辨率特征，我们提出了隐式对象感知融合 (IOF) 和显式对象感知融合 (EOF) 模块。引入了对象原型生成 (OPG)，通过利用先前帧中高质量特征来抽象并记忆具有信息细节的对象原型。进行了广泛的实验以验证我们方法的有效性。尽管CamSAM2仅为SAM2增加了极少的可学习参数，但在三种VCOS数据集上仍显著优于SAM2，特别是在使用点击提示（MoCA-Mask）时获得了12.2 mDice的提升，在使用掩码提示（SUN-SEG-Hard）时获得了19.6 mDice的提升，以Hiera-T作为骨干网络。代码将在 \href{this https URL}{this http URL} 公开。 

---
# Decoupled Dynamics Framework with Neural Fields for 3D Spatio-temporal Prediction of Vehicle Collisions 

**Title (ZH)**: 基于神经场的解耦动力学框架在三维时空预测车辆碰撞中的应用 

**Authors**: Sanghyuk Kim, Minsik Seo, Namwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19712)  

**Abstract**: This study proposes a neural framework that predicts 3D vehicle collision dynamics by independently modeling global rigid-body motion and local structural deformation. Unlike approaches directly predicting absolute displacement, this method explicitly separates the vehicle's overall translation and rotation from its structural deformation. Two specialized networks form the core of the framework: a quaternion-based Rigid Net for rigid motion and a coordinate-based Deformation Net for local deformation. By independently handling fundamentally distinct physical phenomena, the proposed architecture achieves accurate predictions without requiring separate supervision for each component. The model, trained on only 10% of available simulation data, significantly outperforms baseline models, including single multi-layer perceptron (MLP) and deep operator networks (DeepONet), with prediction errors reduced by up to 83%. Extensive validation demonstrates strong generalization to collision conditions outside the training range, accurately predicting responses even under severe impacts involving extreme velocities and large impact angles. Furthermore, the framework successfully reconstructs high-resolution deformation details from low-resolution inputs without increased computational effort. Consequently, the proposed approach provides an effective, computationally efficient method for rapid and reliable assessment of vehicle safety across complex collision scenarios, substantially reducing the required simulation data and time while preserving prediction fidelity. 

**Abstract (ZH)**: 本研究提出了一种神经框架，通过独立建模全局刚体运动和局部结构变形来预测3D车辆碰撞动力学。该方法不同于直接预测绝对位移的方法，明确地将车辆的整体平移和旋转与其结构变形区分开来。该框架的核心由两个专门网络构成：基于四元数的刚体网（Rigid Net）处理刚体运动，基于坐标的具体变形网（Deformation Net）处理局部变形。通过独立处理基本不同的物理现象，所提出的架构能够在不需要为每个组件提供单独监督的情况下实现准确的预测。该模型仅使用可用模拟数据的10%进行训练，比基线模型（包括单层多层感知机和深度算子网络）显著表现出更高的性能，预测误差降低高达83%。广泛的验证结果表明，该框架能够强烈泛化到训练范围外的碰撞条件，即使在极端速度和大撞击角度的严重碰撞中也能准确预测响应。此外，该框架能够从低分辨率输入重构高分辨率变形细节，而不增加计算成本。因此，所提出的方法为复杂碰撞场景中的车辆安全性快速可靠评估提供了一种有效且计算效率高的方法，显著减少了所需的模拟数据和时间，同时保持预测准确性。 

---
# OpenSDI: Spotting Diffusion-Generated Images in the Open World 

**Title (ZH)**: OpenSDI: 在开放世界中识别扩散生成的图像 

**Authors**: Yabin Wang, Zhiwu Huang, Xiaopeng Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.19653)  

**Abstract**: This paper identifies OpenSDI, a challenge for spotting diffusion-generated images in open-world settings. In response to this challenge, we define a new benchmark, the OpenSDI dataset (OpenSDID), which stands out from existing datasets due to its diverse use of large vision-language models that simulate open-world diffusion-based manipulations. Another outstanding feature of OpenSDID is its inclusion of both detection and localization tasks for images manipulated globally and locally by diffusion models. To address the OpenSDI challenge, we propose a Synergizing Pretrained Models (SPM) scheme to build up a mixture of foundation models. This approach exploits a collaboration mechanism with multiple pretrained foundation models to enhance generalization in the OpenSDI context, moving beyond traditional training by synergizing multiple pretrained models through prompting and attending strategies. Building on this scheme, we introduce MaskCLIP, an SPM-based model that aligns Contrastive Language-Image Pre-Training (CLIP) with Masked Autoencoder (MAE). Extensive evaluations on OpenSDID show that MaskCLIP significantly outperforms current state-of-the-art methods for the OpenSDI challenge, achieving remarkable relative improvements of 14.23% in IoU (14.11% in F1) and 2.05% in accuracy (2.38% in F1) compared to the second-best model in localization and detection tasks, respectively. Our dataset and code are available at this https URL. 

**Abstract (ZH)**: 本文识别了OpenSDI这一挑战，旨在发现开放世界设置中生成的图像。为应对这一挑战，我们定义了一个新的基准，即OpenSDI数据集（OpenSDID），该数据集因广泛采用了模拟开放世界生成性操纵的大规模视觉-语言模型而脱颖而出。OpenSDID的另一突出特点是包含了由扩散模型全局和局部操纵的图像的检测和定位任务。为应对OpenSDI挑战，我们提出了一种综合预训练模型（SPM）方案，构建基于多种基础模型的混合模型。该方法通过预训练模型之间的协作机制，促进了OpenSDI情境下的泛化能力，超越了传统的单一预训练模型训练方式，通过提示和注意策略实现了多种预训练模型的协同工作。在此基础上，我们引入了MaskCLIP模型，这是一种基于SPM的模型，将对比语言-图像预训练（CLIP）与掩码自编码器（MAE）相结合。在OpenSDID上的广泛评估表明，MaskCLIP在局部和全局操作任务中显著优于当前最先进的方法，分别在IoU（F1为14.11%）和精度（F1为2.38%）上实现了14.23%和2.05%的相对改进。我们的数据集和代码可在以下网址获取。 

---
# Analyzable Chain-of-Musical-Thought Prompting for High-Fidelity Music Generation 

**Title (ZH)**: 可分析的音乐思维链提示生成高保真音乐 

**Authors**: Max W. Y. Lam, Yijin Xing, Weiya You, Jingcheng Wu, Zongyu Yin, Fuqiang Jiang, Hangyu Liu, Feng Liu, Xingda Li, Wei-Tsung Lu, Hanyu Chen, Tong Feng, Tianwei Zhao, Chien-Hung Liu, Xuchen Song, Yang Li, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.19611)  

**Abstract**: Autoregressive (AR) models have demonstrated impressive capabilities in generating high-fidelity music. However, the conventional next-token prediction paradigm in AR models does not align with the human creative process in music composition, potentially compromising the musicality of generated samples. To overcome this limitation, we introduce MusiCoT, a novel chain-of-thought (CoT) prompting technique tailored for music generation. MusiCoT empowers the AR model to first outline an overall music structure before generating audio tokens, thereby enhancing the coherence and creativity of the resulting compositions. By leveraging the contrastive language-audio pretraining (CLAP) model, we establish a chain of "musical thoughts", making MusiCoT scalable and independent of human-labeled data, in contrast to conventional CoT methods. Moreover, MusiCoT allows for in-depth analysis of music structure, such as instrumental arrangements, and supports music referencing -- accepting variable-length audio inputs as optional style references. This innovative approach effectively addresses copying issues, positioning MusiCoT as a vital practical method for music prompting. Our experimental results indicate that MusiCoT consistently achieves superior performance across both objective and subjective metrics, producing music quality that rivals state-of-the-art generation models.
Our samples are available at this https URL. 

**Abstract (ZH)**: 自回归（AR）模型在生成高保真音乐方面展现了出色的能力。然而，AR模型中传统的下一个token预测范式与音乐创作中的人类创意过程不一致，这可能会损害生成样本的音乐性。为克服这一限制，我们提出了MusiCoT，这是一种适用于音乐生成的新型链式思维（CoT）提示技术。MusiCoT使AR模型先概述整体音乐结构，再生成音频token，从而增强生成作品的一致性和创造性。通过利用对比语言-音频预训练（CLAP）模型，我们建立了一条“音乐思维链”，使其具有可扩展性和独立性，不同于传统的CoT方法。此外，MusiCoT支持对音乐结构的深入分析，如乐器编配，并支持音乐参考——可接受变长音频输入作为可选的风格参考。这一创新方法有效解决了复制问题，定位MusiCoT为音乐提示的重要实用方法。实验结果表明，MusiCoT在客观和主观指标上均表现出色，生成的音乐质量与最先进的生成模型不相上下。我们的样本可在此处获得。 

---
# DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image 

**Title (ZH)**: DeClotH：单张图像中可分解的3D衣物和人体重建 

**Authors**: Hyeongjin Nam, Donghwan Kim, Jeongtaek Oh, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.19373)  

**Abstract**: Most existing methods of 3D clothed human reconstruction from a single image treat the clothed human as a single object without distinguishing between cloth and human body. In this regard, we present DeClotH, which separately reconstructs 3D cloth and human body from a single image. This task remains largely unexplored due to the extreme occlusion between cloth and the human body, making it challenging to infer accurate geometries and textures. Moreover, while recent 3D human reconstruction methods have achieved impressive results using text-to-image diffusion models, directly applying such an approach to this problem often leads to incorrect guidance, particularly in reconstructing 3D cloth. To address these challenges, we propose two core designs in our framework. First, to alleviate the occlusion issue, we leverage 3D template models of cloth and human body as regularizations, which provide strong geometric priors to prevent erroneous reconstruction by the occlusion. Second, we introduce a cloth diffusion model specifically designed to provide contextual information about cloth appearance, thereby enhancing the reconstruction of 3D cloth. Qualitative and quantitative experiments demonstrate that our proposed approach is highly effective in reconstructing both 3D cloth and the human body. More qualitative results are provided at this https URL. 

**Abstract (ZH)**: 单图三维着装人体重构中区分布与人体的独立重建方法 

---
# Wavelet-based Global-Local Interaction Network with Cross-Attention for Multi-View Diabetic Retinopathy Detection 

**Title (ZH)**: 基于小波的全局-局部交互网络与跨注意力机制的多视角糖尿病视网膜病变检测 

**Authors**: Yongting Hu, Yuxin Lin, Chengliang Liu, Xiaoling Luo, Xiaoyan Dou, Qihao Xu, Yong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19329)  

**Abstract**: Multi-view diabetic retinopathy (DR) detection has recently emerged as a promising method to address the issue of incomplete lesions faced by single-view DR. However, it is still challenging due to the variable sizes and scattered locations of lesions. Furthermore, existing multi-view DR methods typically merge multiple views without considering the correlations and redundancies of lesion information across them. Therefore, we propose a novel method to overcome the challenges of difficult lesion information learning and inadequate multi-view fusion. Specifically, we introduce a two-branch network to obtain both local lesion features and their global dependencies. The high-frequency component of the wavelet transform is used to exploit lesion edge information, which is then enhanced by global semantic to facilitate difficult lesion learning. Additionally, we present a cross-view fusion module to improve multi-view fusion and reduce redundancy. Experimental results on large public datasets demonstrate the effectiveness of our method. The code is open sourced on this https URL. 

**Abstract (ZH)**: 多视图糖尿病视网膜病变检测 recently emerged as a promising method to address the issue of incomplete lesions faced by single-view diabetic retinopathy. However, it is still challenging due to the variable sizes and scattered locations of lesions. Furthermore, existing multi-view diabetic retinopathy methods typically merge multiple views without considering the correlations and redundancies of lesion information across them. Therefore, we propose a novel method to overcome the challenges of difficult lesion information learning and inadequate multi-view fusion. Specifically, we introduce a two-branch network to obtain both local lesion features and their global dependencies. The high-frequency component of the wavelet transform is used to exploit lesion edge information, which is then enhanced by global semantic to facilitate difficult lesion learning. Additionally, we present a cross-view fusion module to improve multi-view fusion and reduce redundancy. Experimental results on large public datasets demonstrate the effectiveness of our method. The code is open sourced at this https URL. 

---
# Context-Aware Semantic Segmentation: Enhancing Pixel-Level Understanding with Large Language Models for Advanced Vision Applications 

**Title (ZH)**: 基于上下文的语义分割：通过大规模语言模型提升像素级理解以实现先进视觉应用 

**Authors**: Ben Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2503.19276)  

**Abstract**: Semantic segmentation has made significant strides in pixel-level image understanding, yet it remains limited in capturing contextual and semantic relationships between objects. Current models, such as CNN and Transformer-based architectures, excel at identifying pixel-level features but fail to distinguish semantically similar objects (e.g., "doctor" vs. "nurse" in a hospital scene) or understand complex contextual scenarios (e.g., differentiating a running child from a regular pedestrian in autonomous driving). To address these limitations, we proposed a novel Context-Aware Semantic Segmentation framework that integrates Large Language Models (LLMs) with state-of-the-art vision backbones. Our hybrid model leverages the Swin Transformer for robust visual feature extraction and GPT-4 for enriching semantic understanding through text embeddings. A Cross-Attention Mechanism is introduced to align vision and language features, enabling the model to reason about context more effectively. Additionally, Graph Neural Networks (GNNs) are employed to model object relationships within the scene, capturing dependencies that are overlooked by traditional models. Experimental results on benchmark datasets (e.g., COCO, Cityscapes) demonstrate that our approach outperforms the existing methods in both pixel-level accuracy (mIoU) and contextual understanding (mAP). This work bridges the gap between vision and language, paving the path for more intelligent and context-aware vision systems in applications including autonomous driving, medical imaging, and robotics. 

**Abstract (ZH)**: 基于上下文的语义分割在像素级图像理解中取得了显著进展，但仍有限制于捕捉物体间的上下文和语义关系。现有的模型，如CNN和基于Transformer的架构，擅长识别像素级特征，但在区分语义相似的对象（例如，医院场景中的“医生”和“护士”）或理解复杂的上下文场景（例如，在自动驾驶中区分奔跑的儿童和普通行人的方面）存在局限性。为了解决这些限制，我们提出了一种新的基于上下文的语义分割框架，该框架将大型语言模型（LLMs）与最先进的视觉骨干网络相结合。我们的混合模型利用Swin Transformer进行稳健的视觉特征提取，并通过文本嵌入丰富语义理解。引入了跨注意力机制以对齐视觉和语言特征，使模型能够更有效地进行上下文推理。此外，使用图神经网络（GNNs）建模场景中的对象关系，捕获传统模型忽略的依赖关系。在基准数据集（如COCO、Cityscapes）上的实验结果表明，我们的方法在像素级准确率（mIoU）和上下文理解（mAP）方面优于现有方法。这项工作填补了视觉和语言之间的鸿沟，为自动驾驶、医学成像和机器人等应用中的更智能和上下文感知视觉系统铺平了道路。 

---
# Face Spoofing Detection using Deep Learning 

**Title (ZH)**: 使用深度学习的面部 spoofing 检测 

**Authors**: Najeebullah, Maaz Salman, Zar Nawab Khan Swati  

**Link**: [PDF](https://arxiv.org/pdf/2503.19223)  

**Abstract**: Digital image spoofing has emerged as a significant security threat in biometric authentication systems, particularly those relying on facial recognition. This study evaluates the performance of three vision based models, MobileNetV2, ResNET50, and Vision Transformer, ViT, for spoof detection in image classification, utilizing a dataset of 150,986 images divided into training , 140,002, testing, 10,984, and validation ,39,574, sets. Spoof detection is critical for enhancing the security of image recognition systems, and this research compares the models effectiveness through accuracy, precision, recall, and F1 score metrics. Results reveal that MobileNetV2 outperforms other architectures on the test dataset, achieving an accuracy of 91.59%, precision of 91.72%, recall of 91.59%, and F1 score of 91.58%, compared to ViT 86.54%, 88.28%, 86.54%, and 86.39%, respectively. On the validation dataset, MobileNetV2, and ViT excel, with MobileNetV2 slightly ahead at 97.17% accuracy versus ViT 96.36%. MobileNetV2 demonstrates faster convergence during training and superior generalization to unseen data, despite both models showing signs of overfitting. These findings highlight MobileNetV2 balanced performance and robustness, making it the preferred choice for spoof detection applications where reliability on new data is essential. The study underscores the importance of model selection in security sensitive contexts and suggests MobileNetV2 as a practical solution for real world deployment. 

**Abstract (ZH)**: 基于数字图像欺骗的生物特征认证系统中视觉模型的欺骗检测性能研究 

---
# PSO-UNet: Particle Swarm-Optimized U-Net Framework for Precise Multimodal Brain Tumor Segmentation 

**Title (ZH)**: PSO-UNet：粒子群优化的U-Net框架用于精确的多模态脑肿瘤分割 

**Authors**: Shoffan Saifullah, Rafał Dreżewski  

**Link**: [PDF](https://arxiv.org/pdf/2503.19152)  

**Abstract**: Medical image segmentation, particularly for brain tumor analysis, demands precise and computationally efficient models due to the complexity of multimodal MRI datasets and diverse tumor morphologies. This study introduces PSO-UNet, which integrates Particle Swarm Optimization (PSO) with the U-Net architecture for dynamic hyperparameter optimization. Unlike traditional manual tuning or alternative optimization approaches, PSO effectively navigates complex hyperparameter search spaces, explicitly optimizing the number of filters, kernel size, and learning rate. PSO-UNet substantially enhances segmentation performance, achieving Dice Similarity Coefficients (DSC) of 0.9578 and 0.9523 and Intersection over Union (IoU) scores of 0.9194 and 0.9097 on the BraTS 2021 and Figshare datasets, respectively. Moreover, the method reduces computational complexity significantly, utilizing only 7.8 million parameters and executing in approximately 906 seconds, markedly faster than comparable U-Net-based frameworks. These outcomes underscore PSO-UNet's robust generalization capabilities across diverse MRI modalities and tumor classifications, emphasizing its clinical potential and clear advantages over conventional hyperparameter tuning methods. Future research will explore hybrid optimization strategies and validate the framework against other bio-inspired algorithms to enhance its robustness and scalability. 

**Abstract (ZH)**: 基于粒子群优化的PSO-UNet在脑肿瘤医学图像分割中的应用：一种精确且计算高效的动态超参数优化方法 

---
# Anomaly Detection Using Computer Vision: A Comparative Analysis of Class Distinction and Performance Metrics 

**Title (ZH)**: 利用计算机视觉进行异常检测：类区分与性能指标的比较分析 

**Authors**: Md. Barkat Ullah Tusher, Shartaz Khan Akash, Amirul Islam Showmik  

**Link**: [PDF](https://arxiv.org/pdf/2503.19100)  

**Abstract**: This paper showcases an experimental study on anomaly detection using computer vision. The study focuses on class distinction and performance evaluation, combining OpenCV with deep learning techniques while employing a TensorFlow-based convolutional neural network for real-time face recognition and classification. The system effectively distinguishes among three classes: authorized personnel (admin), intruders, and non-human entities. A MobileNetV2-based deep learning model is utilized to optimize real-time performance, ensuring high computational efficiency without compromising accuracy. Extensive dataset preprocessing, including image augmentation and normalization, enhances the models generalization capabilities. Our analysis demonstrates classification accuracies of 90.20% for admin, 98.60% for intruders, and 75.80% for non-human detection, while maintaining an average processing rate of 30 frames per second. The study leverages transfer learning, batch normalization, and Adam optimization to achieve stable and robust learning, and a comparative analysis of class differentiation strategies highlights the impact of feature extraction techniques and training methodologies. The results indicate that advanced feature selection and data augmentation significantly enhance detection performance, particularly in distinguishing human from non-human scenes. As an experimental study, this research provides critical insights into optimizing deep learning-based surveillance systems for high-security environments and improving the accuracy and efficiency of real-time anomaly detection. 

**Abstract (ZH)**: 基于计算机视觉的异常检测实验研究：结合OpenCV和深度学习的实时面部识别与分类 

---
# HingeRLC-GAN: Combating Mode Collapse with Hinge Loss and RLC Regularization 

**Title (ZH)**: HingeRLC-GAN：基于Hinge损失和RLC正则化的模式崩溃对抗方法 

**Authors**: Osman Goni, Himadri Saha Arka, Mithun Halder, Mir Moynuddin Ahmed Shibly, Swakkhar Shatabda  

**Link**: [PDF](https://arxiv.org/pdf/2503.19074)  

**Abstract**: Recent advances in Generative Adversarial Networks (GANs) have demonstrated their capability for producing high-quality images. However, a significant challenge remains mode collapse, which occurs when the generator produces a limited number of data patterns that do not reflect the diversity of the training dataset. This study addresses this issue by proposing a number of architectural changes aimed at increasing the diversity and stability of GAN models. We start by improving the loss function with Wasserstein loss and Gradient Penalty to better capture the full range of data variations. We also investigate various network architectures and conclude that ResNet significantly contributes to increased diversity. Building on these findings, we introduce HingeRLC-GAN, a novel approach that combines RLC Regularization and the Hinge loss function. With a FID Score of 18 and a KID Score of 0.001, our approach outperforms existing methods by effectively balancing training stability and increased diversity. 

**Abstract (ZH)**: 最近生成式对抗网络（GANs）的进展展示了其生成高保真图像的能力。然而，模式崩溃这一重要挑战依然存在，当生成器只产生有限的数据模式且未能反映训练数据集的多样性时就会发生这种情况。本研究通过提出多种架构改进措施来应对这一问题，旨在提高GAN模型的多样性和稳定性。我们首先通过使用Wasserstein损失和梯度惩罚改进损失函数，以更好地捕捉数据变异性。我们还探讨了各种网络架构，并得出结论，ResNet显著提高了多样性。在此基础上，我们引入了HingeRLC-GAN，这是一种结合RLC正则化和Hinge损失的新方法。我们的方法在FID得分为18和KID得分为0.001的情况下，通过有效平衡训练稳定性和增加多样性，优于现有方法。 

---
# Computational Thinking with Computer Vision: Developing AI Competency in an Introductory Computer Science Course 

**Title (ZH)**: 基于计算机视觉的计算思维：在入门计算机科学课程中培养人工智能能力 

**Authors**: Tahiya Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.19006)  

**Abstract**: Developing competency in artificial intelligence is becoming increasingly crucial for computer science (CS) students at all levels of the CS curriculum. However, most previous research focuses on advanced CS courses, as traditional introductory courses provide limited opportunities to develop AI skills and knowledge. This paper introduces an introductory CS course where students learn computational thinking through computer vision, a sub-field of AI, as an application context. The course aims to achieve computational thinking outcomes alongside critical thinking outcomes that expose students to AI approaches and their societal implications. Through experiential activities such as individual projects and reading discussions, our course seeks to balance technical learning and critical thinking goals. Our evaluation, based on pre-and post-course surveys, shows an improved sense of belonging, self-efficacy, and AI ethics awareness among students. The results suggest that an AI-focused context can enhance participation and employability, student-selected projects support self-efficacy, and ethically grounded AI instruction can be effective for interdisciplinary audiences. Students' discussions on reading assignments demonstrated deep engagement with the complex challenges in today's AI landscape. Finally, we share insights on scaling such courses for larger cohorts and improving the learning experience for introductory CS students. 

**Abstract (ZH)**: 人工智能领域的技能培养正日益成为计算机科学（CS）学生跨所有CS课程层级的关键能力。然而，大多数先前的研究集中在高级CS课程上，因为传统的入门课程为培养AI技能和知识提供了有限的机会。本文介绍了一门入门级CS课程，在该课程中，学生通过计算机视觉——AI的一个子领域——学习计算思维，以此作为应用背景。该课程旨在通过介绍AI方法及其社会影响来实现计算思维和批判性思维的结果。通过个人项目和阅读讨论等体验性活动，我们的课程寻求在技术学习和批判性思维目标之间取得平衡。基于课前和课后的调查，我们的评估结果显示学生对归属感、自我效能感和AI伦理意识有所增强。研究表明，以AI为重点的背景可以增强参与度和就业能力，学生选择的项目可以支持自我效能感，基于伦理的教学可以有效地服务于跨学科听众。学生在阅读任务上的讨论表明，他们对当今AI领域的复杂挑战有深入的理解。最后，我们分享了扩展此类课程以容纳更大班级规模并改善入门级CS学生学习体验的见解。 

---
# DisentTalk: Cross-lingual Talking Face Generation via Semantic Disentangled Diffusion Model 

**Title (ZH)**: DisentTalk：语义解耦扩散模型驱动的跨语言表情生成 

**Authors**: Kangwei Liu, Junwu Liu, Yun Cao, Jinlin Guo, Xiaowei Yi  

**Link**: [PDF](https://arxiv.org/pdf/2503.19001)  

**Abstract**: Recent advances in talking face generation have significantly improved facial animation synthesis. However, existing approaches face fundamental limitations: 3DMM-based methods maintain temporal consistency but lack fine-grained regional control, while Stable Diffusion-based methods enable spatial manipulation but suffer from temporal inconsistencies. The integration of these approaches is hindered by incompatible control mechanisms and semantic entanglement of facial representations. This paper presents DisentTalk, introducing a data-driven semantic disentanglement framework that decomposes 3DMM expression parameters into meaningful subspaces for fine-grained facial control. Building upon this disentangled representation, we develop a hierarchical latent diffusion architecture that operates in 3DMM parameter space, integrating region-aware attention mechanisms to ensure both spatial precision and temporal coherence. To address the scarcity of high-quality Chinese training data, we introduce CHDTF, a Chinese high-definition talking face dataset. Extensive experiments show superior performance over existing methods across multiple metrics, including lip synchronization, expression quality, and temporal consistency. Project Page: this https URL. 

**Abstract (ZH)**: 最近在说话人脸生成方面的进展显著提高了面部动画合成的质量。然而，现有方法面临根本性的局限性：基于3DMM的方法保持了时间一致性，但在区域细节控制上不足，而基于Stable Diffusion的方法能够实现空间操控，但在时间一致性上存在问题。这些方法的整合受到不兼容控制机制和面部表示语义纠缠的阻碍。本文提出DisentTalk，介绍了一种数据驱动的语义解纠缠框架，将3DMM表情参数分解为有意义的子空间，以实现精细的面部控制。在此解纠缠表示的基础上，我们开发了一种分层的隐空间扩散架构，该架构在3DMM参数空间中运行，结合区域意识的注意力机制，确保时空一致性。为了解决高质量中文训练数据稀缺的问题，我们引入了CHDTF中文高分辨率说话人脸数据集。广泛实验表明，在多个指标上，包括唇同步、表情质量和时间一致性等方面，该方法优于现有方法。项目页面：这个 https URL。 

---
# Automated diagnosis of lung diseases using vision transformer: a comparative study on chest x-ray classification 

**Title (ZH)**: 基于视觉变换器的肺部疾病自动化诊断：胸部X光分类的比较研究 

**Authors**: Muhammad Ahmad, Sardar Usman, Ildar Batyrshin, Muhammad Muzammil, K. Sajid, M. Hasnain, Muhammad Jalal, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2503.18973)  

**Abstract**: Background: Lung disease is a significant health issue, particularly in children and elderly individuals. It often results from lung infections and is one of the leading causes of mortality in children. Globally, lung-related diseases claim many lives each year, making early and accurate diagnoses crucial. Radiographs are valuable tools for the diagnosis of such conditions. The most prevalent lung diseases, including pneumonia, asthma, allergies, chronic obstructive pulmonary disease (COPD), bronchitis, emphysema, and lung cancer, represent significant public health challenges. Early prediction of these conditions is critical, as it allows for the identification of risk factors and implementation of preventive measures to reduce the likelihood of disease onset
Methods: In this study, we utilized a dataset comprising 3,475 chest X-ray images sourced from from Mendeley Data provided by Talukder, M. A. (2023) [14], categorized into three classes: normal, lung opacity, and pneumonia. We applied five pre-trained deep learning models, including CNN, ResNet50, DenseNet, CheXNet, and U-Net, as well as two transfer learning algorithms such as Vision Transformer (ViT) and Shifted Window (Swin) to classify these images. This approach aims to address diagnostic issues in lung abnormalities by reducing reliance on human intervention through automated classification systems. Our analysis was conducted in both binary and multiclass settings. Results: In the binary classification, we focused on distinguishing between normal and viral pneumonia cases, whereas in the multi-class classification, all three classes (normal, lung opacity, and viral pneumonia) were included. Our proposed methodology (ViT) achieved remarkable performance, with accuracy rates of 99% for binary classification and 95.25% for multiclass classification. 

**Abstract (ZH)**: 背景：肺部疾病是重要的公共卫生问题，特别是在儿童和老年人中。这些疾病往往源于肺部感染，并且是儿童死亡的主要原因之一。在全球范围内，肺部相关疾病每年夺去许多生命，因此早期和准确的诊断至关重要。胸部X线检查是诊断这些病症的重要工具。包括肺炎、哮喘、过敏、慢性阻塞性肺疾病（COPD）、支气管炎、肺气肿和肺癌在内的最常见肺部疾病构成了重大的公共卫生挑战。对这些疾病的早期预测至关重要，因为它有助于识别风险因素并实施预防措施，以降低疾病发生的可能性。

方法：本研究采用了Talukder, M. A. (2023) [14] 通过Mendeley Data 提供的包含3,475张胸部X光图像的数据集，并将其分为三个类别：正常、肺实变和肺炎。我们应用了包括CNN、ResNet50、DenseNet、CheXNet和U-Net在内的五种预训练深度学习模型，以及包括Vision Transformer (ViT)和Shifted Window (Swin)在内的两种迁移学习算法，以对这些图像进行分类。该方法旨在通过自动化分类系统减少对人力判断的依赖，从而解决肺部异常的诊断问题。我们的分析在二分类和多分类两种场景下进行。

结果：在二分类中，我们集中于区分正常与病毒性肺炎病例；而在多分类中，包括了所有三个类别（正常、肺实变和病毒性肺炎）。我们提出的基于ViT的方法在二分类中的准确率为99%，在多分类中的准确率为95.25%。 

---
