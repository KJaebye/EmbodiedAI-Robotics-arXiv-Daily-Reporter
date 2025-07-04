# MISCGrasp: Leveraging Multiple Integrated Scales and Contrastive Learning for Enhanced Volumetric Grasping 

**Title (ZH)**: MISCGrasp: 利用多个集成尺度与对比学习提升体素抓取 

**Authors**: Qingyu Fan, Yinghao Cai, Chao Li, Chunting Jiao, Xudong Zheng, Tao Lu, Bin Liang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02672)  

**Abstract**: Robotic grasping faces challenges in adapting to objects with varying shapes and sizes. In this paper, we introduce MISCGrasp, a volumetric grasping method that integrates multi-scale feature extraction with contrastive feature enhancement for self-adaptive grasping. We propose a query-based interaction between high-level and low-level features through the Insight Transformer, while the Empower Transformer selectively attends to the highest-level features, which synergistically strikes a balance between focusing on fine geometric details and overall geometric structures. Furthermore, MISCGrasp utilizes multi-scale contrastive learning to exploit similarities among positive grasp samples, ensuring consistency across multi-scale features. Extensive experiments in both simulated and real-world environments demonstrate that MISCGrasp outperforms baseline and variant methods in tabletop decluttering tasks. More details are available at this https URL. 

**Abstract (ZH)**: 基于多尺度特征提取与对比特征增强的自适应抓取方法MISCGrasp 

---
# A Late Collaborative Perception Framework for 3D Multi-Object and Multi-Source Association and Fusion 

**Title (ZH)**: 一种用于三维多目标和多源关联与融合的晚期协作感知框架 

**Authors**: Maryem Fadili, Mohamed Anis Ghaoui, Louis Lecrosnier, Steve Pechberti, Redouane Khemmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.02430)  

**Abstract**: In autonomous driving, recent research has increasingly focused on collaborative perception based on deep learning to overcome the limitations of individual perception systems. Although these methods achieve high accuracy, they rely on high communication bandwidth and require unrestricted access to each agent's object detection model architecture and parameters. These constraints pose challenges real-world autonomous driving scenarios, where communication limitations and the need to safeguard proprietary models hinder practical implementation.  To address this issue, we introduce a novel late collaborative framework for 3D multi-source and multi-object fusion, which operates solely on shared 3D bounding box attributes-category, size, position, and orientation-without necessitating direct access to detection models.  Our framework establishes a new state-of-the-art in late fusion, achieving up to five times lower position error compared to existing methods. Additionally, it reduces scale error by a factor of 7.5 and orientation error by half, all while maintaining perfect 100% precision and recall when fusing detections from heterogeneous perception systems. These results highlight the effectiveness of our approach in addressing real-world collaborative perception challenges, setting a new benchmark for efficient and scalable multi-agent fusion. 

**Abstract (ZH)**: 自动驾驶中的自适应深度学习驱动的分阶段协作感知框架 

---
# cVLA: Towards Efficient Camera-Space VLAs 

**Title (ZH)**: cVLA:Towards Efficient Camera-Space Vector Lookup Arrays 

**Authors**: Max Argus, Jelena Bratulic, Houman Masnavi, Maxim Velikanov, Nick Heppert, Abhinav Valada, Thomas Brox  

**Link**: [PDF](https://arxiv.org/pdf/2507.02190)  

**Abstract**: Vision-Language-Action (VLA) models offer a compelling framework for tackling complex robotic manipulation tasks, but they are often expensive to train. In this paper, we propose a novel VLA approach that leverages the competitive performance of Vision Language Models (VLMs) on 2D images to directly infer robot end-effector poses in image frame coordinates. Unlike prior VLA models that output low-level controls, our model predicts trajectory waypoints, making it both more efficient to train and robot embodiment agnostic. Despite its lightweight design, our next-token prediction architecture effectively learns meaningful and executable robot trajectories. We further explore the underutilized potential of incorporating depth images, inference-time techniques such as decoding strategies, and demonstration-conditioned action generation. Our model is trained on a simulated dataset and exhibits strong sim-to-real transfer capabilities. We evaluate our approach using a combination of simulated and real data, demonstrating its effectiveness on a real robotic system. 

**Abstract (ZH)**: Vision-Language-Action (VLA) 模型为解决复杂的机器人操作任务提供了令人信服的框架，但往往训练成本高昂。本文提出了一种新颖的 VLA 方法，利用 Vision Language 模型（VLMs）在 2D 图像上的竞争力来直接推断机器人末端执行器在图像帧坐标系中的姿态。与先前 VLA 模型输出低级控制不同，我们的模型预测轨迹关键点，使其既更高效地训练，又与机器人实体无关。尽管设计轻量，但我们的时间下一个令牌预测架构有效地学习了有意义且可执行的机器人轨迹。我们还探索了结合深度图像、推理时技术如解码策略以及演示条件动作生成的未充分利用的潜力。我们的模型在模拟数据集上训练，并展现出强烈的模拟到现实的迁移能力。我们使用模拟和现实数据相结合进行评估，表明其在真实机器人系统上的有效性。 

---
# Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory 

**Title (ZH)**: 点3R：具有显式空间指针记忆的流式3D重建 

**Authors**: Yuqi Wu, Wenzhao Zheng, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02863)  

**Abstract**: Dense 3D scene reconstruction from an ordered sequence or unordered image collections is a critical step when bringing research in computer vision into practical scenarios. Following the paradigm introduced by DUSt3R, which unifies an image pair densely into a shared coordinate system, subsequent methods maintain an implicit memory to achieve dense 3D reconstruction from more images. However, such implicit memory is limited in capacity and may suffer from information loss of earlier frames. We propose Point3R, an online framework targeting dense streaming 3D reconstruction. To be specific, we maintain an explicit spatial pointer memory directly associated with the 3D structure of the current scene. Each pointer in this memory is assigned a specific 3D position and aggregates scene information nearby in the global coordinate system into a changing spatial feature. Information extracted from the latest frame interacts explicitly with this pointer memory, enabling dense integration of the current observation into the global coordinate system. We design a 3D hierarchical position embedding to promote this interaction and design a simple yet effective fusion mechanism to ensure that our pointer memory is uniform and efficient. Our method achieves competitive or state-of-the-art performance on various tasks with low training costs. Code is available at: this https URL. 

**Abstract (ZH)**: 从有序序列或无序图像集合中进行稠密三维场景重建是将计算机视觉研究引入实际场景的关键步骤。继DUSt3R提出的框架，该框架将图像对稠密地统一到共享坐标系中之后，后续方法通过维护隐式的记忆来实现更多图像的稠密三维重建。然而，这种隐式记忆在容量上有限，并且可能会遭受早期帧信息丢失的问题。我们提出Point3R，这是一个针对稠密流式三维重建的在线框架。具体而言，我们维护一个直接与当前场景三维结构相关的显式的空间指针记忆。该记忆中的每个指针都分配了一个特定的三维位置，并在全局坐标系中聚集附近的场景信息，形成一个变化的空间特征。最新帧中提取的信息与这个指针记忆进行显式的交互，从而使得当前观察能够被稠密地整合到全局坐标系中。我们设计了三维层次位置嵌入来促进这种交互，并设计了一个简单而有效的融合机制，以确保我们的指针记忆均匀且高效。我们的方法在各种任务上实现了有竞争力或最先进的性能，且训练成本较低。代码可从以下链接获得：this https URL。 

---
# LiteReality: Graphics-Ready 3D Scene Reconstruction from RGB-D Scans 

**Title (ZH)**: LiteReality：从RGB-D 扫描中实现 Ready-for-图形的 3D 场景重建 

**Authors**: Zhening Huang, Xiaoyang Wu, Fangcheng Zhong, Hengshuang Zhao, Matthias Nießner, Joan Lasenby  

**Link**: [PDF](https://arxiv.org/pdf/2507.02861)  

**Abstract**: We propose LiteReality, a novel pipeline that converts RGB-D scans of indoor environments into compact, realistic, and interactive 3D virtual replicas. LiteReality not only reconstructs scenes that visually resemble reality but also supports key features essential for graphics pipelines -- such as object individuality, articulation, high-quality physically based rendering materials, and physically based interaction. At its core, LiteReality first performs scene understanding and parses the results into a coherent 3D layout and objects with the help of a structured scene graph. It then reconstructs the scene by retrieving the most visually similar 3D artist-crafted models from a curated asset database. Next, the Material Painting module enhances realism by recovering high-quality, spatially varying materials. Finally, the reconstructed scene is integrated into a simulation engine with basic physical properties to enable interactive behavior. The resulting scenes are compact, editable, and fully compatible with standard graphics pipelines, making them suitable for applications in AR/VR, gaming, robotics, and digital twins. In addition, LiteReality introduces a training-free object retrieval module that achieves state-of-the-art similarity performance on the Scan2CAD benchmark, along with a robust material painting module capable of transferring appearances from images of any style to 3D assets -- even under severe misalignment, occlusion, and poor lighting. We demonstrate the effectiveness of LiteReality on both real-life scans and public datasets. Project page: this https URL; Video: this https URL 

**Abstract (ZH)**: 我们提出LiteReality，一种新颖的流水线，将室内的RGB-D扫描转换为紧凑、逼真且可交互的3D虚拟复制品。LiteReality不仅能重建视觉上相似于现实的场景，还支持图形流水线中必要的关键特征，如物体的独特性、关节运动、高质量的基于物理的渲染材料以及基于物理的交互。其核心过程首先是进行场景理解并将结果解析为一个连贯的3D布局和物体，利用结构化的场景图。然后通过检索受控资产数据库中最具视觉相似性的3D艺术家手工制作的模型重建场景。接下来，材质绘画模块通过恢复高质量、空间变化的材质来增强现实感。最后，重建的场景被整合到具备基本物理属性的模拟引擎中，以实现交互行为。生成的场景紧凑、可编辑，并完全兼容标准的图形流水线，适用于AR/VR、游戏、机器人技术和数字孪生等应用。此外，LiteReality引入了一个无训练对象检索模块，在Scan2CAD基准测试中实现了最先进的相似度性能，还具备一个鲁棒的材质绘画模块，能够将任何风格图像的外观转移到3D资产上，即使在严重的错位、遮挡和不良照明条件下也是如此。我们在实际扫描和公开数据集上展示了LiteReality的有效性。项目页面：https://this-url; 视频：https://this-url。 

---
# USAD: An Unsupervised Data Augmentation Spatio-Temporal Attention Diffusion Network 

**Title (ZH)**: USAD：无监督数据增强空时注意力扩散网络 

**Authors**: Ying Yu, Hang Xiao, Siyao Li, Jiarui Li, Haotian Tang, Hanyu Liu, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.02827)  

**Abstract**: The primary objective of human activity recognition (HAR) is to infer ongoing human actions from sensor data, a task that finds broad applications in health monitoring, safety protection, and sports analysis. Despite proliferating research, HAR still faces key challenges, including the scarcity of labeled samples for rare activities, insufficient extraction of high-level features, and suboptimal model performance on lightweight devices. To address these issues, this paper proposes a comprehensive optimization approach centered on multi-attention interaction mechanisms. First, an unsupervised, statistics-guided diffusion model is employed to perform data augmentation, thereby alleviating the problems of labeled data scarcity and severe class imbalance. Second, a multi-branch spatio-temporal interaction network is designed, which captures multi-scale features of sequential data through parallel residual branches with 3*3, 5*5, and 7*7 convolutional kernels. Simultaneously, temporal attention mechanisms are incorporated to identify critical time points, while spatial attention enhances inter-sensor interactions. A cross-branch feature fusion unit is further introduced to improve the overall feature representation capability. Finally, an adaptive multi-loss function fusion strategy is integrated, allowing for dynamic adjustment of loss weights and overall model optimization. Experimental results on three public datasets, WISDM, PAMAP2, and OPPORTUNITY, demonstrate that the proposed unsupervised data augmentation spatio-temporal attention diffusion network (USAD) achieves accuracies of 98.84%, 93.81%, and 80.92% respectively, significantly outperforming existing approaches. Furthermore, practical deployment on embedded devices verifies the efficiency and feasibility of the proposed method. 

**Abstract (ZH)**: 人类活动识别（HAR）的主要目标是从传感器数据中推断正在进行的人类行为，这一任务广泛应用于健康监测、安全保护和体育分析。尽管进行了大量的研究，HAR仍面临关键挑战，包括稀有活动标记样本稀缺、高级特征提取不足以及轻量级设备上的模型性能不佳。为解决这些问题，本文提出了一种以多注意力交互机制为中心的全面优化方法。首先，采用无监督的统计指导扩散模型进行数据增强，以缓解标记数据稀缺和类间严重不平衡的问题。其次，设计了一种多分支时空交互网络，通过并行残差分支中的3×3、5×5和7×7卷积核捕获序列数据的多尺度特征。同时，嵌入时间注意力机制以识别关键时间点，空间注意力则增强传感器间的交互。进一步引入跨分支特征融合单元以提升整体特征表示能力。最后，集成了一种自适应多损失函数融合策略，允许动态调整损失权重并优化整体模型。在三个公开数据集WISDM、PAMAP2和OPPORTUNITY上的实验结果表明，所提出的无监督数据增强时空注意力扩散网络（USAD）分别实现了98.84%、93.81%和80.92%的准确性，显著优于现有方法。此外，实际部署在嵌入式设备上验证了所提出方法的有效性和可行性。 

---
# Linear Attention with Global Context: A Multipole Attention Mechanism for Vision and Physics 

**Title (ZH)**: 全局上下文下的线性注意力：用于视觉和物理的多极注意力机制 

**Authors**: Alex Colagrande, Paul Caillon, Eva Feillet, Alexandre Allauzen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02748)  

**Abstract**: Transformers have become the de facto standard for a wide range of tasks, from image classification to physics simulations. Despite their impressive performance, the quadratic complexity of standard Transformers in both memory and time with respect to the input length makes them impractical for processing high-resolution inputs. Therefore, several variants have been proposed, the most successful relying on patchification, downsampling, or coarsening techniques, often at the cost of losing the finest-scale details. In this work, we take a different approach. Inspired by state-of-the-art techniques in $n$-body numerical simulations, we cast attention as an interaction problem between grid points. We introduce the Multipole Attention Neural Operator (MANO), which computes attention in a distance-based multiscale fashion. MANO maintains, in each attention head, a global receptive field and achieves linear time and memory complexity with respect to the number of grid points. Empirical results on image classification and Darcy flows demonstrate that MANO rivals state-of-the-art models such as ViT and Swin Transformer, while reducing runtime and peak memory usage by orders of magnitude. We open source our code for reproducibility at this https URL. 

**Abstract (ZH)**: 基于网格点间的相互作用的多极注意力神经算子 

---
# FairHuman: Boosting Hand and Face Quality in Human Image Generation with Minimum Potential Delay Fairness in Diffusion Models 

**Title (ZH)**: FairHuman: 在最小潜在延迟公平性约束下提高人类图像生成中的手部和面部质量 

**Authors**: Yuxuan Wang, Tianwei Cao, Huayu Zhang, Zhongjiang He, Kongming Liang, Zhanyu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.02714)  

**Abstract**: Image generation has achieved remarkable progress with the development of large-scale text-to-image models, especially diffusion-based models. However, generating human images with plausible details, such as faces or hands, remains challenging due to insufficient supervision of local regions during training. To address this issue, we propose FairHuman, a multi-objective fine-tuning approach designed to enhance both global and local generation quality fairly. Specifically, we first construct three learning objectives: a global objective derived from the default diffusion objective function and two local objectives for hands and faces based on pre-annotated positional priors. Subsequently, we derive the optimal parameter updating strategy under the guidance of the Minimum Potential Delay (MPD) criterion, thereby attaining fairness-ware optimization for this multi-objective problem. Based on this, our proposed method can achieve significant improvements in generating challenging local details while maintaining overall quality. Extensive experiments showcase the effectiveness of our method in improving the performance of human image generation under different scenarios. 

**Abstract (ZH)**: 基于多目标微调的FairHuman图像生成方法 

---
# ASDA: Audio Spectrogram Differential Attention Mechanism for Self-Supervised Representation Learning 

**Title (ZH)**: ASDA: 音频频谱差异注意力机制用于自我监督表示学习 

**Authors**: Junyu Wang, Tianrui Wang, Meng Ge, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02666)  

**Abstract**: In recent advancements in audio self-supervised representation learning, the standard Transformer architecture has emerged as the predominant approach, yet its attention mechanism often allocates a portion of attention weights to irrelevant information, potentially impairing the model's discriminative ability. To address this, we introduce a differential attention mechanism, which effectively mitigates ineffective attention allocation through the integration of dual-softmax operations and appropriately tuned differential coefficients. Experimental results demonstrate that our ASDA model achieves state-of-the-art (SOTA) performance across multiple benchmarks, including audio classification (49.0% mAP on AS-2M, 41.5% mAP on AS20K), keyword spotting (98.3% accuracy on SPC-2), and environmental sound classification (96.1% accuracy on ESC-50). These results highlight ASDA's effectiveness in audio tasks, paving the way for broader applications. 

**Abstract (ZH)**: 近期音频自监督表示学习的进展中，标准Transformer架构已成为主导方法，但其注意力机制往往会将部分注意力权重分配给无关信息，可能损害模型的辨别能力。为解决这一问题，我们引入了一种差异性注意力机制，该机制通过集成双softmax操作和适当地调平方差系数，有效地减轻了无效注意力分配。实验结果显示，我们的ASDA模型在多个基准测试中实现了最佳性能，涵盖音频分类（AS-2M上的49.0% mAP，AS20K上的41.5% mAP）、关键词识别（SPC-2上的98.3%准确率）和环境声分类（ESC-50上的96.1%准确率）。这些结果突显了ASDA在音频任务中的有效性，为更广泛的应用铺平了道路。 

---
# Addressing Camera Sensors Faults in Vision-Based Navigation: Simulation and Dataset Development 

**Title (ZH)**: 基于视觉的导航中相机传感器故障的处理：仿真与数据集开发 

**Authors**: Riccardo Gallon, Fabian Schiemenz, Alessandra Menicucci, Eberhard Gill  

**Link**: [PDF](https://arxiv.org/pdf/2507.02602)  

**Abstract**: The increasing importance of Vision-Based Navigation (VBN) algorithms in space missions raises numerous challenges in ensuring their reliability and operational robustness. Sensor faults can lead to inaccurate outputs from navigation algorithms or even complete data processing faults, potentially compromising mission objectives. Artificial Intelligence (AI) offers a powerful solution for detecting such faults, overcoming many of the limitations associated with traditional fault detection methods. However, the primary obstacle to the adoption of AI in this context is the lack of sufficient and representative datasets containing faulty image data.
This study addresses these challenges by focusing on an interplanetary exploration mission scenario. A comprehensive analysis of potential fault cases in camera sensors used within the VBN pipeline is presented. The causes and effects of these faults are systematically characterized, including their impact on image quality and navigation algorithm performance, as well as commonly employed mitigation strategies. To support this analysis, a simulation framework is introduced to recreate faulty conditions in synthetically generated images, enabling a systematic and controlled reproduction of faulty data. The resulting dataset of fault-injected images provides a valuable tool for training and testing AI-based fault detection algorithms. The final link to the dataset will be added after an embargo period. For peer-reviewers, this private link is available. 

**Abstract (ZH)**: 基于视觉的导航算法在空间任务中的重要性不断增加，确保其可靠性和操作鲁棒性提出了众多挑战。传感器故障可能导致导航算法输出不准确或将导致完全的数据处理故障，从而可能影响任务目标。人工智能提供了检测此类故障的强大解决方案，克服了许多传统故障检测方法的局限性。然而，阻碍在此背景下采用人工智能的主要障碍是缺乏包含故障图像数据的充分且具有代表性的数据集。

本文通过关注一次行星际探索任务场景来应对这些挑战。详细分析了VBN管道中使用的相机传感器可能出现的故障情况，并系统地 Characterized 这些故障的原因和影响，包括其对图像质量和导航算法性能的影响，以及常用的缓解策略。为支持这一分析，引入了一种仿真框架，用于在合成生成的图像中再现故障条件，从而能够在系统且可控的方式下重现故障数据。生成的包含故障注入的图像数据集提供了一个有价值的工具，用于训练和测试基于人工智能的故障检测算法。在预印本禁令期结束后，将提供最终的数据集链接。审稿人可以通过私人链接访问此数据集。 

---
# Detecting Multiple Diseases in Multiple Crops Using Deep Learning 

**Title (ZH)**: 多作物多种疾病的深度学习检测方法 

**Authors**: Vivek Yadav, Anugrah Jain  

**Link**: [PDF](https://arxiv.org/pdf/2507.02517)  

**Abstract**: India, as a predominantly agrarian economy, faces significant challenges in agriculture, including substantial crop losses caused by diseases, pests, and environmental stress. Early detection and accurate identification of diseases across different crops are critical for improving yield and ensuring food security. This paper proposes a deep learning based solution for detecting multiple diseases in multiple crops, aimed to cover India's diverse agricultural landscape. We first create a unified dataset encompassing images of 17 different crops and 34 different diseases from various available repositories. Proposed deep learning model is trained on this dataset and outperforms the state-of-the-art in terms of accuracy and the number of crops, diseases covered. We achieve a significant detection accuracy, i.e., 99 percent for our unified dataset which is 7 percent more when compared to state-of-the-art handling 14 crops and 26 different diseases only. By improving the number of crops and types of diseases that can be detected, proposed solution aims to provide a better product for Indian farmers. 

**Abstract (ZH)**: 印度作为一个以农业为主的经济体，面临着农业方面的重大挑战，包括由疾病、害虫和环境压力导致的作物大量损失。早发现和准确识别不同作物的病害对于提高产量和确保粮食安全至关重要。本文提出了一种基于深度学习的解决方案，用于检测多种作物的多种病害，旨在覆盖印度多样的农业景观。我们首先创建了一个统一的数据集，包含来自多个可用仓库的17种不同作物和34种不同病害的图像。所提出的深度学习模型在此数据集上进行了训练，并在准确性和覆盖的作物和疾病种类数量方面优于现有最佳方案。我们实现了显著的检测准确率，即统一数据集上的99%，比仅处理14种作物和26种不同病害的现有最佳方案高出7%。通过增加可检测的作物种类和病害类型数量，所提出的解决方案旨在为印度农民提供更好的产品。 

---
# Temporally-Aware Supervised Contrastive Learning for Polyp Counting in Colonoscopy 

**Title (ZH)**: 基于时间感知的监督对比学习在结肠镜检查中息肉计数 

**Authors**: Luca Parolari, Andrea Cherubini, Lamberto Ballan, Carlo Biffi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02493)  

**Abstract**: Automated polyp counting in colonoscopy is a crucial step toward automated procedure reporting and quality control, aiming to enhance the cost-effectiveness of colonoscopy screening. Counting polyps in a procedure involves detecting and tracking polyps, and then clustering tracklets that belong to the same polyp entity. Existing methods for polyp counting rely on self-supervised learning and primarily leverage visual appearance, neglecting temporal relationships in both tracklet feature learning and clustering stages. In this work, we introduce a paradigm shift by proposing a supervised contrastive loss that incorporates temporally-aware soft targets. Our approach captures intra-polyp variability while preserving inter-polyp discriminability, leading to more robust clustering. Additionally, we improve tracklet clustering by integrating a temporal adjacency constraint, reducing false positive re-associations between visually similar but temporally distant tracklets. We train and validate our method on publicly available datasets and evaluate its performance with a leave-one-out cross-validation strategy. Results demonstrate a 2.2x reduction in fragmentation rate compared to prior approaches. Our results highlight the importance of temporal awareness in polyp counting, establishing a new state-of-the-art. Code is available at this https URL. 

**Abstract (ZH)**: 自动结肠镜检查中的息肉计数是实现自动检查程序报告和质量控制的关键步骤，旨在提高结肠镜筛查的成本效益。我们通过引入基于监督对比损失的方法，结合时空信息，提出了一种新的息肉计数框架，显著提高了息肉计数的准确性和鲁棒性。 

---
# CrowdTrack: A Benchmark for Difficult Multiple Pedestrian Tracking in Real Scenarios 

**Title (ZH)**: CrowdTrack: 一个适用于真实场景中困难多行人跟踪的基准 

**Authors**: Teng Fu, Yuwen Chen, Zhuofan Chen, Mengyang Zhao, Bin Li, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.02479)  

**Abstract**: Multi-object tracking is a classic field in computer vision. Among them, pedestrian tracking has extremely high application value and has become the most popular research category. Existing methods mainly use motion or appearance information for tracking, which is often difficult in complex scenarios. For the motion information, mutual occlusions between objects often prevent updating of the motion state; for the appearance information, non-robust results are often obtained due to reasons such as only partial visibility of the object or blurred images. Although learning how to perform tracking in these situations from the annotated data is the simplest solution, the existing MOT dataset fails to satisfy this solution. Existing methods mainly have two drawbacks: relatively simple scene composition and non-realistic scenarios. Although some of the video sequences in existing dataset do not have the above-mentioned drawbacks, the number is far from adequate for research purposes. To this end, we propose a difficult large-scale dataset for multi-pedestrian tracking, shot mainly from the first-person view and all from real-life complex scenarios. We name it ``CrowdTrack'' because there are numerous objects in most of the sequences. Our dataset consists of 33 videos, containing a total of 5,185 trajectories. Each object is annotated with a complete bounding box and a unique object ID. The dataset will provide a platform to facilitate the development of algorithms that remain effective in complex situations. We analyzed the dataset comprehensively and tested multiple SOTA models on our dataset. Besides, we analyzed the performance of the foundation models on our dataset. The dataset and project code is released at: this https URL . 

**Abstract (ZH)**: 多目标跟踪是计算机视觉中的一个经典领域。其中，行人跟踪具有极高的应用价值，已经成为最热门的研究领域。现有方法主要利用运动或外观信息进行跟踪，在复杂场景中常常难以实现。对于运动信息，对象间的相互遮挡经常阻碍运动状态的更新；对于外观信息，由于只有一部分可见或图像模糊等原因，常常得到不 robust 的结果。尽管从标注数据中学习如何在这些场景下进行跟踪是最简单的解决方案，但现有多目标跟踪（MOT）数据集未能满足这一需求。现有方法主要存在两个缺点：相对简单的场景组成和非现实的场景。尽管现有数据集中的一些视频序列并未存在上述缺点，但数量远远不足以满足研究需求。为此，我们提出了一大规模的复杂场景下多行人跟踪数据集，主要从第一人称视角拍摄，并全部来自真实的复杂场景。我们将其命名为“CrowdTrack”因为几乎所有序列中都有大量对象。该数据集包含33个视频，共5,185条轨迹。每个对象都标注了完整的边界框和唯一的对象ID。该数据集将为在复杂场景下仍能有效工作的算法开发提供平台。我们对数据集进行了全面分析，并在我们的数据集上测试了多个最新模型的性能。此外，我们还分析了基础模型在我们数据集上的性能。该数据集和项目代码已发布在: [这个链接](this https URL) 。 

---
# Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Deepfake Video Detection 

**Title (ZH)**: 超越空间频率：基于像素级时间频率的深fake视频检测 

**Authors**: Taehoon Kim, Jongwook Choi, Yonghyun Jeong, Haeun Noh, Jaejun Yoo, Seungryul Baek, Jongwon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02398)  

**Abstract**: We introduce a deepfake video detection approach that exploits pixel-wise temporal inconsistencies, which traditional spatial frequency-based detectors often overlook. Traditional detectors represent temporal information merely by stacking spatial frequency spectra across frames, resulting in the failure to detect temporal artifacts in the pixel plane. Our approach performs a 1D Fourier transform on the time axis for each pixel, extracting features highly sensitive to temporal inconsistencies, especially in areas prone to unnatural movements. To precisely locate regions containing the temporal artifacts, we introduce an attention proposal module trained in an end-to-end manner. Additionally, our joint transformer module effectively integrates pixel-wise temporal frequency features with spatio-temporal context features, expanding the range of detectable forgery artifacts. Our framework represents a significant advancement in deepfake video detection, providing robust performance across diverse and challenging detection scenarios. 

**Abstract (ZH)**: 基于像素级时间不一致性的深伪视频检测方法：超越传统基于空间频率的检测器 

---
# Holistic Tokenizer for Autoregressive Image Generation 

**Title (ZH)**: 全面的分词器用于自回归图像生成 

**Authors**: Anlin Zheng, Haochen Wang, Yucheng Zhao, Weipeng Deng, Tiancai Wang, Xiangyu Zhang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02358)  

**Abstract**: The vanilla autoregressive image generation model generates visual tokens in a step-by-step fashion, which limits the ability to capture holistic relationships among token sequences. Moreover, most visual tokenizers map local image patches into latent tokens, leading to limited global information. To address this, we introduce \textit{Hita}, a novel image tokenizer for autoregressive (AR) image generation. It introduces a holistic-to-local tokenization scheme with learnable holistic queries and local patch tokens. Besides, Hita incorporates two key strategies for improved alignment with the AR generation process: 1) it arranges a sequential structure with holistic tokens at the beginning followed by patch-level tokens while using causal attention to maintain awareness of previous tokens; and 2) before feeding the de-quantized tokens into the decoder, Hita adopts a lightweight fusion module to control information flow to prioritize holistic tokens. Extensive experiments show that Hita accelerates the training speed of AR generators and outperforms those trained with vanilla tokenizers, achieving \textbf{2.59 FID} and \textbf{281.9 IS} on the ImageNet benchmark. A detailed analysis of the holistic representation highlights its ability to capture global image properties such as textures, materials, and shapes. Additionally, Hita also demonstrates effectiveness in zero-shot style transfer and image in-painting. The code is available at \href{this https URL}{this https URL} 

**Abstract (ZH)**: 基于整体到局部的自回归图像生成模型Hita及其应用 

---
# Neural Network-based Study for Rice Leaf Disease Recognition and Classification: A Comparative Analysis Between Feature-based Model and Direct Imaging Model 

**Title (ZH)**: 基于神经网络的水稻叶片疾病识别与分类研究：特征模型与直接成像模型的比较分析 

**Authors**: Farida Siddiqi Prity, Mirza Raquib, Saydul Akbar Murad, Md. Jubayar Alam Rafi, Md. Khairul Bashar Bhuiyan, Anupam Kumar Bairagi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02322)  

**Abstract**: Rice leaf diseases significantly reduce productivity and cause economic losses, highlighting the need for early detection to enable effective management and improve yields. This study proposes Artificial Neural Network (ANN)-based image-processing techniques for timely classification and recognition of rice diseases. Despite the prevailing approach of directly inputting images of rice leaves into ANNs, there is a noticeable absence of thorough comparative analysis between the Feature Analysis Detection Model (FADM) and Direct Image-Centric Detection Model (DICDM), specifically when it comes to evaluating the effectiveness of Feature Extraction Algorithms (FEAs). Hence, this research presents initial experiments on the Feature Analysis Detection Model, utilizing various image Feature Extraction Algorithms, Dimensionality Reduction Algorithms (DRAs), Feature Selection Algorithms (FSAs), and Extreme Learning Machine (ELM). The experiments are carried out on datasets encompassing bacterial leaf blight, brown spot, leaf blast, leaf scald, Sheath blight rot, and healthy leaf, utilizing 10-fold Cross-Validation method. A Direct Image-Centric Detection Model is established without the utilization of any FEA, and the evaluation of classification performance relies on different metrics. Ultimately, an exhaustive contrast is performed between the achievements of the Feature Analysis Detection Model and Direct Image-Centric Detection Model in classifying rice leaf diseases. The results reveal that the highest performance is attained using the Feature Analysis Detection Model. The adoption of the proposed Feature Analysis Detection Model for detecting rice leaf diseases holds excellent potential for improving crop health, minimizing yield losses, and enhancing overall productivity and sustainability of rice farming. 

**Abstract (ZH)**: 基于人工神经网络的图像处理技术在水稻叶片疾病及时分类与识别中的应用：特征分析检测模型与直接图像中心检测模型的比较研究 

---
# MAGIC: Mask-Guided Diffusion Inpainting with Multi-Level Perturbations and Context-Aware Alignment for Few-Shot Anomaly Generation 

**Title (ZH)**: MAGIC：基于掩膜引导的多级扰动与上下文意识对齐扩散补全以实现少样本异常生成 

**Authors**: JaeHyuck Choi, MinJun Kim, JeHyeong Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02314)  

**Abstract**: Few-shot anomaly generation is emerging as a practical solution for augmenting the scarce anomaly data in industrial quality control settings. An ideal generator would meet three demands at once, namely (i) keep the normal background intact, (ii) inpaint anomalous regions to tightly overlap with the corresponding anomaly masks, and (iii) generate anomalous regions in a semantically valid location, while still producing realistic, diverse appearances from only a handful of real examples. Existing diffusion-based methods usually satisfy at most two of these requirements: global anomaly generators corrupt the background, whereas mask-guided ones often falter when the mask is imprecise or misplaced. We propose MAGIC--Mask-guided inpainting with multi-level perturbations and Context-aware alignment--to resolve all three issues. At its core, MAGIC fine-tunes a Stable Diffusion inpainting backbone that preserves normal regions and ensures strict adherence of the synthesized anomaly to the supplied mask, directly addressing background corruption and misalignment. To offset the diversity loss that fine-tuning can cause, MAGIC adds two complementary perturbation strategies: (i) Gaussian prompt-level perturbation applied during fine-tuning and inference that broadens the global appearance of anomalies while avoiding low-fidelity textual appearances, and (ii) mask-guided spatial noise injection that enriches local texture variations. Additionally, the context-aware mask alignment module forms semantic correspondences and relocates masks so that every anomaly remains plausibly contained within the host object, eliminating out-of-boundary artifacts. Under a consistent identical evaluation protocol on the MVTec-AD dataset, MAGIC outperforms previous state-of-the-arts in downstream anomaly tasks. 

**Abstract (ZH)**: 基于掩膜的多级扰动与上下文对齐的生成模型Magic：三重需求的 Few-shot 异常生成 

---
# Spotlighting Partially Visible Cinematic Language for Video-to-Audio Generation via Self-distillation 

**Title (ZH)**: 突出部分可见的cinematic语言以实现视频到音频生成的自精练 

**Authors**: Feizhen Huang, Yu Wu, Yutian Lin, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.02271)  

**Abstract**: Video-to-Audio (V2A) Generation achieves significant progress and plays a crucial role in film and video post-production. However, current methods overlook the cinematic language, a critical component of artistic expression in filmmaking. As a result, their performance deteriorates in scenarios where Foley targets are only partially visible. To address this challenge, we propose a simple self-distillation approach to extend V2A models to cinematic language scenarios. By simulating the cinematic language variations, the student model learns to align the video features of training pairs with the same audio-visual correspondences, enabling it to effectively capture the associations between sounds and partial visual information. Our method not only achieves impressive improvements under partial visibility across all evaluation metrics, but also enhances performance on the large-scale V2A dataset, VGGSound. 

**Abstract (ZH)**: 视频到音频（V2A）生成在影视后期制作中取得了显著进展并发挥着关键作用。然而，当前方法忽视了电影语言这一影视创作艺术表达中的关键成分。因此，在只有部分配音目标可见的场景中，其性能下降。为了解决这一挑战，我们提出了一种简单的自我精炼方法，将V2A模型扩展到电影语言场景中。通过模拟电影语言的变化，学生模型学习调整训练配对的视频特征与相同音视频对应关系，使其能够有效地捕捉声音与部分视觉信息之间的关联。我们的方法不仅在所有评估指标下实现了在部分可见情况下的显著改进，还提高了在大规模V2A数据集VGGSound上的性能。 

---
