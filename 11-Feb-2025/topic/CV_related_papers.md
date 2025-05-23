# Predictive Red Teaming: Breaking Policies Without Breaking Robots 

**Title (ZH)**: 预测性红队行动：不破坏机器人破解政策 

**Authors**: Anirudha Majumdar, Mohit Sharma, Dmitry Kalashnikov, Sumeet Singh, Pierre Sermanet, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2502.06575)  

**Abstract**: Visuomotor policies trained via imitation learning are capable of performing challenging manipulation tasks, but are often extremely brittle to lighting, visual distractors, and object locations. These vulnerabilities can depend unpredictably on the specifics of training, and are challenging to expose without time-consuming and expensive hardware evaluations. We propose the problem of predictive red teaming: discovering vulnerabilities of a policy with respect to environmental factors, and predicting the corresponding performance degradation without hardware evaluations in off-nominal scenarios. In order to achieve this, we develop RoboART: an automated red teaming (ART) pipeline that (1) modifies nominal observations using generative image editing to vary different environmental factors, and (2) predicts performance under each variation using a policy-specific anomaly detector executed on edited observations. Experiments across 500+ hardware trials in twelve off-nominal conditions for visuomotor diffusion policies demonstrate that RoboART predicts performance degradation with high accuracy (less than 0.19 average difference between predicted and real success rates). We also demonstrate how predictive red teaming enables targeted data collection: fine-tuning with data collected under conditions predicted to be adverse boosts baseline performance by 2-7x. 

**Abstract (ZH)**: 基于预测性红队的策略脆弱性发现与性能预测方法 

---
# Sustainable Adaptation for Autonomous Driving with the Mixture of Progressive Experts Networ 

**Title (ZH)**: 基于渐进专家网络混合的自主驾驶可持续适应方法 

**Authors**: Yixin Cui, Shuo Yang, Chi Wan, Xincheng Li, Jiaming Xing, Yuanjian Zhang, Yanjun Huang, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05943)  

**Abstract**: Learning-based autonomous driving methods require continuous acquisition of domain knowledge to adapt to diverse driving scenarios. However, due to the inherent challenges of long-tailed data distribution, current approaches still face limitations in complex and dynamic driving environments, particularly when encountering new scenarios and data. This underscores the necessity for enhanced continual learning capabilities to improve system adaptability. To address these challenges, the paper introduces a dynamic progressive optimization framework that facilitates adaptation to variations in dynamic environments, achieved by integrating reinforcement learning and supervised learning for data aggregation. Building on this framework, we propose the Mixture of Progressive Experts (MoPE) network. The proposed method selectively activates multiple expert models based on the distinct characteristics of each task and progressively refines the network architecture to facilitate adaptation to new tasks. Simulation results show that the MoPE model outperforms behavior cloning methods, achieving up to a 7.3% performance improvement in intricate urban road environments. 

**Abstract (ZH)**: 基于学习的自动驾驶方法需要不断获取领域知识以适应多样的驾驶场景。然而，由于长尾数据分布固有的挑战，当前方法在复杂和动态的驾驶环境中仍然面临局限，特别是在遇到新场景和数据时。这突显了增强持续学习能力的必要性，以提高系统的适应性。为应对这些挑战，本文引入了一个动态渐进优化框架，通过结合强化学习和监督学习进行数据聚合，以适应动态环境中的变化。在此框架基础上，我们提出了混合渐进专家网络（MoPE）。所提出的方法根据每个任务的特定特性有选择地激活多个专家模型，并逐步优化网络架构，以促进对新任务的适应。仿真实验结果表明，MoPE模型优于行为克隆方法，在复杂的城市道路环境中性能提升最高可达7.3%。 

---
# AToM: Adaptive Theory-of-Mind-Based Human Motion Prediction in Long-Term Human-Robot Interactions 

**Title (ZH)**: AToM：基于自适应心智理论的人motion预测在长时间人机交互中的应用 

**Authors**: Yuwen Liao, Muqing Cao, Xinhang Xu, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.05792)  

**Abstract**: Humans learn from observations and experiences to adjust their behaviours towards better performance. Interacting with such dynamic humans is challenging, as the robot needs to predict the humans accurately for safe and efficient operations. Long-term interactions with dynamic humans have not been extensively studied by prior works. We propose an adaptive human prediction model based on the Theory-of-Mind (ToM), a fundamental social-cognitive ability that enables humans to infer others' behaviours and intentions. We formulate the human internal belief about others using a game-theoretic model, which predicts the future motions of all agents in a navigation scenario. To estimate an evolving belief, we use an Unscented Kalman Filter to update the behavioural parameters in the human internal model. Our formulation provides unique interpretability to dynamic human behaviours by inferring how the human predicts the robot. We demonstrate through long-term experiments in both simulations and real-world settings that our prediction effectively promotes safety and efficiency in downstream robot planning. Code will be available at this https URL. 

**Abstract (ZH)**: 人类通过观察和经验调整行为以获得更好的表现。与这样的动态人类交互具有挑战性，因为机器人需要准确预测人类行为以确保安全和高效的操作。先前的研究尚未广泛探讨长期与动态人类的交互。我们提出了一种基于理论思维（ToM）的自适应人类预测模型，理论思维是一种基础的社会认知能力，使人类能够推断他人的行为和意图。我们使用博弈论模型来表达人类对他人内部信念，并预测导航场景中所有代理的未来运动。为估计不断 evolving 的信念，我们使用无迹卡尔曼滤波器来更新人类内部模型中的行为参数。我们的建模为动态人类行为提供了独特的可解释性，通过推断人类如何预测机器人。我们在仿真和真实-world 设置中通过长期实验演示，我们的预测有效促进了下游机器人规划的安全性和效率。代码将在此处 https:// 提供。 

---
# PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map 

**Title (ZH)**: PINGS：基于点的隐式神经映射中的高斯散点与距离场相结合 

**Authors**: Yue Pan, Xingguang Zhong, Liren Jin, Louis Wiesmann, Marija Popović, Jens Behley, Cyrill Stachniss  

**Link**: [PDF](https://arxiv.org/pdf/2502.05752)  

**Abstract**: Robots require high-fidelity reconstructions of their environment for effective operation. Such scene representations should be both, geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, the scalable incremental mapping of both fields consistently and at the same time with high quality remains challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We devise a LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to the state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by leveraging the constraints from the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction. 

**Abstract (ZH)**: 机器人需要对其环境进行高保真重建以实现有效操作。这样的场景表示既要几何上准确，又要具备照片现实性，以支持下游任务。虽然可以通过构建来自距离传感器的距离场和来自相机的辐射场来实现这一点，但同时以高质量的方式进行可扩展的增量映射仍具有挑战性。在本文中，我们提出了一种新的地图表示法，该表示法将连续的带符号距离场和高斯采样辐射场统一在一种弹性且紧凑的基于点的隐式神经地图中。通过在这些领域之间强求几何一致性，我们通过利用这两种模态的优势实现了相互改进。我们使用所提出的地图表示法开发了一种基于激光雷达和视觉的SLAM系统PINGS，并在多个具有挑战性的大规模数据集上对其进行评估。实验结果表明，PINGS能够通过紧凑的神经点集增量构建全局一致的距离和辐射场。与现有方法相比，PINGS通过利用距离场的约束条件，在新视角下实现了更优的 photometric 和几何渲染。此外，PINGS 利用辐射场中的密集 photometric 提示和多视角一致性，生成更准确的距离场，从而提高运动估计和网格重建的精度。 

---
# Accelerating Outlier-robust Rotation Estimation by Stereographic Projection 

**Title (ZH)**: 基于立体投影加速抗离群点旋转估计 

**Authors**: Taosi Xu, Yinlong Liu, Xianbo Wang, Zhi-Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06337)  

**Abstract**: Rotation estimation plays a fundamental role in many computer vision and robot tasks. However, efficiently estimating rotation in large inputs containing numerous outliers (i.e., mismatches) and noise is a recognized challenge. Many robust rotation estimation methods have been designed to address this challenge. Unfortunately, existing methods are often inapplicable due to their long computation time and the risk of local optima. In this paper, we propose an efficient and robust rotation estimation method. Specifically, our method first investigates geometric constraints involving only the rotation axis. Then, it uses stereographic projection and spatial voting techniques to identify the rotation axis and angle. Furthermore, our method efficiently obtains the optimal rotation estimation and can estimate multiple rotations simultaneously. To verify the feasibility of our method, we conduct comparative experiments using both synthetic and real-world data. The results show that, with GPU assistance, our method can solve large-scale ($10^6$ points) and severely corrupted (90\% outlier rate) rotation estimation problems within 0.07 seconds, with an angular error of only 0.01 degrees, which is superior to existing methods in terms of accuracy and efficiency. 

**Abstract (ZH)**: 旋转估计在许多计算机视觉和机器人任务中起着基础性作用。然而，在包含大量离群点（即错误匹配）和噪声的大输入中高效地估计旋转是一个公认的挑战。许多稳健的旋转估计方法已经被设计出来以应对这一挑战。不幸的是，现有方法往往由于计算时间过长和容易陷入局部最优而不可用。在本文中，我们提出了一种高效且稳健的旋转估计方法。具体而言，该方法首先探讨仅涉及旋转轴的几何约束。然后，它使用立体投影和空间投票技术来识别旋转轴和角度。此外，该方法能有效地获得最佳旋转估计，并能同时估计多个旋转。为了验证该方法的可行性，我们在合成和真实数据上进行了对比实验。结果显示，在GPU辅助下，该方法可以在0.07秒内解决包含100万点的大规模和严重污染（90%离群点比率）的旋转估计问题，并且角度误差仅为0.01度，无论在准确性和效率方面都优于现有方法。 

---
# Vision-in-the-loop Simulation for Deep Monocular Pose Estimation of UAV in Ocean Environment 

**Title (ZH)**: 海洋环境中基于视SEE-in-the-loop仿真的单目无人机姿态估计 

**Authors**: Maneesha Wickramasuriya, Beomyeol Yu, Taeyoung Lee, Murray Snyder  

**Link**: [PDF](https://arxiv.org/pdf/2502.05409)  

**Abstract**: This paper proposes a vision-in-the-loop simulation environment for deep monocular pose estimation of a UAV operating in an ocean environment. Recently, a deep neural network with a transformer architecture has been successfully trained to estimate the pose of a UAV relative to the flight deck of a research vessel, overcoming several limitations of GPS-based approaches. However, validating the deep pose estimation scheme in an actual ocean environment poses significant challenges due to the limited availability of research vessels and the associated operational costs. To address these issues, we present a photo-realistic 3D virtual environment leveraging recent advancements in Gaussian splatting, a novel technique that represents 3D scenes by modeling image pixels as Gaussian distributions in 3D space, creating a lightweight and high-quality visual model from multiple viewpoints. This approach enables the creation of a virtual environment integrating multiple real-world images collected in situ. The resulting simulation enables the indoor testing of flight maneuvers while verifying all aspects of flight software, hardware, and the deep monocular pose estimation scheme. This approach provides a cost-effective solution for testing and validating the autonomous flight of shipboard UAVs, specifically focusing on vision-based control and estimation algorithms. 

**Abstract (ZH)**: 本文提出了一种视景环路仿真环境，用于海洋环境下无人机单目姿态估计。最近，一种具有变压器架构的深度神经网络已成功训练，用于估计无人机相对于研究船飞行甲板的姿态，克服了基于GPS方法的若干限制。然而，在实际海洋环境中验证深度姿态估计方案面临着显著挑战，主要是由于研究船只的有限可用性和相应的运营成本。为解决这些问题，我们提出了一个基于最近Gaussian splatting进展的逼真3D虚拟环境，这是一种通过将图像像素建模为3D空间中的高斯分布来表示3D场景的新技术，从而从多个视角创建了一个轻量级且高质量的视觉模型。此方法使创建一个集成了多个现场采集实况图像的虚拟环境成为可能。该仿真方法能够在室内测试飞行机动性的同时，验证飞行软件、硬件以及单目姿态估计方案的所有方面。这种方法为测试和验证机载无人机的自主飞行提供了一种成本有效的方法，特别关注基于视觉的控制和估计算法。 

---
# NextBestPath: Efficient 3D Mapping of Unseen Environments 

**Title (ZH)**: NextBestPath: 效率高的未见环境三维建图 

**Authors**: Shiyao Li, Antoine Guédon, Clémentin Boittiaux, Shizhe Chen, Vincent Lepetit  

**Link**: [PDF](https://arxiv.org/pdf/2502.05378)  

**Abstract**: This work addresses the problem of active 3D mapping, where an agent must find an efficient trajectory to exhaustively reconstruct a new scene. Previous approaches mainly predict the next best view near the agent's location, which is prone to getting stuck in local areas. Additionally, existing indoor datasets are insufficient due to limited geometric complexity and inaccurate ground truth meshes. To overcome these limitations, we introduce a novel dataset AiMDoom with a map generator for the Doom video game, enabling to better benchmark active 3D mapping in diverse indoor environments. Moreover, we propose a new method we call next-best-path (NBP), which predicts long-term goals rather than focusing solely on short-sighted views. The model jointly predicts accumulated surface coverage gains for long-term goals and obstacle maps, allowing it to efficiently plan optimal paths with a unified model. By leveraging online data collection, data augmentation and curriculum learning, NBP significantly outperforms state-of-the-art methods on both the existing MP3D dataset and our AiMDoom dataset, achieving more efficient mapping in indoor environments of varying complexity. 

**Abstract (ZH)**: 这种工作解决了主动3D建图的问题，其中代理剂必须找到一条高效的路径来全面重建一个新的场景。此前的方法主要预测代理剂位置附近的最优视角，容易陷入局部区域。此外，现有的室内数据集由于几何复杂度有限且ground truth网格不准确而不足。为克服这些限制，我们引入了一个名为AiMDoom的新数据集及其映射生成器，用于Doom视频游戏，以更好地在多样化的室内环境中基准测试主动3D建图。此外，我们提出了一种新的方法，称为最优路径（Next-Best-Path，NBP），该方法预测长期目标而非仅为短视视角。该模型联合预测长期目标的累积表面覆盖增益和障碍物地图，使其能够使用统一模型高效地规划最优路径。通过利用在线数据采集、数据增强和级联学习，NBP在现有的MP3D数据集和我们的AiMDoom数据集上均显著优于现有方法，在不同复杂度的室内环境中实现了更高效的建图。 

---
# AppVLM: A Lightweight Vision Language Model for Online App Control 

**Title (ZH)**: AppVLM：一种轻量级的视觉语言模型用于在线应用控制 

**Authors**: Georgios Papoudakis, Thomas Coste, Zhihao Wu, Jianye Hao, Jun Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.06395)  

**Abstract**: The utilisation of foundation models as smartphone assistants, termed app agents, is a critical research challenge. These agents aim to execute human instructions on smartphones by interpreting textual instructions and performing actions via the device's interface. While promising, current approaches face significant limitations. Methods that use large proprietary models, such as GPT-4o, are computationally expensive, while those that use smaller fine-tuned models often lack adaptability to out-of-distribution tasks. In this work, we introduce AppVLM, a lightweight Vision-Language Model (VLM). First, we fine-tune it offline on the AndroidControl dataset. Then, we refine its policy by collecting data from the AndroidWorld environment and performing further training iterations. Our results indicate that AppVLM achieves the highest action prediction accuracy in offline evaluation on the AndroidControl dataset, compared to all evaluated baselines, and matches GPT-4o in online task completion success rate in the AndroidWorld environment, while being up to ten times faster. This makes AppVLM a practical and efficient solution for real-world deployment. 

**Abstract (ZH)**: 基于基础模型的智能手机助手应用：AppVLM的研究与实现 

---
# KARST: Multi-Kernel Kronecker Adaptation with Re-Scaling Transmission for Visual Classification 

**Title (ZH)**: KARST: 多核克罗内克适应与重新缩放传输的视觉分类 

**Authors**: Yue Zhu, Haiwen Diao, Shang Gao, Long Chen, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06779)  

**Abstract**: Fine-tuning pre-trained vision models for specific tasks is a common practice in computer vision. However, this process becomes more expensive as models grow larger. Recently, parameter-efficient fine-tuning (PEFT) methods have emerged as a popular solution to improve training efficiency and reduce storage needs by tuning additional low-rank modules within pre-trained backbones. Despite their advantages, they struggle with limited representation capabilities and misalignment with pre-trained intermediate features. To address these issues, we introduce an innovative Multi-Kernel Kronecker Adaptation with Re-Scaling Transmission (KARST) for various recognition tasks. Specifically, its multi-kernel design extends Kronecker projections horizontally and separates adaptation matrices into multiple complementary spaces, reducing parameter dependency and creating more compact subspaces. Besides, it incorporates extra learnable re-scaling factors to better align with pre-trained feature distributions, allowing for more flexible and balanced feature aggregation. Extensive experiments validate that our KARST outperforms other PEFT counterparts with a negligible inference cost due to its re-parameterization characteristics. Code is publicly available at: this https URL. 

**Abstract (ZH)**: 细调预训练视觉模型以进行特定任务是计算机视觉中的一种常用做法。然而，随着模型规模的扩大，这一过程变得更加昂贵。最近，参数高效细调（PEFT）方法作为通过在预训练主干内部调整额外的低秩模块来提高训练效率并减少存储需求的一种流行解决方案而兴起。尽管它们具有优势，但在表示能力和与预训练中间特征的对齐方面仍存在问题。为了解决这些问题，我们引入了一种用于各类识别任务的创新多核克罗内克自适应与重缩放传输（KARST）。具体而言，其多核设计在水平方向上扩展了克罗内克投影，并将适应矩阵分离到多个互补空间中，减少了参数依赖并创建了更紧凑的子空间。此外，它还引入了额外的可学习重缩放因子，以更好地与预训练特征分布对齐，从而实现更灵活和平衡的特征聚合。广泛的实验验证了我们的KARST在忽略重参数化成本的情况下，优于其他PEFT方法。代码已公开：this https URL。 

---
# CHIRLA: Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis 

**Title (ZH)**: CHIRLA: 综合高分辨率识别和再识别以进行大规模分析 

**Authors**: Bessie Dominguez-Dager, Felix Escalona, Francisco Gomez-Donoso, Miguel Cazorla  

**Link**: [PDF](https://arxiv.org/pdf/2502.06681)  

**Abstract**: Person re-identification (Re-ID) is a key challenge in computer vision, requiring the matching of individuals across different cameras, locations, and time periods. While most research focuses on short-term scenarios with minimal appearance changes, real-world applications demand robust Re-ID systems capable of handling long-term scenarios, where persons' appearances can change significantly due to variations in clothing and physical characteristics. In this paper, we present CHIRLA, Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis, a novel dataset specifically designed for long-term person Re-ID. CHIRLA consists of recordings from strategically placed cameras over a seven-month period, capturing significant variations in both temporal and appearance attributes, including controlled changes in participants' clothing and physical features. The dataset includes 22 individuals, four connected indoor environments, and seven cameras. We collected more than five hours of video that we semi-automatically labeled to generate around one million bounding boxes with identity annotations. By introducing this comprehensive benchmark, we aim to facilitate the development and evaluation of Re-ID algorithms that can reliably perform in challenging, long-term real-world scenarios. 

**Abstract (ZH)**: 全面高分辨率长期人群重识别数据集CHIRLA 

---
# Few-Shot Classification and Anatomical Localization of Tissues in SPECT Imaging 

**Title (ZH)**: 基于SPECT成像的少量样本分类与组织解剖定位 

**Authors**: Mohammed Abdul Hafeez Khan, Samuel Morries Boddepalli, Siddhartha Bhattacharyya, Debasis Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2502.06632)  

**Abstract**: Accurate classification and anatomical localization are essential for effective medical diagnostics and research, which may be efficiently performed using deep learning techniques. However, availability of limited labeled data poses a significant challenge. To address this, we adapted Prototypical Networks and the Propagation-Reconstruction Network (PRNet) for few-shot classification and localization, respectively, in Single Photon Emission Computed Tomography (SPECT) images. For the proof of concept we used a 2D-sliced image cropped around heart. The Prototypical Network, with a pre-trained ResNet-18 backbone, classified ventricles, myocardium, and liver tissues with 96.67% training and 93.33% validation accuracy. PRNet, adapted for 2D imaging with an encoder-decoder architecture and skip connections, achieved a training loss of 1.395, accurately reconstructing patches and capturing spatial relationships. These results highlight the potential of Prototypical Networks for tissue classification with limited labeled data and PRNet for anatomical landmark localization, paving the way for improved performance in deep learning frameworks. 

**Abstract (ZH)**: 准确的分类和解剖定位对于有效的医学诊断和研究至关重要，这可以通过深度学习技术高效地实现。然而，可用的有限标记数据提出了显著挑战。为应对这一挑战，我们针对单光子发射计算机断层摄影(SPECT)图像，将原型网络（Prototypical Networks）和传播重建网络（PRNet）分别应用于少样本分类和定位。作为概念验证，我们使用了心脏周围的2D切片图像。预训练的ResNet-18作为原型网络的骨干网络，分类心室、心肌和肝组织，训练准确率为96.67%，验证准确率为93.33%。PRNet通过编码-解码架构和跳跃连接被改编用于2D成像，训练损失为1.395，准确地重建了图像斑块并捕捉了空间关系。这些结果强调了在有限标记数据下使用原型网络进行组织分类和使用PRNet进行解剖标志点定位的潜力，为深度学习框架中的性能提升铺平了道路。 

---
# Conformal Predictions for Human Action Recognition with Vision-Language Models 

**Title (ZH)**: 视觉语言模型驱动的人类动作识别构形预测 

**Authors**: Bary Tim, Fuchs Clément, Macq Benoît  

**Link**: [PDF](https://arxiv.org/pdf/2502.06631)  

**Abstract**: Human-In-The-Loop (HITL) frameworks are integral to many real-world computer vision systems, enabling human operators to make informed decisions with AI assistance. Conformal Predictions (CP), which provide label sets with rigorous guarantees on ground truth inclusion probabilities, have recently gained traction as a valuable tool in HITL settings. One key application area is video surveillance, closely associated with Human Action Recognition (HAR). This study explores the application of CP on top of state-of-the-art HAR methods that utilize extensively pre-trained Vision-Language Models (VLMs). Our findings reveal that CP can significantly reduce the average number of candidate classes without modifying the underlying VLM. However, these reductions often result in distributions with long tails. To address this, we introduce a method based on tuning the temperature parameter of the VLMs to minimize these tails without requiring additional calibration data. Our code is made available on GitHub at the address this https URL. 

**Abstract (ZH)**: 基于人类在环的条件随机场在先进视觉语言模型上的人体动作识别中的应用及其改进方法 

---
# TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models 

**Title (ZH)**: TripoSG：大规模矫正流模型下的高保真3D形状合成 

**Authors**: Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, Yan-Pei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2502.06608)  

**Abstract**: Recent advancements in diffusion techniques have propelled image and video generation to unprece- dented levels of quality, significantly accelerating the deployment and application of generative AI. However, 3D shape generation technology has so far lagged behind, constrained by limitations in 3D data scale, complexity of 3D data process- ing, and insufficient exploration of advanced tech- niques in the 3D domain. Current approaches to 3D shape generation face substantial challenges in terms of output quality, generalization capa- bility, and alignment with input conditions. We present TripoSG, a new streamlined shape diffu- sion paradigm capable of generating high-fidelity 3D meshes with precise correspondence to input images. Specifically, we propose: 1) A large-scale rectified flow transformer for 3D shape generation, achieving state-of-the-art fidelity through training on extensive, high-quality data. 2) A hybrid supervised training strategy combining SDF, normal, and eikonal losses for 3D VAE, achieving high- quality 3D reconstruction performance. 3) A data processing pipeline to generate 2 million high- quality 3D samples, highlighting the crucial rules for data quality and quantity in training 3D gen- erative models. Through comprehensive experi- ments, we have validated the effectiveness of each component in our new framework. The seamless integration of these parts has enabled TripoSG to achieve state-of-the-art performance in 3D shape generation. The resulting 3D shapes exhibit en- hanced detail due to high-resolution capabilities and demonstrate exceptional fidelity to input im- ages. Moreover, TripoSG demonstrates improved versatility in generating 3D models from diverse image styles and contents, showcasing strong gen- eralization capabilities. To foster progress and innovation in the field of 3D generation, we will make our model publicly available. 

**Abstract (ZH)**: Recent advancements in扩散技术推动了图像和视频生成的质量达到前所未有的水平，大大加速了生成型AI的应用部署。然而，3D形状生成技术至今仍落后于这一进展，受限于3D数据规模的限制、3D数据处理的复杂性以及3D领域高级技术探索的不足。当前的3D形状生成方法在输出质量、泛化能力和输入条件对齐方面面临重大挑战。我们提出了一种名为TripoSG的新颖精简形状扩散范式，能够生成与输入图像具精确对应关系的高保真3D网格。具体而言，我们提出了：1）一种大规模校正流 Transformer，通过大量高质量数据训练实现最先进的保真度；2）一种结合体素距离场(SDF)、法线和ikelon损失的混合监督训练策略，以实现高性能的3D重建；3）一个数据处理管道，生成200万个高质量3D样本，强调了3D生成模型训练中数据质量和数量的关键规则。通过全面的实验，我们验证了新框架中每个组件的有效性。这些组件的无缝集成使TripoSG在3D形状生成方面达到了最先进的性能。生成的3D形状由于具备高分辨率能力而更加细腻，并且与输入图像具有极高的保真度。此外，TripoSG展示了从不同图像风格和内容生成3D模型的增强灵活性，呈现出强大的泛化能力。为了促进3D生成领域的进展和创新，我们将公开发布我们的模型。 

---
# DefTransNet: A Transformer-based Method for Non-Rigid Point Cloud Registration in the Simulation of Soft Tissue Deformation 

**Title (ZH)**: DefTransNet：一种基于变换器的方法，用于软组织变形模拟中的非刚性点云注册 

**Authors**: Sara Monji-Azad, Marvin Kinz, Siddharth Kothari, Robin Khanna, Amrei Carla Mihan, David Maennel, Claudia Scherl, Juergen Hesser  

**Link**: [PDF](https://arxiv.org/pdf/2502.06336)  

**Abstract**: Soft-tissue surgeries, such as tumor resections, are complicated by tissue deformations that can obscure the accurate location and shape of tissues. By representing tissue surfaces as point clouds and applying non-rigid point cloud registration (PCR) methods, surgeons can better understand tissue deformations before, during, and after surgery. Existing non-rigid PCR methods, such as feature-based approaches, struggle with robustness against challenges like noise, outliers, partial data, and large deformations, making accurate point correspondence difficult. Although learning-based PCR methods, particularly Transformer-based approaches, have recently shown promise due to their attention mechanisms for capturing interactions, their robustness remains limited in challenging scenarios. In this paper, we present DefTransNet, a novel end-to-end Transformer-based architecture for non-rigid PCR. DefTransNet is designed to address the key challenges of deformable registration, including large deformations, outliers, noise, and partial data, by inputting source and target point clouds and outputting displacement vector fields. The proposed method incorporates a learnable transformation matrix to enhance robustness to affine transformations, integrates global and local geometric information, and captures long-range dependencies among points using Transformers. We validate our approach on four datasets: ModelNet, SynBench, 4DMatch, and DeformedTissue, using both synthetic and real-world data to demonstrate the generalization of our proposed method. Experimental results demonstrate that DefTransNet outperforms current state-of-the-art registration networks across various challenging conditions. Our code and data are publicly available. 

**Abstract (ZH)**: 基于Transformer的非刚性点云配准网络DefTransNet：解决变形配准关键挑战 

---
# UniDemoir\'e: Towards Universal Image Demoir\'eing with Data Generation and Synthesis 

**Title (ZH)**: UniDemoir\'e: 向通用去moire处理迈进：基于数据生成与合成 

**Authors**: Zemin Yang, Yujing Sun, Xidong Peng, Siu Ming Yiu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.06324)  

**Abstract**: Image demoiréing poses one of the most formidable challenges in image restoration, primarily due to the unpredictable and anisotropic nature of moiré patterns. Limited by the quantity and diversity of training data, current methods tend to overfit to a single moiré domain, resulting in performance degradation for new domains and restricting their robustness in real-world applications. In this paper, we propose a universal image demoiréing solution, UniDemoiré, which has superior generalization capability. Notably, we propose innovative and effective data generation and synthesis methods that can automatically provide vast high-quality moiré images to train a universal demoiréing model. Our extensive experiments demonstrate the cutting-edge performance and broad potential of our approach for generalized image demoiréing. 

**Abstract (ZH)**: 图像去moire化是图像恢复领域面临的最严峻挑战之一，主要由于moire图案的不可预测和各向异性性质。受限于训练数据的数量和多样性，当前方法往往只在一个特定的moire领域内过拟合，导致在新的领域中性能下降，并限制了其在实际应用中的鲁棒性。在本文中，我们提出了一种通用的图像去moire解决方案UniDemoiré，具有优异的泛化能力。特别地，我们提出了创新且有效的数据生成和合成方法，能够自动生成大量的高质量moire图像，用于训练一个通用的去moire模型。广泛的实验结果证明了我们方法在通用图像去moire领域的前沿性能和广泛潜力。 

---
# From Pixels to Components: Eigenvector Masking for Visual Representation Learning 

**Title (ZH)**: 从像素到组件：特征向量掩码在视觉表示学习中的应用 

**Authors**: Alice Bizeul, Thomas Sutter, Alain Ryser, Bernhard Schölkopf, Julius von Kügelgen, Julia E. Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2502.06314)  

**Abstract**: Predicting masked from visible parts of an image is a powerful self-supervised approach for visual representation learning. However, the common practice of masking random patches of pixels exhibits certain failure modes, which can prevent learning meaningful high-level features, as required for downstream tasks. We propose an alternative masking strategy that operates on a suitable transformation of the data rather than on the raw pixels. Specifically, we perform principal component analysis and then randomly mask a subset of components, which accounts for a fixed ratio of the data variance. The learning task then amounts to reconstructing the masked components from the visible ones. Compared to local patches of pixels, the principal components of images carry more global information. We thus posit that predicting masked from visible components involves more high-level features, allowing our masking strategy to extract more useful representations. This is corroborated by our empirical findings which demonstrate improved image classification performance for component over pixel masking. Our method thus constitutes a simple and robust data-driven alternative to traditional masked image modeling approaches. 

**Abstract (ZH)**: 从图像可见部分预测掩蔽部分是视觉表示学习的一种强大自监督方法。然而，随机遮掩像素块的常见做法会表现出某些失败模式，这可能阻碍学习用于下游任务所需的意义深远的高级特征。我们提出了一种替代的遮掩策略，该策略在数据的适当变换上操作，而不是在原始像素上操作。具体来说，我们执行主成分分析，然后随机遮掩部分成分，这些成分占固定比例的数据方差。然后的学习任务是从可见部分重建掩蔽的成分。与像素的局部区域相比，图像的主成分包含更多的全局信息。因此，我们认为从可见成分预测掩蔽成分涉及更多的高级特征，使我们的遮掩策略能够提取更有用的表示。我们的实证结果也证实了这一点，即与像素遮掩相比，组件遮掩在图像分类性能上有所提高。因此，我们的方法构成了传统掩蔽图像建模方法的一种简单且稳健的数据驱动替代方案。 

---
# End-to-End Multi-Microphone Speaker Extraction Using Relative Transfer Functions 

**Title (ZH)**: 使用相对传输函数的端到端多麦克风说话人提取 

**Authors**: Aviad Eisenberg, Sharon Gannot, Shlomo E. Chazan  

**Link**: [PDF](https://arxiv.org/pdf/2502.06285)  

**Abstract**: This paper introduces a multi-microphone method for extracting a desired speaker from a mixture involving multiple speakers and directional noise in a reverberant environment. In this work, we propose leveraging the instantaneous relative transfer function (RTF), estimated from a reference utterance recorded in the same position as the desired source. The effectiveness of the RTF-based spatial cue is compared with direction of arrival (DOA)-based spatial cue and the conventional spectral embedding. Experimental results in challenging acoustic scenarios demonstrate that using spatial cues yields better performance than the spectral-based cue and that the instantaneous RTF outperforms the DOA-based spatial cue. 

**Abstract (ZH)**: 该文介绍了一种在混响环境中利用多麦克风方法从多说话人混合信号及定向噪声中提取目标说话人的技术。在该项工作中，我们提出利用参考语音在同一位置录制的目标声源的瞬时相对传输函数（RTF）进行提取。实验结果表明，使用空间线索优于基于谱特征的线索，且瞬时RTF优于基于到达角度（DOA）的空间线索。 

---
# Towards Efficient and Intelligent Laser Weeding: Method and Dataset for Weed Stem Detection 

**Title (ZH)**: 面向高效智能激光除草：杂草茎检测的方法与数据集 

**Authors**: Dingning Liu, Jinzhe Li, Haoyang Su, Bei Cui, Zhihui Wang, Qingbo Yuan, Wanli Ouyang, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06255)  

**Abstract**: Weed control is a critical challenge in modern agriculture, as weeds compete with crops for essential nutrient resources, significantly reducing crop yield and quality. Traditional weed control methods, including chemical and mechanical approaches, have real-life limitations such as associated environmental impact and efficiency. An emerging yet effective approach is laser weeding, which uses a laser beam as the stem cutter. Although there have been studies that use deep learning in weed recognition, its application in intelligent laser weeding still requires a comprehensive understanding. Thus, this study represents the first empirical investigation of weed recognition for laser weeding. To increase the efficiency of laser beam cut and avoid damaging the crops of interest, the laser beam shall be directly aimed at the weed root. Yet, weed stem detection remains an under-explored problem. We integrate the detection of crop and weed with the localization of weed stem into one end-to-end system. To train and validate the proposed system in a real-life scenario, we curate and construct a high-quality weed stem detection dataset with human annotations. The dataset consists of 7,161 high-resolution pictures collected in the field with annotations of 11,151 instances of weed. Experimental results show that the proposed system improves weeding accuracy by 6.7% and reduces energy cost by 32.3% compared to existing weed recognition systems. 

**Abstract (ZH)**: 激光除草中的杂草识别研究：一种综合作物和杂草检测与杂草茎定位的端到端系统 

---
# Universal Approximation of Visual Autoregressive Transformers 

**Title (ZH)**: 视觉自回归变换器的通用逼近能力 

**Authors**: Yifang Chen, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.06167)  

**Abstract**: We investigate the fundamental limits of transformer-based foundation models, extending our analysis to include Visual Autoregressive (VAR) transformers. VAR represents a big step toward generating images using a novel, scalable, coarse-to-fine ``next-scale prediction'' framework. These models set a new quality bar, outperforming all previous methods, including Diffusion Transformers, while having state-of-the-art performance for image synthesis tasks. Our primary contributions establish that, for single-head VAR transformers with a single self-attention layer and single interpolation layer, the VAR Transformer is universal. From the statistical perspective, we prove that such simple VAR transformers are universal approximators for any image-to-image Lipschitz functions. Furthermore, we demonstrate that flow-based autoregressive transformers inherit similar approximation capabilities. Our results provide important design principles for effective and computationally efficient VAR Transformer strategies that can be used to extend their utility to more sophisticated VAR models in image generation and other related areas. 

**Abstract (ZH)**: 我们探讨基于变换器的基础模型的基本限制，扩展分析以包括视觉自回归（VAR）变换器。VAR 代表了使用新颖的、可扩展的从粗到细“下一级预测”框架生成图像的一大步。这些模型设定了新的质量标准，优于所有先前的方法，包括扩散变换器，并且在图像合成任务上具有先进的性能。我们主要的贡献表明，对于具有单个自注意力层和单个插值层的单头VAR变换器，VAR变换器是通用的。从统计学角度来看，我们证明了这种简单的VAR变换器是任何图像到图像Lipschitz函数的universal approximator。此外，我们还证明了基于流的自回归变换器继承了类似的影响能力。我们的结果提供了重要的设计原则，用于有效的、计算效率高的VAR变换器策略，这些策略可以被用于扩展其在图像生成和其他相关领域的应用。 

---
# Improved YOLOv5s model for key components detection of power transmission lines 

**Title (ZH)**: 改进的YOLOv5s模型在输电线路关键组件检测中的应用 

**Authors**: Chen Chen, Guowu Yuan, Hao Zhou, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.06127)  

**Abstract**: High-voltage transmission lines are located far from the road, resulting in inconvenient inspection work and rising maintenance costs. Intelligent inspection of power transmission lines has become increasingly important. However, subsequent intelligent inspection relies on accurately detecting various key components. Due to the low detection accuracy of key components in transmission line image inspection, this paper proposed an improved object detection model based on the YOLOv5s (You Only Look Once Version 5 Small) model to improve the detection accuracy of key components of transmission lines. According to the characteristics of the power grid inspection image, we first modify the distance measurement in the k-means clustering to improve the anchor matching of the YOLOv5s model. Then, we add the convolutional block attention module (CBAM) attention mechanism to the backbone network to improve accuracy. Finally, we apply the focal loss function to reduce the impact of class imbalance. Our improved method's mAP (mean average precision) reached 98.1%, the precision reached 97.5%, the recall reached 94.4%, and the detection rate reached 84.8 FPS (frames per second). The experimental results show that our improved model improves detection accuracy and has performance advantages over other models. 

**Abstract (ZH)**: 高压输电线路远离道路，导致检修不便且维护成本上升。基于YOLOv5s的智能输电线路检测方法及其应用 

---
# MMGDreamer: Mixed-Modality Graph for Geometry-Controllable 3D Indoor Scene Generation 

**Title (ZH)**: MMGDreamer：混合模态图用于几何可控的3D室内场景生成 

**Authors**: Zhifei Yang, Keyang Lu, Chao Zhang, Jiaxing Qi, Hanqi Jiang, Ruifei Ma, Shenglin Yin, Yifan Xu, Mingzhe Xing, Zhen Xiao, Jieyi Long, Xiangde Liu, Guangyao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2502.05874)  

**Abstract**: Controllable 3D scene generation has extensive applications in virtual reality and interior design, where the generated scenes should exhibit high levels of realism and controllability in terms of geometry. Scene graphs provide a suitable data representation that facilitates these applications. However, current graph-based methods for scene generation are constrained to text-based inputs and exhibit insufficient adaptability to flexible user inputs, hindering the ability to precisely control object geometry. To address this issue, we propose MMGDreamer, a dual-branch diffusion model for scene generation that incorporates a novel Mixed-Modality Graph, visual enhancement module, and relation predictor. The mixed-modality graph allows object nodes to integrate textual and visual modalities, with optional relationships between nodes. It enhances adaptability to flexible user inputs and enables meticulous control over the geometry of objects in the generated scenes. The visual enhancement module enriches the visual fidelity of text-only nodes by constructing visual representations using text embeddings. Furthermore, our relation predictor leverages node representations to infer absent relationships between nodes, resulting in more coherent scene layouts. Extensive experimental results demonstrate that MMGDreamer exhibits superior control of object geometry, achieving state-of-the-art scene generation performance. Project page: this https URL. 

**Abstract (ZH)**: 可控的3D场景生成在虚拟现实和室内设计中有广泛的应用，生成的场景在几何方面应具有高度的真实感和可控性。场景图提供了一种合适的数据表示形式，有助于这些应用。然而，当前基于图的方法在场景生成中仅限于文本输入，并且对灵活的用户输入适应性不足，阻碍了对对象几何细节的精确控制。为了解决这一问题，我们提出MMGDreamer，一种结合了新型混合模态图、视觉增强模块和关系预测器的双分支扩散模型。混合模态图使对象节点能够结合文本和视觉模态，并且节点之间可以有选择地建立关系，增强了对灵活用户输入的适应性，并能够对生成场景中的对象几何进行细致控制。视觉增强模块通过利用文本嵌入构建视觉表示来丰富仅文本节点的视觉保真度。此外，我们的关系预测器利用节点表示来推断节点之间缺失的关系，从而产生更加连贯的场景布局。大量实验结果表明，MMGDreamer 在对象几何控制方面表现出优越性，达到最先进的场景生成性能。项目页面：this https URL。 

---
# EPBC-YOLOv8: An efficient and accurate improved YOLOv8 underwater detector based on an attention mechanism 

**Title (ZH)**: EPBC-YOLOv8：基于注意力机制的高效精准改进YOLOv8水下检测器 

**Authors**: Xing Jiang, Xiting Zhuang, Jisheng Chen, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05788)  

**Abstract**: In this study, we enhance underwater target detection by integrating channel and spatial attention into YOLOv8's backbone, applying Pointwise Convolution in FasterNeXt for the FasterPW model, and leveraging Weighted Concat in a BiFPN-inspired WFPN structure for improved cross-scale connections and robustness. Utilizing CARAFE for refined feature reassembly, our framework addresses underwater image degradation, achieving mAP at 0.5 scores of 76.7 percent and 79.0 percent on URPC2019 and URPC2020 datasets, respectively. These scores are 2.3 percent and 0.7 percent higher than the original YOLOv8, showcasing enhanced precision in detecting marine organisms. 

**Abstract (ZH)**: 本研究通过将通道和空间注意力机制集成到YOLOv8的骨干网中，利用 FasterNeXt 中的点wise 卷积对 FasterPW 模型进行改进，并采用受 BiFPN 启发的 WFPN 架构中的加权拼接来增强跨尺度连接和稳健性。利用 CARAFE 进行精细特征重组，本框架解决了水下图像退化问题，在URPC2019和URPC2020数据集上分别实现了0.5分数下的mAP为76.7%和79.0%，高于原始YOLOv8的2.3%和0.7%，展示了在检测海洋生物方面更高的精度。 

---
# UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control 

**Title (ZH)**: UniDB：通过随机最优控制的统一扩散桥梁框架 

**Authors**: Kaizhen Zhu, Mokai Pan, Yuexin Ma, Yanwei Fu, Jingyi Yu, Jingya Wang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05749)  

**Abstract**: Recent advances in diffusion bridge models leverage Doob's $h$-transform to establish fixed endpoints between distributions, demonstrating promising results in image translation and restoration tasks. However, these approaches frequently produce blurred or excessively smoothed image details and lack a comprehensive theoretical foundation to explain these shortcomings. To address these limitations, we propose UniDB, a unified framework for diffusion bridges based on Stochastic Optimal Control (SOC). UniDB formulates the problem through an SOC-based optimization and derives a closed-form solution for the optimal controller, thereby unifying and generalizing existing diffusion bridge models. We demonstrate that existing diffusion bridges employing Doob's $h$-transform constitute a special case of our framework, emerging when the terminal penalty coefficient in the SOC cost function tends to infinity. By incorporating a tunable terminal penalty coefficient, UniDB achieves an optimal balance between control costs and terminal penalties, substantially improving detail preservation and output quality. Notably, UniDB seamlessly integrates with existing diffusion bridge models, requiring only minimal code modifications. Extensive experiments across diverse image restoration tasks validate the superiority and adaptability of the proposed framework. Our code is available at this https URL. 

**Abstract (ZH)**: 近期基于Doob $h$-变换的扩散桥梁模型进步显著，能够在保持端点固定的条件下连接不同的概率分布，在图像转换和恢复任务中展现出有希望的结果。然而，这些方法经常产生模糊或过度平滑的图像细节，并缺乏全面的理论基础来解释这些不足。为了解决这些限制，我们提出了基于随机最优控制（SOC）的统一框架UniDB。UniDB通过基于SOC的优化问题形式化表述，并推导出最优控制器的闭式解，从而统一并推广了现有的扩散桥梁模型。我们证明，使用Doob $h$-变换的现有扩散桥梁模型可以被视为我们框架的一个特例，在SOC成本函数中的终止惩罚系数趋向无穷大时出现。通过引入可调的终止惩罚系数，UniDB 实现了控制成本与终止惩罚之间的最佳平衡，显著改善了细节保留和输出质量。值得注意的是，UniDB 可无缝集成到现有的扩散桥梁模型中，只需进行少量代码修改。广泛实验表明，我们提出的框架具有优越性和适应性。我们的代码可在以下链接获取。 

---
# 4D VQ-GAN: Synthesising Medical Scans at Any Time Point for Personalised Disease Progression Modelling of Idiopathic Pulmonary Fibrosis 

**Title (ZH)**: 4D VQ-GAN：生成特发性肺纤维化个性化疾病进展建模的任意时间点医学影像合成 

**Authors**: An Zhao, Moucheng Xu, Ahmed H. Shahin, Wim Wuyts, Mark G. Jones, Joseph Jacob, Daniel C. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2502.05713)  

**Abstract**: Understanding the progression trajectories of diseases is crucial for early diagnosis and effective treatment planning. This is especially vital for life-threatening conditions such as Idiopathic Pulmonary Fibrosis (IPF), a chronic, progressive lung disease with a prognosis comparable to many cancers. Computed tomography (CT) imaging has been established as a reliable diagnostic tool for IPF. Accurately predicting future CT scans of early-stage IPF patients can aid in developing better treatment strategies, thereby improving survival outcomes. In this paper, we propose 4D Vector Quantised Generative Adversarial Networks (4D-VQ-GAN), a model capable of generating realistic CT volumes of IPF patients at any time point. The model is trained using a two-stage approach. In the first stage, a 3D-VQ-GAN is trained to reconstruct CT volumes. In the second stage, a Neural Ordinary Differential Equation (ODE) based temporal model is trained to capture the temporal dynamics of the quantised embeddings generated by the encoder in the first stage. We evaluate different configurations of our model for generating longitudinal CT scans and compare the results against ground truth data, both quantitatively and qualitatively. For validation, we conduct survival analysis using imaging biomarkers derived from generated CT scans and achieve a C-index comparable to that of biomarkers derived from the real CT scans. The survival analysis results demonstrate the potential clinical utility inherent to generated longitudinal CT scans, showing that they can reliably predict survival outcomes. 

**Abstract (ZH)**: 理解疾病进展轨迹对于早期诊断和有效治疗规划至关重要。这对于如特发性肺纤维化（IPF）等致命性疾病尤为关键，IPF是一种慢性进行性肺疾病，预后与许多癌症相似。计算机断层扫描（CT）成像已被证明是诊断IPF的一种可靠工具。准确预测早期IPF患者的未来CT扫描有助于制定更好的治疗策略，从而提高生存率。在本文中，我们提出了一种4D向量量化生成对抗网络（4D-VQ-GAN）模型，该模型能够生成任意时间点IPF患者的逼真CT体积。该模型采用两阶段训练方法。第一阶段，使用3D-VQ-GAN训练重建CT体积；第二阶段，使用基于神经常微分方程（ODE）的时间模型训练，以捕捉第一阶段编码器生成的量化嵌入的时间动态变化。我们评估了不同配置的模型用于生成纵向CT扫描，并与真实数据进行对比，从定量和定性两个方面进行了比较。通过使用从生成的CT扫描中提取的成像生物标志物进行生存分析，我们在C指数上达到了与真实CT扫描生物标志物相当的结果。生存分析结果表明生成的纵向CT扫描具有潜在的临床应用价值，表明它们能够可靠地预测生存结局。 

---
# Semantic-Aware Adaptive Video Streaming Using Latent Diffusion Models for Wireless Networks 

**Title (ZH)**: 基于潜在扩散模型的面向无线网络的语义意识自适应视频流传输 

**Authors**: Zijiang Yan, Jianhua Pei, Hongda Wu, Hina Tabassum, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05695)  

**Abstract**: This paper proposes a novel framework for real-time adaptive-bitrate video streaming by integrating latent diffusion models (LDMs) within the FFmpeg techniques. This solution addresses the challenges of high bandwidth usage, storage inefficiencies, and quality of experience (QoE) degradation associated with traditional constant bitrate streaming (CBS) and adaptive bitrate streaming (ABS). The proposed approach leverages LDMs to compress I-frames into a latent space, offering significant storage and semantic transmission savings without sacrificing high visual quality. While it keeps B-frames and P-frames as adjustment metadata to ensure efficient video reconstruction at the user side, the proposed framework is complemented with the most state-of-the-art denoising and video frame interpolation (VFI) techniques. These techniques mitigate semantic ambiguity and restore temporal coherence between frames, even in noisy wireless communication environments. Experimental results demonstrate the proposed method achieves high-quality video streaming with optimized bandwidth usage, outperforming state-of-the-art solutions in terms of QoE and resource efficiency. This work opens new possibilities for scalable real-time video streaming in 5G and future post-5G networks. 

**Abstract (ZH)**: 本文提出了一种将潜在扩散模型（LDMs）整合到FFmpeg技术中的新颖框架，以实现实时自适应比特率视频流传输。该解决方案解决了传统恒定比特率流传输（CBS）和自适应比特率流传输（ABS）中高带宽使用、存储效率低下及用户体验（QoE）下降的挑战。所提出的方法利用LDMs将I-帧压缩到潜在空间中，同时不牺牲高质量视觉效果，节省了显著的存储和语义传输空间。该方法保持了B-帧和P-帧作为调整元数据，以确保在用户端高效重建视频。与此同时，该框架结合了最先进的去噪和视频帧插补（VFI）技术，这些技术在嘈杂的无线通信环境中降低了语义模糊性并恢复了帧间的时态一致性。实验结果表明，所提出的方法实现了高质量视频流传输，并在QoE和资源效率方面优于现有最先进的解决方案。本文为5G及未来后5G网络中的可扩展实时视频流传输打开了新的可能性。 

---
# Event Stream-based Visual Object Tracking: HDETrack V2 and A High-Definition Benchmark 

**Title (ZH)**: 基于事件流的视觉目标跟踪：HDETrack V2和高清晰度基准 

**Authors**: Shiao Wang, Xiao Wang, Chao Wang, Liye Jin, Lin Zhu, Bo Jiang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05574)  

**Abstract**: We then introduce a novel hierarchical knowledge distillation strategy that incorporates the similarity matrix, feature representation, and response map-based distillation to guide the learning of the student Transformer network. We also enhance the model's ability to capture temporal dependencies by applying the temporal Fourier transform to establish temporal relationships between video frames. We adapt the network model to specific target objects during testing via a newly proposed test-time tuning strategy to achieve high performance and flexibility in target tracking. Recognizing the limitations of existing event-based tracking datasets, which are predominantly low-resolution, we propose EventVOT, the first large-scale high-resolution event-based tracking dataset. It comprises 1141 videos spanning diverse categories such as pedestrians, vehicles, UAVs, ping pong, etc. Extensive experiments on both low-resolution (FE240hz, VisEvent, FELT), and our newly proposed high-resolution EventVOT dataset fully validated the effectiveness of our proposed method. Both the benchmark dataset and source code have been released on this https URL 

**Abstract (ZH)**: 我们引入了一种新颖的分层知识蒸馏策略，该策略结合了相似矩阵、特征表示和基于响应图的蒸馏来指导学生Transformer网络的学习。我们还通过应用时域傅里叶变换来增强模型捕捉时间依赖性的能力，以在视频帧之间建立时间关系。我们提出了一种新的测试时调优策略，使网络模型在测试期间适应特定的目标对象，从而实现目标跟踪的高性能和灵活性。鉴于现有事件驱动跟踪数据集的主要局限性（大部分为低分辨率），我们提出了EventVOT，这是首个大规模高分辨率事件驱动跟踪数据集，包含1141个跨不同类别（如行人、车辆、无人机、乒乓球等）的视频。在低分辨率（FE240Hz、VisEvent、FELT）和我们新提出的高分辨率EventVOT数据集上的广泛实验全面验证了我们提出方法的有效性。基准数据集及源代码已发布于此<https://>。 

---
# A Physical Coherence Benchmark for Evaluating Video Generation Models via Optical Flow-guided Frame Prediction 

**Title (ZH)**: 基于光学流引导帧预测的视频生成模型物理一致性评估基准 

**Authors**: Yongfan Chen, Xiuwen Zhu, Tianyu Li, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05503)  

**Abstract**: Recent advances in video generation models demonstrate their potential as world simulators, but they often struggle with videos deviating from physical laws, a key concern overlooked by most text-to-video benchmarks. We introduce a benchmark designed specifically to assess the Physical Coherence of generated videos, PhyCoBench. Our benchmark includes 120 prompts covering 7 categories of physical principles, capturing key physical laws observable in video content. We evaluated four state-of-the-art (SoTA) T2V models on PhyCoBench and conducted manual assessments. Additionally, we propose an automated evaluation model: PhyCoPredictor, a diffusion model that generates optical flow and video frames in a cascade manner. Through a consistency evaluation comparing automated and manual sorting, the experimental results show that PhyCoPredictor currently aligns most closely with human evaluation. Therefore, it can effectively evaluate the physical coherence of videos, providing insights for future model optimization. Our benchmark, which includes physical coherence prompts, automatic evaluation tool PhyCoPredictor, and generated video dataset, will all be released on GitHub shortly. 

**Abstract (ZH)**: Recent Advances in Video Generation Models Demonstrate Their Potential as World Simulators, but Often Struggle with Physical Law Compliance: Introducing PhyCoBench for Assessing Physical Coherence 

---
# DCENWCNet: A Deep CNN Ensemble Network for White Blood Cell Classification with LIME-Based Explainability 

**Title (ZH)**: DCENWCNet：一种基于LIME解释性的深卷积神经网络集成模型用于白细胞分类 

**Authors**: Sibasish Dhibar  

**Link**: [PDF](https://arxiv.org/pdf/2502.05459)  

**Abstract**: White blood cells (WBC) are important parts of our immune system, and they protect our body against infections by eliminating viruses, bacteria, parasites and fungi. The number of WBC types and the total number of WBCs provide important information about our health status. A traditional method, convolutional neural networks (CNN), a deep learning architecture, can classify the blood cell from a part of an object and perform object recognition. Various CNN models exhibit potential; however, their development often involves ad-hoc processes that neglect unnecessary layers, leading to issues with unbalanced datasets and insufficient data augmentation. To address these challenges, we propose a novel ensemble approach that integrates three CNN architectures, each uniquely configured with different dropout and max-pooling layer settings to enhance feature learning. This ensemble model, named DCENWCNet, effectively balances the bias-variance trade-off. When evaluated on the widely recognized Rabbin-WBC dataset, our model outperforms existing state-of-the-art networks, achieving highest mean accuracy. Additionally, it demonstrates superior performance in precision, recall, F1-score, and Area Under the ROC Curve (AUC) across all categories. To delve deeper into the interpretability of classifiers, we employ reliable post-hoc explanation techniques, including Local Interpretable Model-Agnostic Explanations (LIME). These methods approximate the behavior of a black-box model by elucidating the relationships between feature values and predictions. Interpretable results enable users to comprehend and validate the model's predictions, thereby increasing their confidence in the automated diagnosis. 

**Abstract (ZH)**: 白细胞（WBC）是免疫系统的重要组成部分，它们通过消除病毒、细菌、寄生虫和真菌来保护我们的身体免受感染。白细胞类型的数量和总数量提供了关于我们健康状态的重要信息。传统的卷积神经网络（CNN）是一种深度学习架构，可以对物体的一部分进行血液细胞分类并且执行对象识别。各种CNN模型展现出潜力，但其开发往往涉及缺乏系统性的过程，导致数据集不平衡和数据增强不足的问题。为解决这些挑战，我们提出了一种新颖的集成方法，该方法将三种不同配置的CNN架构集成在一起，每种架构具有不同的dropout和最大池化层设置，以增强特征学习。该集成模型名为DCENWCNet，有效地平衡了偏差-方差Trade-off。当在广泛认可的Rabbin-WBC数据集上进行评估时，我们的模型优于现有最先进的网络，实现了最高的平均准确性。此外，该模型在精确度、召回率、F1分数以及ROC曲线下面积（AUC）方面在所有类别中都表现出色。为了深入探索分类器的可解释性，我们采用可靠的后验解释技术，包括局部可解释的模型无关解释（LIME），这些方法通过阐明特征值与预测之间的关系来近似黑盒模型的行为。可解释的结果使用户能够理解和验证模型的预测，从而增加他们对自动化诊断的信心。 

---
# Convolutional Deep Colorization for Image Compression: A Color Grid Based Approach 

**Title (ZH)**: 基于颜色网格的卷积深度着色方法及其在图像压缩中的应用 

**Authors**: Ian Tassin, Kristen Goebel, Brittany Lasher  

**Link**: [PDF](https://arxiv.org/pdf/2502.05402)  

**Abstract**: The search for image compression optimization techniques is a topic of constant interest both in and out of academic circles. One method that shows promise toward future improvements in this field is image colorization since image colorization algorithms can reduce the amount of color data that needs to be stored for an image. Our work focuses on optimizing a color grid based approach to fully-automated image color information retention with regard to convolutional colorization network architecture for the purposes of image compression. More generally, using a convolutional neural network for image re-colorization, we want to minimize the amount of color information that is stored while still being able to faithfully re-color images. Our results yielded a promising image compression ratio, while still allowing for successful image recolorization reaching high CSIM values. 

**Abstract (ZH)**: 图像压缩优化技术的研究一直是学术界和业界持续关注的话题。一种对未来该领域改进显示出潜力的方法是图像着色，因为图像着色算法可以减少需要存储的颜色数据量。我们的工作集中在优化基于颜色网格的全自动化图像着色方法，并针对卷积着色网络架构进行图像压缩。更一般地说，我们使用卷积神经网络进行图像重新着色，目的是在能够忠实重新着色图像的同时，尽量减少存储的颜色信息量。我们的结果显示，实现了有希望的图像压缩比，同时仍能成功进行图像着色，达到较高的CSIM值。 

---
# Coarse-to-Fine Structure-Aware Artistic Style Transfer 

**Title (ZH)**: 从粗到细结构感知的 artistic 风格转移 

**Authors**: Kunxiao Liu, Guowu Yuan, Hao Wu, Wenhua Qian  

**Link**: [PDF](https://arxiv.org/pdf/2502.05387)  

**Abstract**: Artistic style transfer aims to use a style image and a content image to synthesize a target image that retains the same artistic expression as the style image while preserving the basic content of the content image. Many recently proposed style transfer methods have a common problem; that is, they simply transfer the texture and color of the style image to the global structure of the content image. As a result, the content image has a local structure that is not similar to the local structure of the style image. In this paper, we present an effective method that can be used to transfer style patterns while fusing the local style structure into the local content structure. In our method, dif-ferent levels of coarse stylized features are first reconstructed at low resolution using a Coarse Network, in which style color distribution is roughly transferred, and the content structure is combined with the style structure. Then, the reconstructed features and the content features are adopted to synthesize high-quality structure-aware stylized images with high resolution using a Fine Network with three structural selective fusion (SSF) modules. The effectiveness of our method is demonstrated through the generation of appealing high-quality stylization results and a com-parison with some state-of-the-art style transfer methods. 

**Abstract (ZH)**: 艺术风格迁移旨在使用风格图像和内容图像合成一个目标图像，该目标图像保留与风格图像相同的艺术表现力，同时保留内容图像的基本内容。许多最近提出的方法都存在一个共同问题，即它们简单地将风格图像的纹理和颜色转移到内容图像的全局结构上。结果，内容图像具有与风格图像局部结构不相似的局部结构。在本文中，我们提出了一种有效的方法，可以在融合局部风格结构到局部内容结构的同时转移风格模式。在我们的方法中，首先使用粗网络以低分辨率重构不同层次的粗略风格化特征，在此过程中风格色彩分布被大致转移，并结合内容结构与风格结构。然后，使用包含三个结构选择性融合（SSF）模块的细网络采用重构的特征和内容特征合成高质量的结构意识风格化图像。通过生成引人注目的高质量风格化结果并与一些先进风格迁移方法进行比较，我们展示了我们方法的有效性。 

---
# Multi-Class Segmentation of Aortic Branches and Zones in Computed Tomography Angiography: The AortaSeg24 Challenge 

**Title (ZH)**: Aortic 分支和区域在计算机断层血管成像中的多类分割：AortaSeg24 挑战赛 

**Authors**: Muhammad Imran, Jonathan R. Krebs, Vishal Balaji Sivaraman, Teng Zhang, Amarjeet Kumar, Walker R. Ueland, Michael J. Fassler, Jinlong Huang, Xiao Sun, Lisheng Wang, Pengcheng Shi, Maximilian Rokuss, Michael Baumgartner, Yannick Kirchhof, Klaus H. Maier-Hein, Fabian Isensee, Shuolin Liu, Bing Han, Bong Thanh Nguyen, Dong-jin Shin, Park Ji-Woo, Mathew Choi, Kwang-Hyun Uhm, Sung-Jea Ko, Chanwoong Lee, Jaehee Chun, Jin Sung Kim, Minghui Zhang, Hanxiao Zhang, Xin You, Yun Gu, Zhaohong Pan, Xuan Liu, Xiaokun Liang, Markus Tiefenthaler, Enrique Almar-Munoz, Matthias Schwab, Mikhail Kotyushev, Rostislav Epifanov, Marek Wodzinski, Henning Muller, Abdul Qayyum, Moona Mazher, Steven A. Niederer, Zhiwei Wang, Kaixiang Yang, Jintao Ren, Stine Sofia Korreman, Yuchong Gao, Hongye Zeng, Haoyu Zheng, Rui Zheng, Jinghua Yue, Fugen Zhou, Bo Liu, Alexander Cosman, Muxuan Liang, Chang Zhao, Gilbert R. Upchurch Jr., Jun Ma, Yuyin Zhou, Michol A. Cooper, Wei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05330)  

**Abstract**: Multi-class segmentation of the aorta in computed tomography angiography (CTA) scans is essential for diagnosing and planning complex endovascular treatments for patients with aortic dissections. However, existing methods reduce aortic segmentation to a binary problem, limiting their ability to measure diameters across different branches and zones. Furthermore, no open-source dataset is currently available to support the development of multi-class aortic segmentation methods. To address this gap, we organized the AortaSeg24 MICCAI Challenge, introducing the first dataset of 100 CTA volumes annotated for 23 clinically relevant aortic branches and zones. This dataset was designed to facilitate both model development and validation. The challenge attracted 121 teams worldwide, with participants leveraging state-of-the-art frameworks such as nnU-Net and exploring novel techniques, including cascaded models, data augmentation strategies, and custom loss functions. We evaluated the submitted algorithms using the Dice Similarity Coefficient (DSC) and Normalized Surface Distance (NSD), highlighting the approaches adopted by the top five performing teams. This paper presents the challenge design, dataset details, evaluation metrics, and an in-depth analysis of the top-performing algorithms. The annotated dataset, evaluation code, and implementations of the leading methods are publicly available to support further research. All resources can be accessed at this https URL. 

**Abstract (ZH)**: CTA扫描中主动脉多类分割对于诊断和规划主动脉夹层患者的复杂经血管治疗至关重要。然而，现有方法将主动脉分割简化为二元问题，限制了其在不同分支和区域测量直径的能力。此外，目前没有开源数据集支持多类主动脉分割方法的发展。为解决这一问题，我们组织了AortaSeg24 MICCAI挑战，引入了第一个包含100个标注了23个临床相关主动脉分支和区域的CTA体积的数据集。该数据集旨在促进模型的开发和验证。来自全球的121支队伍参加了挑战，参赛者利用了最先进的框架如nnU-Net，并探索了诸如级联模型、数据增强策略和自定义损失函数等新技术。我们使用Dice相似性系数（DSC）和归一化表面距离（NSD）评估提交的算法，并强调了前五名队伍所采用的方法。本文介绍了挑战设计、数据集详情、评估指标及对表现最佳算法的深入分析。标注数据集、评估代码及领先方法的实现均公开发布，以支持进一步研究。所有资源可访问此URL：[此 https URL]。 

---
# Drone Detection and Tracking with YOLO and a Rule-based Method 

**Title (ZH)**: 基于YOLO和基于规则的方法的无人机检测与跟踪 

**Authors**: Purbaditya Bhattacharya, Patrick Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2502.05292)  

**Abstract**: Drones or unmanned aerial vehicles are traditionally used for military missions, warfare, and espionage. However, the usage of drones has significantly increased due to multiple industrial applications involving security and inspection, transportation, research purposes, and recreational drone flying. Such an increased volume of drone activity in public spaces requires regulatory actions for purposes of privacy protection and safety. Hence, detection of illegal drone activities such as boundary encroachment becomes a necessity. Such detection tasks are usually automated and performed by deep learning models which are trained on annotated image datasets. This paper builds on a previous work and extends an already published open source dataset. A description and analysis of the entire dataset is provided. The dataset is used to train the YOLOv7 deep learning model and some of its minor variants and the results are provided. Since the detection models are based on a single image input, a simple cross-correlation based tracker is used to reduce detection drops and improve tracking performance in videos. Finally, the entire drone detection system is summarized. 

**Abstract (ZH)**: 无人机或无人驾驶航空器传统上用于军事任务、战争和谍报。然而，由于涉及安全、检查、交通、研究目的和 recreational 无人机飞行的多种工业应用，无人机的使用量显著增加。随着公共空间内无人机活动量的增加，需要采取监管措施以保护隐私和确保安全。因此，检测非法无人机活动（如越界）变得必要。这些检测任务通常被自动化，并由在注释图像数据集上训练的深度学习模型执行。本文在此前工作的基础上，扩展了一个已发布的开源数据集，并提供了整个数据集的描述和分析。数据集用于训练 YOLOv7 深度学习模型及其一些minor变体，并提供了结果。由于检测模型基于单张图像输入，使用了基于简单相关系数的跟踪器来减少视频中的检测失误，提高跟踪性能。最后，总结了整个无人机检测系统。 

---
# Homeomorphism Prior for False Positive and Negative Problem in Medical Image Dense Contrastive Representation Learning 

**Title (ZH)**: 医学图像密集对比表示学习中的假阳性与假阴性问题的同胚先验 

**Authors**: Yuting He, Boyu Wang, Rongjun Ge, Yang Chen, Guanyu Yang, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.05282)  

**Abstract**: Dense contrastive representation learning (DCRL) has greatly improved the learning efficiency for image-dense prediction tasks, showing its great potential to reduce the large costs of medical image collection and dense annotation. However, the properties of medical images make unreliable correspondence discovery, bringing an open problem of large-scale false positive and negative (FP&N) pairs in DCRL. In this paper, we propose GEoMetric vIsual deNse sImilarity (GEMINI) learning which embeds the homeomorphism prior to DCRL and enables a reliable correspondence discovery for effective dense contrast. We propose a deformable homeomorphism learning (DHL) which models the homeomorphism of medical images and learns to estimate a deformable mapping to predict the pixels' correspondence under topological preservation. It effectively reduces the searching space of pairing and drives an implicit and soft learning of negative pairs via a gradient. We also propose a geometric semantic similarity (GSS) which extracts semantic information in features to measure the alignment degree for the correspondence learning. It will promote the learning efficiency and performance of deformation, constructing positive pairs reliably. We implement two practical variants on two typical representation learning tasks in our experiments. Our promising results on seven datasets which outperform the existing methods show our great superiority. We will release our code on a companion link: this https URL. 

**Abstract (ZH)**: 密集对比表示学习的几何语义相似性学习（GEMINI）：嵌入同胚先验以提高可靠的对应发现 

---
# Self-supervised Domain Adaptation for Breaking the Limits of Low-quality Fundus Image Quality Enhancement 

**Title (ZH)**: 自我监督领域适应以突破低质量眼底图像质量增强的限制 

**Authors**: Qingshan Hou, Peng Cao, Jiaqi Wang, Xiaoli Liu, Jinzhu Yang, Osmar R. Zaiane  

**Link**: [PDF](https://arxiv.org/pdf/2301.06943)  

**Abstract**: Retinal fundus images have been applied for the diagnosis and screening of eye diseases, such as Diabetic Retinopathy (DR) or Diabetic Macular Edema (DME). However, both low-quality fundus images and style inconsistency potentially increase uncertainty in the diagnosis of fundus disease and even lead to misdiagnosis by ophthalmologists. Most of the existing image enhancement methods mainly focus on improving the image quality by leveraging the guidance of high-quality images, which is difficult to be collected in medical applications. In this paper, we tackle image quality enhancement in a fully unsupervised setting, i.e., neither paired images nor high-quality images. To this end, we explore the potential of the self-supervised task for improving the quality of fundus images without the requirement of high-quality reference images. Specifically, we construct multiple patch-wise domains via an auxiliary pre-trained quality assessment network and a style clustering. To achieve robust low-quality image enhancement and address style inconsistency, we formulate two self-supervised domain adaptation tasks to disentangle the features of image content, low-quality factor and style information by exploring intrinsic supervision signals within the low-quality images. Extensive experiments are conducted on EyeQ and Messidor datasets, and results show that our DASQE method achieves new state-of-the-art performance when only low-quality images are available. 

**Abstract (ZH)**: Retinal fundus图像在眼科疾病诊断和筛查中的应用，如糖尿病视网膜病变(DR)或糖尿病黄斑水肿(DME)，但由于低质量的视网膜图像和风格不一致性可能会增加视网膜疾病诊断的不确定性，甚至导致眼科医生误诊。现有的大多数图像增强方法主要通过高质量图像的指导来提高图像质量，但在医疗应用中难以收集高质量图像。本文在无监督环境下解决图像质量增强问题，即既没有配对图像，也没有高质量图像。为此，我们探索自监督任务在无需高质量参考图像的情况下提高视网膜图像质量的潜力。具体地，我们通过辅助预训练的质量评估网络和风格聚类构建了多个 patches 级别的域。为了实现鲁棒的低质量图像增强并解决风格不一致性问题，我们制定了两个自监督域适应任务，通过探索低质量图像内的内在监督信号，将图像内容、低质量因素和风格信息分离。在EyeQ和Messidor数据集上进行的广泛实验表明，当仅有低质量图像可用时，我们的DASQE方法达到了新的性能最佳水平。 

---
