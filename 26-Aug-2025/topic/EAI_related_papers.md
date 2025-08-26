# FlowVLA: Thinking in Motion with a Visual Chain of Thought 

**Title (ZH)**: FlowVLA: 在视觉链思过程中思考运动 

**Authors**: Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Wenxuan Song, Jiayi Chen, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.18269)  

**Abstract**: Many Vision-Language-Action (VLA) models rely on an internal world model trained via next-frame prediction. This approach, however, struggles with physical reasoning as it entangles static appearance with dynamic motion, often resulting in implausible visual forecasts and inefficient policy learning. To address these limitations, we introduce the Visual Chain of Thought (Visual CoT): a pre-training framework that encourages a model to reason about how a scene evolves before predicting what it will look like. We instantiate this principle in FlowVLA, which predicts a future frame ($v_{t+1}$) only after generating an intermediate optical flow representation ($f_t$) that encodes motion dynamics. This ``$v_t \rightarrow f_t \rightarrow v_{t+1}$'' reasoning process is implemented within a single autoregressive Transformer, guiding the model to learn disentangled dynamics. As a result, FlowVLA produces coherent visual predictions and facilitates more efficient policy learning. Experiments on challenging robotics manipulation benchmarks demonstrate state-of-the-art performance with substantially improved sample efficiency, pointing toward a more principled foundation for world modeling. Project page: this https URL 

**Abstract (ZH)**: 许多视觉-语言-动作（VLA）模型依赖于通过下一帧预测训练的内部世界模型。然而，这种方法在物理推理方面存在困难，因为它将静态外观与动态运动纠缠在一起，往往导致不现实的视觉预测和低效的策略学习。为了解决这些问题，我们引入了视觉链式思维（Visual CoT）：一种预训练框架，鼓励模型在预测未来视觉状态之前先推断场景的演变过程。我们以此原则为基础，在FlowVLA中通过生成一种中间的光学流表示（$f_t$）来预测未来帧（$v_{t+1}$），其中$ f_t $编码了运动动态。这一“$v_t \rightarrow f_t \rightarrow v_{t+1$”推理过程在单一的自回归Transformer中实现，引导模型学习解纠缠的动力学。因此，FlowVLA产生了连贯的视觉预测并促进了更高效的策略学习。实验结果表明，FlowVLA在具有挑战性的机器人操控基准测试中表现出最先进的性能，并且显著提高了样本效率，朝着更为原则的世界建模奠定了基础。项目页面: [这里](this https URL)。 

---
# SafeBimanual: Diffusion-based Trajectory Optimization for Safe Bimanual Manipulation 

**Title (ZH)**: SafeBimanual：基于扩散的轨迹优化以实现安全双臂操作 

**Authors**: Haoyuan Deng, Wenkai Guo, Qianzhun Wang, Zhenyu Wu, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18268)  

**Abstract**: Bimanual manipulation has been widely applied in household services and manufacturing, which enables the complex task completion with coordination requirements. Recent diffusion-based policy learning approaches have achieved promising performance in modeling action distributions for bimanual manipulation. However, they ignored the physical safety constraints of bimanual manipulation, which leads to the dangerous behaviors with damage to robots and objects. To this end, we propose a test-time trajectory optimization framework named SafeBimanual for any pre-trained diffusion-based bimanual manipulation policies, which imposes the safety constraints on bimanual actions to avoid dangerous robot behaviors with improved success rate. Specifically, we design diverse cost functions for safety constraints in different dual-arm cooperation patterns including avoidance of tearing objects and collision between arms and objects, which optimizes the manipulator trajectories with guided sampling of diffusion denoising process. Moreover, we employ a vision-language model (VLM) to schedule the cost functions by specifying keypoints and corresponding pairwise relationship, so that the optimal safety constraint is dynamically generated in the entire bimanual manipulation process. SafeBimanual demonstrates superiority on 8 simulated tasks in RoboTwin with a 13.7% increase in success rate and a 18.8% reduction in unsafe interactions over state-of-the-art diffusion-based methods. Extensive experiments on 4 real-world tasks further verify its practical value by improving the success rate by 32.5%. 

**Abstract (ZH)**: 双臂操作已广泛应用于家庭服务和制造业，能够通过协调完成复杂的任务。基于扩散的策略学习方法在模拟双臂操作的动作分布建模方面取得了显著的成果。然而，这些方法忽略了双臂操作的物理安全约束，导致了可能对机器人和物体造成损害的危险行为。为此，我们提出了一种名为SafeBimanual的测试时轨迹优化框架，适用于任何预训练的基于扩散的双臂操作策略，通过施加安全约束来避免危险行为并提高成功率。具体而言，我们在不同的双臂合作模式下设计了多样化的成本函数，包括避免物体损坏和臂与物体的碰撞，这些成本函数通过指导扩散去噪过程的采样来优化操作轨迹。此外，我们采用视觉语言模型（VLM）通过指定关键点及其对应关系来安排成本函数，从而在整个双臂操作过程中动态生成最优的安全约束。SafeBimanual在RoboTwin中8个仿真任务上的成功率提高了13.7%，不安全交互降低了18.8%，优于现有最佳的扩散基方法。在4个实际任务上的广泛实验进一步验证了其实际价值，成功率提高了32.5%。 

---
# Arnold: a generalist muscle transformer policy 

**Title (ZH)**: Arnold: 通用肌肉变换器策略 

**Authors**: Alberto Silvio Chiappa, Boshi An, Merkourios Simos, Chengkun Li, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2508.18066)  

**Abstract**: Controlling high-dimensional and nonlinear musculoskeletal models of the human body is a foundational scientific challenge. Recent machine learning breakthroughs have heralded policies that master individual skills like reaching, object manipulation and locomotion in musculoskeletal systems with many degrees of freedom. However, these agents are merely "specialists", achieving high performance for a single skill. In this work, we develop Arnold, a generalist policy that masters multiple tasks and embodiments. Arnold combines behavior cloning and fine-tuning with PPO to achieve expert or super-expert performance in 14 challenging control tasks from dexterous object manipulation to locomotion. A key innovation is Arnold's sensorimotor vocabulary, a compositional representation of the semantics of heterogeneous sensory modalities, objectives, and actuators. Arnold leverages this vocabulary via a transformer architecture to deal with the variable observation and action spaces of each task. This framework supports efficient multi-task, multi-embodiment learning and facilitates rapid adaptation to novel tasks. Finally, we analyze Arnold to provide insights into biological motor control, corroborating recent findings on the limited transferability of muscle synergies across tasks. 

**Abstract (ZH)**: 高维度和非线性人体运动学模型的控制是基础科学挑战。近期机器学习突破已经使得能够在多自由度的肌体系统中掌握诸如抓取、物体操作和运动等单个技能。然而，这些智能体仅仅是“专家”，在单一技能上达到高绩效。在这项工作中，我们开发了Arnold，一种能够掌握多种任务和体态的一般主义智能体。Arnold 将行为克隆、微调与PPO结合，以在包括灵巧物体操作到运动在内的14项具有挑战性的控制任务中达到专家或超专家级的性能。一个关键创新是Arnold的感官运动词汇表，这是一种组合式表示异质感觉模态、目标和执行器语义的方法。Arnold 利用这种词汇表通过变压器架构来处理每项任务中多变的观测和动作空间。该框架支持高效的多任务、多体态学习，并促进对新任务的快速适应。最后，我们对Arnold进行分析，以提供对生物运动控制的洞察，证实了最近关于肌肉协同在不同任务间有限转移性发现的合理性。 

---
# A holistic perception system of internal and external monitoring for ground autonomous vehicles: AutoTRUST paradigm 

**Title (ZH)**: 基于内部和外部监测的地面自主车辆全面感知系统：AutoTRUST框架 

**Authors**: Alexandros Gkillas, Christos Anagnostopoulos, Nikos Piperigkos, Dimitris Tsiktsiris, Theofilos Christodoulou, Theofanis Siamatras, Dimitrios Triantafyllou, Christos Basdekis, Theoktisti Marinopoulou, Panagiotis Lepentsiotis, Elefterios Blitsis, Aggeliki Zacharaki, Nearchos Stylianidis, Leonidas Katelaris, Lamberto Salvan, Aris S. Lalos, Christos Laoudias, Antonios Lalas, Konstantinos Votis  

**Link**: [PDF](https://arxiv.org/pdf/2508.17969)  

**Abstract**: This paper introduces a holistic perception system for internal and external monitoring of autonomous vehicles, with the aim of demonstrating a novel AI-leveraged self-adaptive framework of advanced vehicle technologies and solutions that optimize perception and experience on-board. Internal monitoring system relies on a multi-camera setup designed for predicting and identifying driver and occupant behavior through facial recognition, exploiting in addition a large language model as virtual assistant. Moreover, the in-cabin monitoring system includes AI-empowered smart sensors that measure air-quality and perform thermal comfort analysis for efficient on and off-boarding. On the other hand, external monitoring system perceives the surrounding environment of vehicle, through a LiDAR-based cost-efficient semantic segmentation approach, that performs highly accurate and efficient super-resolution on low-quality raw 3D point clouds. The holistic perception framework is developed in the context of EU's Horizon Europe programm AutoTRUST, and has been integrated and deployed on a real electric vehicle provided by ALKE. Experimental validation and evaluation at the integration site of Joint Research Centre at Ispra, Italy, highlights increased performance and efficiency of the modular blocks of the proposed perception architecture. 

**Abstract (ZH)**: 基于人工智能的自主车辆全方位感知系统：一种先进的自我适应框架研究 

---
# Egocentric Instruction-oriented Affordance Prediction via Large Multimodal Model 

**Title (ZH)**: 以自我中心指令为导向的大规模多模态可用性预测 

**Authors**: Bokai Ji, Jie Gu, Xiaokang Ma, Chu Tang, Jingmin Chen, Guangxia Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17922)  

**Abstract**: Affordance is crucial for intelligent robots in the context of object manipulation. In this paper, we argue that affordance should be task-/instruction-dependent, which is overlooked by many previous works. That is, different instructions can lead to different manipulation regions and directions even for the same object. According to this observation, we present a new dataset comprising fifteen thousand object-instruction-affordance triplets. All scenes in the dataset are from an egocentric viewpoint, designed to approximate the perspective of a human-like robot. Furthermore, we investigate how to enable large multimodal models (LMMs) to serve as affordance predictors by implementing a ``search against verifiers'' pipeline. An LMM is asked to progressively predict affordances, with the output at each step being verified by itself during the iterative process, imitating a reasoning process. Experiments show that our method not only unlocks new instruction-oriented affordance prediction capabilities, but also achieves outstanding performance broadly. 

**Abstract (ZH)**: 智能机器人在物体操作情境下，功能感知至关重要。本文认为，功能感知应具有任务/指令依赖性，这是许多先前工作的不足之处。即，即使面对同一个物体，不同的指令也会导致不同的操作区域和方向。基于这一观察，我们提出了一包含十五 thousand 物体-指令-功能感知三元组的新数据集。所有场景均从第一人称视角设计，旨在模拟类人机器人视角。此外，我们探讨了通过实施“搜索对抗验证者”管道，使大规模多模态模型（LMM）能够作为功能感知预测器的可能性。在迭代过程中，LMM 被要求逐步预测功能，并在每一阶段输出被自身验证，模仿推理过程。实验表明，我们的方法不仅解锁了新的以指令为导向的功能感知预测能力，还在广泛的应用中取得了优异的表现。 

---
# Talking to Robots: A Practical Examination of Speech Foundation Models for HRI Applications 

**Title (ZH)**: 与机器人对话：语言基础模型在人机交互应用中的实用性考察 

**Authors**: Theresa Pekarek Rosin, Julia Gachot, Henri-Leon Kordt, Matthias Kerzel, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2508.17753)  

**Abstract**: Automatic Speech Recognition (ASR) systems in real-world settings need to handle imperfect audio, often degraded by hardware limitations or environmental noise, while accommodating diverse user groups. In human-robot interaction (HRI), these challenges intersect to create a uniquely challenging recognition environment. We evaluate four state-of-the-art ASR systems on eight publicly available datasets that capture six dimensions of difficulty: domain-specific, accented, noisy, age-variant, impaired, and spontaneous speech. Our analysis demonstrates significant variations in performance, hallucination tendencies, and inherent biases, despite similar scores on standard benchmarks. These limitations have serious implications for HRI, where recognition errors can interfere with task performance, user trust, and safety. 

**Abstract (ZH)**: 自动语音识别(ASR)系统在实际应用场景中需要处理不Perfect的音频，这些音频常常受到硬件限制或环境噪声的影响，同时还要适应多元化的用户群体。在人机交互(HRI)中，这些挑战交汇在一起形成了一个特别具有挑战性的识别环境。我们评估了四种最先进的ASR系统在八个公开可用的数据集上的性能，这些数据集涵盖了六种难度维度：领域特定的、带方言的、噪声下的、年龄段变化的、受损的以及自发的语音。我们的分析显示，尽管在标准基准上的得分相似，这些系统在性能、错觉倾向和固有偏见方面存在显著差异。这些局限性对HRI有严重的影响，因为识别错误可能会干扰任务性能、用户信任以及安全性。 

---
# MEVITA: Open-Source Bipedal Robot Assembled from E-Commerce Components via Sheet Metal Welding 

**Title (ZH)**: MEVITA：通过板材焊接组装的基于电子商务组件的开源双足机器人 

**Authors**: Kento Kawaharazuka, Shogo Sawaguchi, Ayumu Iwata, Keita Yoneda, Temma Suzuki, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2508.17684)  

**Abstract**: Various bipedal robots have been developed to date, and in recent years, there has been a growing trend toward releasing these robots as open-source platforms. This shift is fostering an environment in which anyone can freely develop bipedal robots and share their knowledge, rather than relying solely on commercial products. However, most existing open-source bipedal robots are designed to be fabricated using 3D printers, which limits their scalability in size and often results in fragile structures. On the other hand, some metal-based bipedal robots have been developed, but they typically involve a large number of components, making assembly difficult, and in some cases, the parts themselves are not readily available through e-commerce platforms. To address these issues, we developed MEVITA, an open-source bipedal robot that can be built entirely from components available via e-commerce. Aiming for the minimal viable configuration for a bipedal robot, we utilized sheet metal welding to integrate complex geometries into single parts, thereby significantly reducing the number of components and enabling easy assembly for anyone. Through reinforcement learning in simulation and Sim-to-Real transfer, we demonstrated robust walking behaviors across various environments, confirming the effectiveness of our approach. All hardware, software, and training environments can be obtained from this https URL . 

**Abstract (ZH)**: 开源电商组件可构建的 bipedal 机器人 MEVITA：基于板材焊接的简约设计与强化学习验证 

---
# GWM: Towards Scalable Gaussian World Models for Robotic Manipulation 

**Title (ZH)**: GWM：面向机器人操作的可扩展高斯世界模型研究 

**Authors**: Guanxing Lu, Baoxiong Jia, Puhao Li, Yixin Chen, Ziwei Wang, Yansong Tang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17600)  

**Abstract**: Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model. 

**Abstract (ZH)**: 基于学习的世界模型内的机器人策略训练正成为趋势，由于现实世界交互效率低下。现有的基于图像的世界模型和策略已经显示出先前的成功，但缺乏一致的三维空间和物理理解所需的稳健的几何信息，即使是基于互联网规模的视频源进行预训练。为此，我们提出了一种新的世界模型分支——高斯世界模型（GWM），用于机器人操作，通过推断机器人动作影响下的高斯原语传播来重建未来状态。其核心是一个潜在扩散变换器（DiT）结合3D变分自编码器，实现了基于高斯散点图的细粒度场景级未来状态重建。GWM不仅可以通过自我监督的未来预测训练增强仿生学习代理的视觉表示，还可以作为神经模拟器支持基于模型的强化学习。模拟和现实世界实验均表明，GWM可以精确预测多样化机器人动作条件下的未来场景，并且可以进一步用于训练表现超越当前最先进的方法的策略，展示了三维世界模型的初步数据规模化潜力。 

---
# LodeStar: Long-horizon Dexterity via Synthetic Data Augmentation from Human Demonstrations 

**Title (ZH)**: LodeStar: 长期灵巧操作通过合成数据增强的人类示范 

**Authors**: Weikang Wan, Jiawei Fu, Xiaodi Yuan, Yifeng Zhu, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17547)  

**Abstract**: Developing robotic systems capable of robustly executing long-horizon manipulation tasks with human-level dexterity is challenging, as such tasks require both physical dexterity and seamless sequencing of manipulation skills while robustly handling environment variations. While imitation learning offers a promising approach, acquiring comprehensive datasets is resource-intensive. In this work, we propose a learning framework and system LodeStar that automatically decomposes task demonstrations into semantically meaningful skills using off-the-shelf foundation models, and generates diverse synthetic demonstration datasets from a few human demos through reinforcement learning. These sim-augmented datasets enable robust skill training, with a Skill Routing Transformer (SRT) policy effectively chaining the learned skills together to execute complex long-horizon manipulation tasks. Experimental evaluations on three challenging real-world long-horizon dexterous manipulation tasks demonstrate that our approach significantly improves task performance and robustness compared to previous baselines. Videos are available at this http URL. 

**Abstract (ZH)**: 开发能够在长时段操作任务中以人类级灵巧性 robust 地执行操作的机器人系统具有挑战性，因为这样的任务不仅需要物理灵巧性，还需要在处理环境变化的同时无缝地组合操作技能。虽然模仿学习提供了一种有前景的方法，但获得全面的数据集是资源密集型的。在这项工作中，我们提出了一种学习框架和系统 LodeStar，该系统使用现成的基础模型自动将任务示例分解为语义上有意义的技能，并通过强化学习从少量的人类示例中生成多样化的合成示例数据集。这些基于模拟的数据集使技能训练更加 robust，Skill Routing Transformer (SRT) 策略能够有效地将学习到的技能串联起来执行复杂的长时段操作任务。在三个具有挑战性的现实世界的长时段灵巧操作任务上的实验评估表明，与先前的方法相比，我们的方法显着提高了任务性能和 robust 性。视频可在以下网址获取。 

---
# Variational Shape Inference for Grasp Diffusion on SE(3) 

**Title (ZH)**: SE(3) 上抓取扩散的变分形状推断 

**Authors**: S. Talha Bukhari, Kaivalya Agrawal, Zachary Kingston, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2508.17482)  

**Abstract**: Grasp synthesis is a fundamental task in robotic manipulation which usually has multiple feasible solutions. Multimodal grasp synthesis seeks to generate diverse sets of stable grasps conditioned on object geometry, making the robust learning of geometric features crucial for success. To address this challenge, we propose a framework for learning multimodal grasp distributions that leverages variational shape inference to enhance robustness against shape noise and measurement sparsity. Our approach first trains a variational autoencoder for shape inference using implicit neural representations, and then uses these learned geometric features to guide a diffusion model for grasp synthesis on the SE(3) manifold. Additionally, we introduce a test-time grasp optimization technique that can be integrated as a plugin to further enhance grasping performance. Experimental results demonstrate that our shape inference for grasp synthesis formulation outperforms state-of-the-art multimodal grasp synthesis methods on the ACRONYM dataset by 6.3%, while demonstrating robustness to deterioration in point cloud density compared to other approaches. Furthermore, our trained model achieves zero-shot transfer to real-world manipulation of household objects, generating 34% more successful grasps than baselines despite measurement noise and point cloud calibration errors. 

**Abstract (ZH)**: 多模态抓取合成是机器人操作中的一个基础任务，通常有多重可行的解决方案。多模态抓取合成旨在基于物体几何形状生成多样化的稳定抓取集合，因此学习几何特征的稳健性对于成功至关重要。为解决这一挑战，我们提出了一个利用变分形状推断的框架，以增强对形状噪声和测量稀疏性的鲁棒性。我们的方法首先使用隐式神经表示训练一个变分自编码器进行形状推断，然后使用这些学习到的几何特征引导SE(3)流形上的抓取合成。此外，我们引入了一种测试时抓取优化技术，该技术可以作为插件进一步增强抓取性能。实验结果表明，我们的抓取合成形状推断方法在ACRONYM数据集上优于最先进的多模态抓取合成方法，性能高出6.3%，并且在点云密度恶化的情况下比其他方法更具有鲁棒性。进一步而言，我们训练的模型实现了对家庭用品的真实世界操作的零样本迁移，尽管存在测量噪声和点云校准误差的情况下，生成的成功抓取数量比基线方法多34%。 

---
# Morphological Cognition: Classifying MNIST Digits Through Morphological Computation Alone 

**Title (ZH)**: 形态认知：仅通过形态计算分类MNIST数字 

**Authors**: Alican Mertan, Nick Cheney  

**Link**: [PDF](https://arxiv.org/pdf/2508.17469)  

**Abstract**: With the rise of modern deep learning, neural networks have become an essential part of virtually every artificial intelligence system, making it difficult even to imagine different models for intelligent behavior. In contrast, nature provides us with many different mechanisms for intelligent behavior, most of which we have yet to replicate. One of such underinvestigated aspects of intelligence is embodiment and the role it plays in intelligent behavior. In this work, we focus on how the simple and fixed behavior of constituent parts of a simulated physical body can result in an emergent behavior that can be classified as cognitive by an outside observer. Specifically, we show how simulated voxels with fixed behaviors can be combined to create a robot such that, when presented with an image of an MNIST digit zero, it moves towards the left; and when it is presented with an image of an MNIST digit one, it moves towards the right. Such robots possess what we refer to as ``morphological cognition'' -- the ability to perform cognitive behavior as a result of morphological processes. To the best of our knowledge, this is the first demonstration of a high-level mental faculty such as image classification performed by a robot without any neural circuitry. We hope that this work serves as a proof-of-concept and fosters further research into different models of intelligence. 

**Abstract (ZH)**: 基于模拟物理体的形态认知：无需神经电路的图像分类示例 

---
# Optimizing Grasping in Legged Robots: A Deep Learning Approach to Loco-Manipulation 

**Title (ZH)**: 基于深度学习的腿部机器人抓取优化：动作- manipulation学习 

**Authors**: Dilermando Almeida, Guilherme Lazzarini, Juliano Negri, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.17466)  

**Abstract**: Quadruped robots have emerged as highly efficient and versatile platforms, excelling in navigating complex and unstructured terrains where traditional wheeled robots might fail. Equipping these robots with manipulator arms unlocks the advanced capability of loco-manipulation to perform complex physical interaction tasks in areas ranging from industrial automation to search-and-rescue missions. However, achieving precise and adaptable grasping in such dynamic scenarios remains a significant challenge, often hindered by the need for extensive real-world calibration and pre-programmed grasp configurations. This paper introduces a deep learning framework designed to enhance the grasping capabilities of quadrupeds equipped with arms, focusing on improved precision and adaptability. Our approach centers on a sim-to-real methodology that minimizes reliance on physical data collection. We developed a pipeline within the Genesis simulation environment to generate a synthetic dataset of grasp attempts on common objects. By simulating thousands of interactions from various perspectives, we created pixel-wise annotated grasp-quality maps to serve as the ground truth for our model. This dataset was used to train a custom CNN with a U-Net-like architecture that processes multi-modal input from an onboard RGB and depth cameras, including RGB images, depth maps, segmentation masks, and surface normal maps. The trained model outputs a grasp-quality heatmap to identify the optimal grasp point. We validated the complete framework on a four-legged robot. The system successfully executed a full loco-manipulation task: autonomously navigating to a target object, perceiving it with its sensors, predicting the optimal grasp pose using our model, and performing a precise grasp. This work proves that leveraging simulated training with advanced sensing offers a scalable and effective solution for object handling. 

**Abstract (ZH)**: 四足机器人的带 manipulator 的精准和适应性抓取：一种基于模拟的深度学习方法 

---
# Evolutionary Brain-Body Co-Optimization Consistently Fails to Select for Morphological Potential 

**Title (ZH)**: 进化性的脑体共优化一致性地未能选择出形态潜力。 

**Authors**: Alican Mertan, Nick Cheney  

**Link**: [PDF](https://arxiv.org/pdf/2508.17464)  

**Abstract**: Brain-body co-optimization remains a challenging problem, despite increasing interest from the community in recent years. To understand and overcome the challenges, we propose exhaustively mapping a morphology-fitness landscape to study it. To this end, we train controllers for each feasible morphology in a design space of 1,305,840 distinct morphologies, constrained by a computational budget. First, we show that this design space constitutes a good model for studying the brain-body co-optimization problem, and our attempt to exhaustively map it roughly captures the landscape. We then proceed to analyze how evolutionary brain-body co-optimization algorithms work in this design space. The complete knowledge of the morphology-fitness landscape facilitates a better understanding of the results of evolutionary brain-body co-optimization algorithms and how they unfold over evolutionary time in the morphology space. This investigation shows that the experimented algorithms cannot consistently find near-optimal solutions. The search, at times, gets stuck on morphologies that are sometimes one mutation away from better morphologies, and the algorithms cannot efficiently track the fitness gradient in the morphology-fitness landscape. We provide evidence that experimented algorithms regularly undervalue the fitness of individuals with newly mutated bodies and, as a result, eliminate promising morphologies throughout evolution. Our work provides the most concrete demonstration of the challenges of evolutionary brain-body co-optimization. Our findings ground the trends in the literature and provide valuable insights for future work. 

**Abstract (ZH)**: 脑体协同优化仍然是一个具有挑战性的问题，尽管近年来该领域引起了越来越多社区的关注。为了了解和克服这些挑战，我们提出全面映射形态-适应度景观以研究该问题。为此，我们在一个包含1,305,840种不同形态的设计空间中，受到计算预算的限制，为每种可行的形态训练控制器。首先，我们展示了该设计空间构成研究脑体协同优化问题的良好模型，我们尝试全面映射它大致捕捉了该景观。然后，我们分析了在该设计空间中进化脑体协同优化算法的工作原理。全面了解形态-适应度景观有助于更好地理解进化脑体协同优化算法的结果及其在形态空间中的演化过程。这项研究显示，所试验的算法无法一致地找到近似最优解。搜索有时会在接近更好形态但仅相差一个突变的形态上停滞，且算法无法有效地追踪形态-适应度景观中的适应度梯度。我们提供了证据表明，所试验的算法经常低估新突变体型个体的适应度，并因此在整个进化过程中淘汰有前途的形态。我们工作提供了进化脑体协同优化挑战的最直接证据。我们的发现为文献中的趋势提供了依据，并为进一步研究提供了宝贵的见解。 

---
# OVITA: Open-Vocabulary Interpretable Trajectory Adaptations 

**Title (ZH)**: OVITA: 开词汇量可解释轨迹适应 

**Authors**: Anurag Maurya, Tashmoy Ghosh, Anh Nguyen, Ravi Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2508.17260)  

**Abstract**: Adapting trajectories to dynamic situations and user preferences is crucial for robot operation in unstructured environments with non-expert users. Natural language enables users to express these adjustments in an interactive manner. We introduce OVITA, an interpretable, open-vocabulary, language-driven framework designed for adapting robot trajectories in dynamic and novel situations based on human instructions. OVITA leverages multiple pre-trained Large Language Models (LLMs) to integrate user commands into trajectories generated by motion planners or those learned through demonstrations. OVITA employs code as an adaptation policy generated by an LLM, enabling users to adjust individual waypoints, thus providing flexible control. Another LLM, which acts as a code explainer, removes the need for expert users, enabling intuitive interactions. The efficacy and significance of the proposed OVITA framework is demonstrated through extensive simulations and real-world environments with diverse tasks involving spatiotemporal variations on heterogeneous robotic platforms such as a KUKA IIWA robot manipulator, Clearpath Jackal ground robot, and CrazyFlie drone. 

**Abstract (ZH)**: 适配动态情况和用户偏好的轨迹调整对于在非结构化环境中由非专家用户操作的机器人至关重要。自然语言使用户能够以交互方式表达这些调整。我们提出了OVITA，一种基于人类指令、可解释的、含开放词汇表的、语言驱动的框架，用于在动态和新颖情况下调整机器人轨迹。OVITA利用多个预训练的大规模语言模型（LLMs）将用户命令整合到路径规划器生成的轨迹或通过示范学习的轨迹中。OVITA采用由LLM生成的代码作为适应策略，允许用户调整单个航点，从而提供灵活控制。另一个LLM作为代码解释器，消除了对专家用户的需求，使交互更加直观。通过广泛的仿真实验和涉及多种任务、异构机器人平台（如KUKA IIWA机械臂、Clearpath Jackal地面机器人和CrazyFlie无人机）的时空变化的实际环境，展示了提出的OVITA框架的有效性和重要性。 

---
# LaGarNet: Goal-Conditioned Recurrent State-Space Models for Pick-and-Place Garment Flattening 

**Title (ZH)**: LaGarNet: 基于目标条件的递归状态空间模型用于衣物整理的拾取与放置 

**Authors**: Halid Abdulrahim Kadi, Kasim Terzić  

**Link**: [PDF](https://arxiv.org/pdf/2508.17070)  

**Abstract**: We present a novel goal-conditioned recurrent state space (GC-RSSM) model capable of learning latent dynamics of pick-and-place garment manipulation. Our proposed method LaGarNet matches the state-of-the-art performance of mesh-based methods, marking the first successful application of state-space models on complex garments. LaGarNet trains on a coverage-alignment reward and a dataset collected through a general procedure supported by a random policy and a diffusion policy learned from few human demonstrations; it substantially reduces the inductive biases introduced in the previous similar methods. We demonstrate that a single-policy LaGarNet achieves flattening on four different types of garments in both real-world and simulation settings. 

**Abstract (ZH)**: 我们提出了一种新颖的基于目标条件的递归状态空间（GC-RSSM）模型，该模型能够学习拾取和放置服装操作的潜在动力学。我们提出的LaGarNet方法匹配了基于网格方法的最先进性能，标志着状态空间模型首次成功应用于复杂服装。LaGarNet通过覆盖对齐奖励在由随机策略和支持的有限人类示范学习的扩散策略支持的一般程序下进行训练，显著降低了先前类似方法引入的归纳偏置。我们展示了单策略的LaGarNet在现实世界和仿真环境中均实现了四种不同类型的服装的平整化。 

---
# HumanoidVerse: A Versatile Humanoid for Vision-Language Guided Multi-Object Rearrangement 

**Title (ZH)**: HumanoidVerse: 适用于视觉-语言引导多物体重排的多功能类人型机器人 

**Authors**: Haozhuo Zhang, Jingkai Sun, Michele Caprio, Jian Tang, Shanghang Zhang, Qiang Zhang, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16943)  

**Abstract**: We introduce HumanoidVerse, a novel framework for vision-language guided humanoid control that enables a single physically simulated robot to perform long-horizon, multi-object rearrangement tasks across diverse scenes. Unlike prior methods that operate in fixed settings with single-object interactions, our approach supports consecutive manipulation of multiple objects, guided only by natural language instructions and egocentric camera RGB observations. HumanoidVerse is trained via a multi-stage curriculum using a dual-teacher distillation pipeline, enabling fluid transitions between sub-tasks without requiring environment resets. To support this, we construct a large-scale dataset comprising 350 multi-object tasks spanning four room layouts. Extensive experiments in the Isaac Gym simulator demonstrate that our method significantly outperforms prior state-of-the-art in both task success rate and spatial precision, and generalizes well to unseen environments and instructions. Our work represents a key step toward robust, general-purpose humanoid agents capable of executing complex, sequential tasks under real-world sensory constraints. The video visualization results can be found on the project page: this https URL. 

**Abstract (ZH)**: 我们介绍了HumanoidVerse：一种新的基于视觉语言引导的人形机器人控制框架，使单个物理模拟机器人能够在多样化场景中执行长时 horizon、多对象重组任务。不同于先前在固定环境中仅支持单一对象交互的方法，我们的方法仅通过自然语言指令和第一人称摄像头RGB观察来支持连续操纵多个对象。HumanoidVerse通过一个多阶段课程采用双教师蒸馏管道进行训练，能够在无需重置环境的情况下流畅地在子任务之间进行转换。为此，我们构建了一个包含350个跨四种房间布局的多对象任务的大规模数据集。在Isaac Gym模拟器中的广泛实验表明，我们的方法在任务成功率和空间精度上都优于先前的最先进方法，并且能够很好地泛化到未见过的环境和指令。我们的工作是朝着具备在现实世界传感约束下执行复杂序列任务的稳健且通用的人形代理迈出的关键一步。该项目的视频可视化结果可在项目页面上找到：this https URL。 

---
# A Workflow for Map Creation in Autonomous Vehicle Simulations 

**Title (ZH)**: 自主车辆仿真中的地图创建工作流 

**Authors**: Zubair Islam, Ahmaad Ansari, George Daoud, Mohamed El-Darieby  

**Link**: [PDF](https://arxiv.org/pdf/2508.16856)  

**Abstract**: The fast development of technology and artificial intelligence has significantly advanced Autonomous Vehicle (AV) research, emphasizing the need for extensive simulation testing. Accurate and adaptable maps are critical in AV development, serving as the foundation for localization, path planning, and scenario testing. However, creating simulation-ready maps is often difficult and resource-intensive, especially with simulators like CARLA (CAR Learning to Act). Many existing workflows require significant computational resources or rely on specific simulators, limiting flexibility for developers. This paper presents a custom workflow to streamline map creation for AV development, demonstrated through the generation of a 3D map of a parking lot at Ontario Tech University. Future work will focus on incorporating SLAM technologies, optimizing the workflow for broader simulator compatibility, and exploring more flexible handling of latitude and longitude values to enhance map generation accuracy. 

**Abstract (ZH)**: 快速发展的技术与人工智能显著推进了自动驾驶车辆（AV）研究，强调了需进行广泛模拟测试的必要性。精确且适应性强的地图对AV开发至关重要，是定位、路径规划和场景测试的基础。然而，创建可用于模拟的地图往往困难且资源密集，特别是在使用如CARLA等模拟器时。现有许多工作流程需要大量计算资源或依赖特定的模拟器，限制了开发者的灵活性。本文提出了一种定制工作流以简化AV开发中的地图创建过程，并通过在滑铁卢大学安大略分校生成一个停车场的3D地图予以展示。未来的工作将侧重于集成SLAM技术、优化工作流以提高对更广泛模拟器的兼容性、以及探索更灵活的纬度和经度值处理方法以提高地图生成精度。 

---
# Physical Embodiment Enables Information Processing Beyond Explicit Sensing in Active Matter 

**Title (ZH)**: 物理体型使活性物质中的信息处理超越显式传感成为可能 

**Authors**: Diptabrata Paul, Nikola Milosevic, Nico Scherf, Frank Cichos  

**Link**: [PDF](https://arxiv.org/pdf/2508.17921)  

**Abstract**: Living microorganisms have evolved dedicated sensory machinery to detect environmental perturbations, processing these signals through biochemical networks to guide behavior. Replicating such capabilities in synthetic active matter remains a fundamental challenge. Here, we demonstrate that synthetic active particles can adapt to hidden hydrodynamic perturbations through physical embodiment alone, without explicit sensing mechanisms. Using reinforcement learning to control self-thermophoretic particles, we show that they learn navigation strategies to counteract unobserved flow fields by exploiting information encoded in their physical dynamics. Remarkably, particles successfully navigate perturbations that are not included in their state inputs, revealing that embodied dynamics can serve as an implicit sensing mechanism. This discovery establishes physical embodiment as a computational resource for information processing in active matter, with implications for autonomous microrobotic systems and bio-inspired computation. 

**Abstract (ZH)**: 活化的微生物已经进化出专门的感测装置来探测环境变化，并通过生物化学网络处理这些信号以指导行为。在合成活性物质中复制这种能力仍然是一个基本挑战。在这里，我们证明合成活性颗粒仅通过物理体现就可以适应隐藏的水动力扰动，而无需明确的感测机制。通过使用强化学习来控制自热泳颗粒，我们展示它们通过利用其物理动力学中编码的信息学习导航策略以对抗未观察到的流场。令人惊讶的是，颗粒成功导航了不在其状态输入中的扰动，表明物理动力学可以作为隐式感测机制。这一发现将物理体现确立为活性物质中信息处理的计算资源，并对自主微机器人系统和生物启发计算具有重要意义。 

---
# SoK: Cybersecurity Assessment of Humanoid Ecosystem 

**Title (ZH)**: SoK: 人类身态生态系统网络安全评估 

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici  

**Link**: [PDF](https://arxiv.org/pdf/2508.17481)  

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics. 

**Abstract (ZH)**: 类人机器人在医疗、工业、国防和服务领域中的实践部署正在不断进步。虽然类人机器人通常被视为网络物理系统（CPS），但它们依赖于传统的网络软件栈（如Linux操作系统）、机器人操作系统（ROS）中间件以及空中更新通道，这些特性为其带来了不同于传统CPS模型的安全特性，并使其暴露于常规CPS模型未能充分解决的漏洞中。之前的研究主要关注特定威胁，如激光雷达欺骗或对抗性机器学习（AML）。这种狭窄的视角忽视了针对一个组件的攻击如何在机器人相互连接的系统中引发连锁反应。我们通过系统知识综合（SoK）研究，采用全面的方法，将来自机器人学、CPS和网络安全部门的零散研究整合起来。我们为类人机器人引入了一个七层安全模型，将39种已知攻击和35种防御措施按硬件到人机交互的全生态系统进行了组织。基于该安全模型，我们开发了一个量化39×35攻击-防御矩阵，并通过蒙特卡洛分析进行了验证。通过评估三个实际机器人（Pepper、G1 EDU和Digit）来展示我们的方法，评分分析显示不同平台的安全部成熟度存在差异，得分范围从39.9%到79.5%不等。该项工作引入了一种结构化、基于证据的评估方法，能够实现系统的安全评估、跨平台基准测试，并指导类人机器人安全投资的优先级确定。 

---
# Fiducial Marker Splatting for High-Fidelity Robotics Simulations 

**Title (ZH)**: 信标标记点绘制用于高保真机器人模拟 

**Authors**: Diram Tabaa, Gianni Di Caro  

**Link**: [PDF](https://arxiv.org/pdf/2508.17012)  

**Abstract**: High-fidelity 3D simulation is critical for training mobile robots, but its traditional reliance on mesh-based representations often struggle in complex environments, such as densely packed greenhouses featuring occlusions and repetitive structures. Recent neural rendering methods, like Gaussian Splatting (GS), achieve remarkable visual realism but lack flexibility to incorporate fiducial markers, which are essential for robotic localization and control. We propose a hybrid framework that combines the photorealism of GS with structured marker representations. Our core contribution is a novel algorithm for efficiently generating GS-based fiducial markers (e.g., AprilTags) within cluttered scenes. Experiments show that our approach outperforms traditional image-fitting techniques in both efficiency and pose-estimation accuracy. We further demonstrate the framework's potential in a greenhouse simulation. This agricultural setting serves as a challenging testbed, as its combination of dense foliage, similar-looking elements, and occlusions pushes the limits of perception, thereby highlighting the framework's value for real-world applications. 

**Abstract (ZH)**: 高保真3D仿真对于训练移动机器人至关重要，但其传统的基于网格的表示形式在复杂环境中往往难以应对，例如包括遮挡和重复结构的密集温室。最近的神经渲染方法，如高斯绘画（GS），实现了令人印象深刻的视觉真实感，但在整合用于机器人定位和控制的特征标记方面缺乏灵活性。我们提出了一种混合框架，将GS的视觉真实感与结构化标记表示相结合。我们的核心贡献是一种新型算法，用于在杂乱场景中高效生成基于GS的特征标记（如AprilTags）。实验表明，我们的方法在效率和姿态估计精度上均优于传统的图像匹配技术。我们进一步展示了该框架在温室仿真中的潜力。这种农业设置作为一个具有挑战性的测试平台，其密集植被、相似元素和遮挡的组合，对感知提出了极限挑战，从而突显了该框架在实际应用中的价值。 

---
# Observations of atypical users from a pilot deployment of a public-space social robot in a church 

**Title (ZH)**: 公共空间社会机器人在教堂试点部署中的非典型用户观察 

**Authors**: Andrew Blair, Peggy Gregory, Mary Ellen Foster  

**Link**: [PDF](https://arxiv.org/pdf/2508.16622)  

**Abstract**: Though a goal of HRI is the natural integration of social robots into everyday public spaces, real-world studies still occur mostly within controlled environments with predetermined participants. True public spaces present an environment which is largely unconstrained and unpredictable, frequented by a diverse range of people whose goals can often conflict with those of the robot. When combined with the general unfamiliarity most people have with social robots, this leads to unexpected human-robot interactions in these public spaces that are rarely discussed or detected in other contexts. In this paper, we describe atypical users we observed interacting with our robot, and those who did not, during a three-day pilot deployment within a large working church and visitor attraction. We then discuss theoretical future advances in the field that could address these challenges, as well as immediate practical mitigations and strategies to help improve public space human-robot interactions in the present. This work contributes empirical insights into the dynamics of human-robot interaction in public environments and offers actionable guidance for more effective future deployments for social robot designers. 

**Abstract (ZH)**: 虽然人机交互的目标是将社会机器人自然地整合到日常公共空间中，但现实世界的研究仍主要在受控环境中进行，参与者事先确定。真正的公共空间提供了一个基本不受限制且难以预测的环境，来往的人群多样，他们的目标有时会与机器人相冲突。当结合大多数人对社会机器人的普遍陌生感时，这导致在这些公共空间中出现了非典型的、难以预料的人机互动，而在其他情况下这些互动往往未被讨论或检测到。在本文中，我们描述了在一大型工作教堂和旅游景点为期三天的试点部署中观察到的非典型用户及其未进行人机互动的情况。随后我们讨论了可以通过理论上的未来发展来应对这些挑战的方法，同时也提出了一些即时的实践缓解措施和策略，以帮助改善当前公共空间中的人机互动。本文提供了有关公共环境中人机互动动态的实证见解，并为社会机器人设计师提供了可操作的指导，以实现更有效的未来部署。 

---
# PerPilot: Personalizing VLM-based Mobile Agents via Memory and Exploration 

**Title (ZH)**: PerPilot: 基于记忆和探索的个性化VLM驱动移动代理 

**Authors**: Xin Wang, Zhiyao Cui, Hao Li, Ya Zeng, Chenxu Wang, Ruiqi Song, Yihang Chen, Kun Shao, Qiaosheng Zhang, Jinzhuo Liu, Siyue Ren, Shuyue Hu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18040)  

**Abstract**: Vision language model (VLM)-based mobile agents show great potential for assisting users in performing instruction-driven tasks. However, these agents typically struggle with personalized instructions -- those containing ambiguous, user-specific context -- a challenge that has been largely overlooked in previous research. In this paper, we define personalized instructions and introduce PerInstruct, a novel human-annotated dataset covering diverse personalized instructions across various mobile scenarios. Furthermore, given the limited personalization capabilities of existing mobile agents, we propose PerPilot, a plug-and-play framework powered by large language models (LLMs) that enables mobile agents to autonomously perceive, understand, and execute personalized user instructions. PerPilot identifies personalized elements and autonomously completes instructions via two complementary approaches: memory-based retrieval and reasoning-based exploration. Experimental results demonstrate that PerPilot effectively handles personalized tasks with minimal user intervention and progressively improves its performance with continued use, underscoring the importance of personalization-aware reasoning for next-generation mobile agents. The dataset and code are available at: this https URL 

**Abstract (ZH)**: 基于视觉语言模型的移动代理在执行个性化指令驱动任务中展现出巨大潜力，然而这些代理通常难以处理包含模糊且用户特定上下文的个性化指令，这一挑战在以往研究中被忽视。本文定义了个性化指令，并介绍了一个名为PerInstruct的新颖人类标注数据集，涵盖各类移动场景中的个性化指令。鉴于现有移动代理的个性化能力有限，我们提出了由大规模语言模型驱动的可插拔框架PerPilot，使移动代理能够自主感知、理解和执行个性化用户指令。PerPilot通过基于记忆的检索和基于推理的探索两种互补方法识别个性化元素并自主完成指令。实验结果表明，PerPilot能够有效地处理个性化任务，并在持续使用中逐步提高性能，强调了个性化感知推理对于下一代移动代理的重要性。数据集和代码可从以下链接获取：this https URL。 

---
# TradingGroup: A Multi-Agent Trading System with Self-Reflection and Data-Synthesis 

**Title (ZH)**: TradingGroup：一种具备自我反省和数据合成能力的多代理交易系统 

**Authors**: Feng Tian, Flora D. Salim, Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.17565)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled powerful agent-based applications in finance, particularly for sentiment analysis, financial report comprehension, and stock forecasting. However, existing systems often lack inter-agent coordination, structured self-reflection, and access to high-quality, domain-specific post-training data such as data from trading activities including both market conditions and agent decisions. These data are crucial for agents to understand the market dynamics, improve the quality of decision-making and promote effective coordination. We introduce TradingGroup, a multi-agent trading system designed to address these limitations through a self-reflective architecture and an end-to-end data-synthesis pipeline. TradingGroup consists of specialized agents for news sentiment analysis, financial report interpretation, stock trend forecasting, trading style adaptation, and a trading decision making agent that merges all signals and style preferences to produce buy, sell or hold decisions. Specifically, we design self-reflection mechanisms for the stock forecasting, style, and decision-making agents to distill past successes and failures for similar reasoning in analogous future scenarios and a dynamic risk-management model to offer configurable dynamic stop-loss and take-profit mechanisms. In addition, TradingGroup embeds an automated data-synthesis and annotation pipeline that generates high-quality post-training data for further improving the agent performance through post-training. Our backtesting experiments across five real-world stock datasets demonstrate TradingGroup's superior performance over rule-based, machine learning, reinforcement learning, and existing LLM-based trading strategies. 

**Abstract (ZH)**: 最近大规模语言模型的进展在金融领域尤其是情感分析、财务报告理解和股票预测中推动了强大代理应用的发展。然而，现有系统往往缺乏代理间的协调、结构化的自我反思以及访问高质量的专业领域后训练数据，如交易活动数据，包括市场条件和代理决策。这些数据对于代理理解市场动态、提高决策质量并促进有效协调至关重要。我们介绍了TradingGroup，这是一种多代理交易平台系统，通过自我反思架构和端到端的数据合成流水线来解决这些限制。TradingGroup包括专门用于新闻情感分析、财务报告解释、股票趋势预测、交易风格适应以及合并所有信号和风格偏好的交易决策代理。具体而言，我们为股票预测、风格和决策代理设计了自我反思机制，以提炼先前成功和失败的经验，用于类似推理的未来场景，并引入了动态风险管理模型，提供可配置的动态止损和止盈机制。此外，TradingGroup嵌入了自动数据合成和注释流水线，生成高质量的后训练数据，进一步提高代理性能。我们的回测实验在五个真实世界的股票数据集中证明了TradingGroup优于基于规则、机器学习、强化学习和现有基于大语言模型的交易策略。 

---
# Evolving Collective Cognition in Human-Agent Hybrid Societies: How Agents Form Stances and Boundaries 

**Title (ZH)**: 人类-代理混合社会中的集体认知演化：代理如何形成立场和边界 

**Authors**: Hanzhong Zhang, Muhua Huang, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17366)  

**Abstract**: Large language models have been widely used to simulate credible human social behaviors. However, it remains unclear whether these models can demonstrate stable capacities for stance formation and identity negotiation in complex interactions, as well as how they respond to human interventions. We propose a computational multi-agent society experiment framework that integrates generative agent-based modeling with virtual ethnographic methods to investigate how group stance differentiation and social boundary formation emerge in human-agent hybrid societies. Across three studies, we find that agents exhibit endogenous stances, independent of their preset identities, and display distinct tonal preferences and response patterns to different discourse strategies. Furthermore, through language interaction, agents actively dismantle existing identity-based power structures and reconstruct self-organized community boundaries based on these stances. Our findings suggest that preset identities do not rigidly determine the agents' social structures. For human researchers to effectively intervene in collective cognition, attention must be paid to the endogenous mechanisms and interactional dynamics within the agents' language networks. These insights provide a theoretical foundation for using generative AI in modeling group social dynamics and studying human-agent collaboration. 

**Abstract (ZH)**: 大型语言模型被广泛用于模拟可信的人类社会行为。然而，尚不清楚这些模型在复杂互动中能否稳定地展现立场形成和身份协商的能力，以及它们如何响应人类干预。我们提出了一种将生成性基于代理 modeling 与虚拟人类学方法相结合的计算多代理社会实验框架，以探究人类-代理混合社会中群体立场分化和社会边界形成如何出现。在三个研究中，我们发现代理表现出内生性立场，不受其预设身份的影响，并对不同的论述策略表现出不同的语调偏好和响应模式。此外，通过语言交互，代理积极瓦解基于身份的既有权力结构，并基于这些立场重新构建自组织社区边界。我们的研究结果表明，预设身份不会刚性地决定代理的社会结构。为了有效干预群体认知，人类研究人员必须关注代理语言网络内部的内生机制和交互动力学。这些见解为使用生成性 AI 模型社会动力学和研究人类-代理协作提供了理论基础。 

---
# MEENA (PersianMMMU): Multimodal-Multilingual Educational Exams for N-level Assessment 

**Title (ZH)**: MEENA (波斯MMMU): 多模态多语言教育考试 for N级评估 

**Authors**: Omid Ghahroodi, Arshia Hemmat, Marzia Nouri, Seyed Mohammad Hadi Hosseini, Doratossadat Dastgheib, Mohammad Vali Sanian, Alireza Sahebi, Reihaneh Zohrabi, Mohammad Hossein Rohban, Ehsaneddin Asgari, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2508.17290)  

**Abstract**: Recent advancements in large vision-language models (VLMs) have primarily focused on English, with limited attention given to other languages. To address this gap, we introduce MEENA (also known as PersianMMMU), the first dataset designed to evaluate Persian VLMs across scientific, reasoning, and human-level understanding tasks. Our dataset comprises approximately 7,500 Persian and 3,000 English questions, covering a wide range of topics such as reasoning, mathematics, physics, diagrams, charts, and Persian art and literature. Key features of MEENA include: (1) diverse subject coverage spanning various educational levels, from primary to upper secondary school, (2) rich metadata, including difficulty levels and descriptive answers, (3) original Persian data that preserves cultural nuances, (4) a bilingual structure to assess cross-linguistic performance, and (5) a series of diverse experiments assessing various capabilities, including overall performance, the model's ability to attend to images, and its tendency to generate hallucinations. We hope this benchmark contributes to enhancing VLM capabilities beyond English. 

**Abstract (ZH)**: 近期大规模视听模型（VLMs）的发展主要集中在英文上，对其他语言的关注度有限。为解决这一问题，我们引入了MEENA（也称为PersianMMMU），这是首个旨在评估波斯语VLMs在科学、推理和人文理解任务方面的数据集。我们的数据集包含约7,500个波斯语和3,000个英文问题，涵盖了推理、数学、物理、图表、文物和文学等多个主题。MEENA的关键特征包括：(1) 范围广泛的内容覆盖，从基础教育到高中教育，(2) 丰富元数据，包括难度级别和描述性答案，(3) 原创的波斯语文本，保留了文化细微差异，(4) 双语结构以评估跨语言性能，以及(5) 一系列多样实验，评估各种能力，包括整体性能、模型对图像的关注能力和生成幻觉的倾向。我们希望这一基准能够促进VLM能力的提升，超越英文语言。 

---
# Federated Reinforcement Learning for Runtime Optimization of AI Applications in Smart Eyewears 

**Title (ZH)**: 智能眼镜中AI应用运行时优化的联邦 reinforcement 学习 

**Authors**: Hamta Sedghani, Abednego Wamuhindo Kambale, Federica Filippini, Francesca Palermo, Diana Trojaniello, Danilo Ardagna  

**Link**: [PDF](https://arxiv.org/pdf/2508.17262)  

**Abstract**: Extended reality technologies are transforming fields such as healthcare, entertainment, and education, with Smart Eye-Wears (SEWs) and Artificial Intelligence (AI) playing a crucial role. However, SEWs face inherent limitations in computational power, memory, and battery life, while offloading computations to external servers is constrained by network conditions and server workload variability. To address these challenges, we propose a Federated Reinforcement Learning (FRL) framework, enabling multiple agents to train collaboratively while preserving data privacy. We implemented synchronous and asynchronous federation strategies, where models are aggregated either at fixed intervals or dynamically based on agent progress. Experimental results show that federated agents exhibit significantly lower performance variability, ensuring greater stability and reliability. These findings underscore the potential of FRL for applications requiring robust real-time AI processing, such as real-time object detection in SEWs. 

**Abstract (ZH)**: 扩展现实技术正在transforming医疗、娱乐和教育等领域，智能眼镜（SEWs）和人工智能（AI）发挥着关键作用。然而，SEWs在计算能力、内存和电池寿命方面存在固有限制，将计算任务卸载到外部服务器又受限于网络条件和服务器工作负载的变异性。为应对这些挑战，我们提出了一种联邦强化学习（FRL）框架，使得多个代理能够协作训练并保护数据隐私。我们实现了同步和异步的联邦策略，模型要么在固定间隔，要么根据代理进度动态聚合。实验结果表明，联邦代理表现出显著更低的性能变异性，从而确保了更高的稳定性和可靠性。这些发现强调了FRL在需要强大实时AI处理的应用，如SEWs中的实时物体检测方面的潜力。 

---
# From reactive to cognitive: brain-inspired spatial intelligence for embodied agents 

**Title (ZH)**: 从反应式到认知式：受脑启发的空间智能对具身代理的应用 

**Authors**: Shouwei Ruan, Liyuan Wang, Caixin Kang, Qihui Zhu, Songming Liu, Xingxing Wei, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17198)  

**Abstract**: Spatial cognition enables adaptive goal-directed behavior by constructing internal models of space. Robust biological systems consolidate spatial knowledge into three interconnected forms: \textit{landmarks} for salient cues, \textit{route knowledge} for movement trajectories, and \textit{survey knowledge} for map-like representations. While recent advances in multi-modal large language models (MLLMs) have enabled visual-language reasoning in embodied agents, these efforts lack structured spatial memory and instead operate reactively, limiting their generalization and adaptability in complex real-world environments. Here we present Brain-inspired Spatial Cognition for Navigation (BSC-Nav), a unified framework for constructing and leveraging structured spatial memory in embodied agents. BSC-Nav builds allocentric cognitive maps from egocentric trajectories and contextual cues, and dynamically retrieves spatial knowledge aligned with semantic goals. Integrated with powerful MLLMs, BSC-Nav achieves state-of-the-art efficacy and efficiency across diverse navigation tasks, demonstrates strong zero-shot generalization, and supports versatile embodied behaviors in the real physical world, offering a scalable and biologically grounded path toward general-purpose spatial intelligence. 

**Abstract (ZH)**: 空间认知通过构建空间内部模型来实现适应性目标导向行为。生物系统将空间知识整合为三种相互连接的形式：地标作为显著线索、路径知识用于运动轨迹、航图知识用于地图式的表示。尽管多模态大语言模型（MLLMs）的近期进展已在具身智能体中实现了视觉-语言推理，但在构建结构化空间记忆方面仍存不足，导致其在复杂现实环境中的泛化能力和适应性受限。在这里，我们提出了借鉴大脑的空间认知框架用于导航 (BSC-Nav)，这是一种构建和利用具身智能体结构化空间记忆的统一框架。BSC-Nav 从以自我为中心的轨迹和上下文线索构建他中心的空间认知地图，并动态检索与语义目标对齐的空间知识。结合强大的 MLLMs，BSC-Nav 在多种导航任务中达到了最先进的效果和效率，并展示了强大的零样本泛化能力，支持在实际物理世界中的多功能具身行为，为通用空间智能提供了可扩展且生物学依据的路径。 

---
# OmniMRI: A Unified Vision--Language Foundation Model for Generalist MRI Interpretation 

**Title (ZH)**: OmniMRI：统一的视觉-语言基础模型，用于通用MRI解释 

**Authors**: Xingxin He, Aurora Rofena, Ruimin Feng, Haozhe Liao, Zhaoye Zhou, Albert Jang, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17524)  

**Abstract**: Magnetic Resonance Imaging (MRI) is indispensable in clinical practice but remains constrained by fragmented, multi-stage workflows encompassing acquisition, reconstruction, segmentation, detection, diagnosis, and reporting. While deep learning has achieved progress in individual tasks, existing approaches are often anatomy- or application-specific and lack generalizability across diverse clinical settings. Moreover, current pipelines rarely integrate imaging data with complementary language information that radiologists rely on in routine practice. Here, we introduce OmniMRI, a unified vision-language foundation model designed to generalize across the entire MRI workflow. OmniMRI is trained on a large-scale, heterogeneous corpus curated from 60 public datasets, over 220,000 MRI volumes and 19 million MRI slices, incorporating image-only data, paired vision-text data, and instruction-response data. Its multi-stage training paradigm, comprising self-supervised vision pretraining, vision-language alignment, multimodal pretraining, and multi-task instruction tuning, progressively equips the model with transferable visual representations, cross-modal reasoning, and robust instruction-following capabilities. Qualitative results demonstrate OmniMRI's ability to perform diverse tasks within a single architecture, including MRI reconstruction, anatomical and pathological segmentation, abnormality detection, diagnostic suggestion, and radiology report generation. These findings highlight OmniMRI's potential to consolidate fragmented pipelines into a scalable, generalist framework, paving the way toward foundation models that unify imaging and clinical language for comprehensive, end-to-end MRI interpretation. 

**Abstract (ZH)**: 全面MRI：统一的视觉-语言基础模型在整个MRI工作流程中的泛化应用 

---
# Agentic AI for Software: thoughts from Software Engineering community 

**Title (ZH)**: 软件领域的代理型AI：来自软件工程社区的思考 

**Authors**: Abhik Roychoudhury  

**Link**: [PDF](https://arxiv.org/pdf/2508.17343)  

**Abstract**: AI agents have recently shown significant promise in software engineering. Much public attention has been transfixed on the topic of code generation from Large Language Models (LLMs) via a prompt. However, software engineering is much more than programming, and AI agents go far beyond instructions given by a prompt.
At the code level, common software tasks include code generation, testing, and program repair. Design level software tasks may include architecture exploration, requirements understanding, and requirements enforcement at the code level. Each of these software tasks involves micro-decisions which can be taken autonomously by an AI agent, aided by program analysis tools. This creates the vision of an AI software engineer, where the AI agent can be seen as a member of a development team.
Conceptually, the key to successfully developing trustworthy agentic AI-based software workflows will be to resolve the core difficulty in software engineering - the deciphering and clarification of developer intent. Specification inference, or deciphering the intent, thus lies at the heart of many software tasks, including software maintenance and program repair. A successful deployment of agentic technology into software engineering would involve making conceptual progress in such intent inference via agents.
Trusting the AI agent becomes a key aspect, as software engineering becomes more automated. Higher automation also leads to higher volume of code being automatically generated, and then integrated into code-bases. Thus to deal with this explosion, an emerging direction is AI-based verification and validation (V & V) of AI generated code. We posit that agentic software workflows in future will include such AIbased V&V. 

**Abstract (ZH)**: AI代理在软件工程中的 recent进展及其挑战：从代码生成到基于代理的验证与验证 

---
# Chinese Court Simulation with LLM-Based Agent System 

**Title (ZH)**: 基于LLM的代理系统模拟中国法庭 

**Authors**: Kaiyuan Zhang, Jiaqi Li, Yueyue Wu, Haitao Li, Cheng Luo, Shaokun Zou, Yujia Zhou, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17322)  

**Abstract**: Mock trial has long served as an important platform for legal professional training and education. It not only helps students learn about realistic trial procedures, but also provides practical value for case analysis and judgment prediction. Traditional mock trials are difficult to access by the public because they rely on professional tutors and human participants. Fortunately, the rise of large language models (LLMs) provides new opportunities for creating more accessible and scalable court simulations. While promising, existing research mainly focuses on agent construction while ignoring the systematic design and evaluation of court simulations, which are actually more important for the credibility and usage of court simulation in practice. To this end, we present the first court simulation framework -- SimCourt -- based on the real-world procedure structure of Chinese courts. Our framework replicates all 5 core stages of a Chinese trial and incorporates 5 courtroom roles, faithfully following the procedural definitions in China. To simulate trial participants with different roles, we propose and craft legal agents equipped with memory, planning, and reflection abilities. Experiment on legal judgment prediction show that our framework can generate simulated trials that better guide the system to predict the imprisonment, probation, and fine of each case. Further annotations by human experts show that agents' responses under our simulation framework even outperformed judges and lawyers from the real trials in many scenarios. These further demonstrate the potential of LLM-based court simulation. 

**Abstract (ZH)**: 基于大型语言模型的法庭模拟框架：SimCourt 

---
# Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model 

**Title (ZH)**: 随心驾驶：基于多头扩散模型的策略级运动规划 

**Authors**: Fan Ding, Xuewen Luo, Hwa Hui Tew, Ruturaj Reddy, Xikun Wang, Junn Yong Loo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16947)  

**Abstract**: Recent advances in motion planning for autonomous driving have led to models capable of generating high-quality trajectories. However, most existing planners tend to fix their policy after supervised training, leading to consistent but rigid driving behaviors. This limits their ability to reflect human preferences or adapt to dynamic, instruction-driven demands. In this work, we propose a diffusion-based multi-head trajectory planner(M-diffusion planner). During the early training stage, all output heads share weights to learn to generate high-quality trajectories. Leveraging the probabilistic nature of diffusion models, we then apply Group Relative Policy Optimization (GRPO) to fine-tune the pre-trained model for diverse policy-specific behaviors. At inference time, we incorporate a large language model (LLM) to guide strategy selection, enabling dynamic, instruction-aware planning without switching models. Closed-loop simulation demonstrates that our post-trained planner retains strong planning capability while achieving state-of-the-art (SOTA) performance on the nuPlan val14 benchmark. Open-loop results further show that the generated trajectories exhibit clear diversity, effectively satisfying multi-modal driving behavior requirements. The code and related experiments will be released upon acceptance of the paper. 

**Abstract (ZH)**: 近年来，自主驾驶中的运动规划进展使得能够生成高质量轨迹的模型得以实现。然而，现有的大多数规划器在监督训练后倾向于固定其策略，导致一致但僵化的驾驶行为。这限制了它们反映人类偏好或适应动态、指令驱动需求的能力。在本文中，我们提出了一种基于扩散的多头轨迹规划器（M-diffusion planner）。在训练的早期阶段，所有输出头共享权重以学习生成高质量的轨迹。利用扩散模型的概率性质，我们随后应用组相对策略优化（GRPO）对预训练模型进行微调，以获得多样化的策略特定行为。在推理阶段，我们引入了一个大型语言模型（LLM）来引导策略选择，从而实现动态、指令感知的规划，而无需切换模型。闭环仿真结果表明，我们的后训练规划器保持了强大的规划能力，并在nuPlan val14基准测试中实现了最先进的性能。开环结果进一步表明，生成的轨迹表现出明显的多样性，有效地满足了多模态驾驶行为的要求。论文被接受后，代码及相关实验将公开。 

---
# Dream to Chat: Model-based Reinforcement Learning on Dialogues with User Belief Modeling 

**Title (ZH)**: 梦境对话：基于模型的对话中用户信念建模增强学习 

**Authors**: Yue Zhao, Xiaoyu Wang, Dan Wang, Zhonglin Jiang, Qingqing Gu, Teng Chen, Ningyuan Xi, Jinxian Qu, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.16876)  

**Abstract**: World models have been widely utilized in robotics, gaming, and auto-driving. However, their applications on natural language tasks are relatively limited. In this paper, we construct the dialogue world model, which could predict the user's emotion, sentiment, and intention, and future utterances. By defining a POMDP, we argue emotion, sentiment and intention can be modeled as the user belief and solved by maximizing the information bottleneck. By this user belief modeling, we apply the model-based reinforcement learning framework to the dialogue system, and propose a framework called DreamCUB. Experiments show that the pretrained dialogue world model can achieve state-of-the-art performances on emotion classification and sentiment identification, while dialogue quality is also enhanced by joint training of the policy, critic and dialogue world model. Further analysis shows that this manner holds a reasonable exploration-exploitation balance and also transfers well to out-of-domain scenarios such as empathetic dialogues. 

**Abstract (ZH)**: 世界的对话模型已经在机器人、游戏和自动驾驶等领域得到了广泛应用。然而，它们在自然语言任务中的应用相对有限。本文构建了对话世界模型，能够预测用户的情绪、情感、意图以及未来的话语。通过定义POMDP，我们认为情绪、情感和意图可以被建模为用户信念，并通过最大化信息瓶颈来解决。基于用户信念建模，我们将基于模型的强化学习框架应用于对话系统，并提出了一种名为DreamCUB的框架。实验表明，预训练的对话世界模型在情绪分类和情感识别上取得了最先进的性能，同时联合训练策略、评论家和对话世界模型也提升了对话质量。进一步的分析表明，该方法在包括共情对话在内的跨域场景中表现出了合理的探索-利用平衡，并且具有较强的泛化能力。 

---
# An Embodied AR Navigation Agent: Integrating BIM with Retrieval-Augmented Generation for Language Guidance 

**Title (ZH)**: 具身AR导航代理：结合BIM与检索增强生成的语言指导 

**Authors**: Hsuan-Kung Yang, Tsu-Ching Hsiao, Ryoichiro Oka, Ryuya Nishino, Satoko Tofukuji, Norimasa Kobori  

**Link**: [PDF](https://arxiv.org/pdf/2508.16602)  

**Abstract**: Delivering intelligent and adaptive navigation assistance in augmented reality (AR) requires more than visual cues, as it demands systems capable of interpreting flexible user intent and reasoning over both spatial and semantic context. Prior AR navigation systems often rely on rigid input schemes or predefined commands, which limit the utility of rich building data and hinder natural interaction. In this work, we propose an embodied AR navigation system that integrates Building Information Modeling (BIM) with a multi-agent retrieval-augmented generation (RAG) framework to support flexible, language-driven goal retrieval and route planning. The system orchestrates three language agents, Triage, Search, and Response, built on large language models (LLMs), which enables robust interpretation of open-ended queries and spatial reasoning using BIM data. Navigation guidance is delivered through an embodied AR agent, equipped with voice interaction and locomotion, to enhance user experience. A real-world user study yields a System Usability Scale (SUS) score of 80.5, indicating excellent usability, and comparative evaluations show that the embodied interface can significantly improves users' perception of system intelligence. These results underscore the importance and potential of language-grounded reasoning and embodiment in the design of user-centered AR navigation systems. 

**Abstract (ZH)**: 在增强现实（AR）中提供智能和适应性的导航辅助需要超越视觉提示，因为它要求系统能够解释灵活的用户意图并推理空间和语义上下文。之前的AR导航系统往往依赖于固定的输入方案或预定义的命令，这限制了丰富建筑数据的用途并妨碍了自然交互。在本工作中，我们提出了一种结合建筑信息模型（BIM）和多代理检索增强生成（RAG）框架的具身AR导航系统，该系统支持灵活的、基于语言的目标检索和路径规划。该系统协调了三个基于大规模语言模型的语言代理—Triage、Search和Response，利用BIM数据实现了对开放查询的稳健解释和空间推理。导航指导通过一个具有语音交互和移动功能的具身AR代理提供，以提升用户体验。实地用户研究获得的系统可用性量表（SUS）得分为80.5，表明了极好的可用性，对比评估表明，具身界面可以显著提高用户对系统智能性的感知。这些结果强调了语言驱动的推理和具身性在用户中心AR导航系统设计中的重要性和潜力。 

---
