# Video Generators are Robot Policies 

**Title (ZH)**: 视频生成器是机器人策略 

**Authors**: Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, Carl Vondrick  

**Link**: [PDF](https://arxiv.org/pdf/2508.00795)  

**Abstract**: Despite tremendous progress in dexterous manipulation, current visuomotor policies remain fundamentally limited by two challenges: they struggle to generalize under perceptual or behavioral distribution shifts, and their performance is constrained by the size of human demonstration data. In this paper, we use video generation as a proxy for robot policy learning to address both limitations simultaneously. We propose Video Policy, a modular framework that combines video and action generation that can be trained end-to-end. Our results demonstrate that learning to generate videos of robot behavior allows for the extraction of policies with minimal demonstration data, significantly improving robustness and sample efficiency. Our method shows strong generalization to unseen objects, backgrounds, and tasks, both in simulation and the real world. We further highlight that task success is closely tied to the generated video, with action-free video data providing critical benefits for generalizing to novel tasks. By leveraging large-scale video generative models, we achieve superior performance compared to traditional behavior cloning, paving the way for more scalable and data-efficient robot policy learning. 

**Abstract (ZH)**: 尽管在灵巧操作方面取得了巨大进展，当前的视觉-运动策略仍受两大挑战的限制：它们在知觉或行为分布转移时难以泛化，且性能受限于人类演示数据的规模。本文中，我们使用视频生成作为机器人策略学习的代理，同时解决这两种限制。我们提出Video Policy，这是一种结合视频和动作生成的模块化框架，可以进行端到端训练。实验结果表明，学习生成机器人行为的视频能够利用少量的演示数据提取策略，显著提高鲁棒性和样本效率。我们的方法在仿真和现实世界中对未见物体、背景和任务都显示出强大的泛化能力。进一步研究表明，任务成功与生成的视频密切相关，无动作的视频数据提供了对新任务泛化的关键益处。通过利用大规模的视频生成模型，我们取得了优于传统行为克隆的表现，为更具扩展性和数据效率的机器人策略学习开辟了新途径。 

---
# On-Device Diffusion Transformer Policy for Efficient Robot Manipulation 

**Title (ZH)**: 设备端扩散变换器策略高效机器人操作 

**Authors**: Yiming Wu, Huan Wang, Zhenghao Chen, Jianxin Pang, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00697)  

**Abstract**: Diffusion Policies have significantly advanced robotic manipulation tasks via imitation learning, but their application on resource-constrained mobile platforms remains challenging due to computational inefficiency and extensive memory footprint. In this paper, we propose LightDP, a novel framework specifically designed to accelerate Diffusion Policies for real-time deployment on mobile devices. LightDP addresses the computational bottleneck through two core strategies: network compression of the denoising modules and reduction of the required sampling steps. We first conduct an extensive computational analysis on existing Diffusion Policy architectures, identifying the denoising network as the primary contributor to latency. To overcome performance degradation typically associated with conventional pruning methods, we introduce a unified pruning and retraining pipeline, optimizing the model's post-pruning recoverability explicitly. Furthermore, we combine pruning techniques with consistency distillation to effectively reduce sampling steps while maintaining action prediction accuracy. Experimental evaluations on the standard datasets, \ie, PushT, Robomimic, CALVIN, and LIBERO, demonstrate that LightDP achieves real-time action prediction on mobile devices with competitive performance, marking an important step toward practical deployment of diffusion-based policies in resource-limited environments. Extensive real-world experiments also show the proposed LightDP can achieve performance comparable to state-of-the-art Diffusion Policies. 

**Abstract (ZH)**: 轻量化扩散策略：一种针对移动平台的实时部署框架 

---
# Towards Data-Driven Adaptive Exoskeleton Assistance for Post-stroke Gait 

**Title (ZH)**: 基于数据驱动的适应性外骨骼助力卒中后步态恢复研究 

**Authors**: Fabian C. Weigend, Dabin K. Choe, Santiago Canete, Conor J. Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2508.00691)  

**Abstract**: Recent work has shown that exoskeletons controlled through data-driven methods can dynamically adapt assistance to various tasks for healthy young adults. However, applying these methods to populations with neuromotor gait deficits, such as post-stroke hemiparesis, is challenging. This is due not only to high population heterogeneity and gait variability but also to a lack of post-stroke gait datasets to train accurate models. Despite these challenges, data-driven methods offer a promising avenue for control, potentially allowing exoskeletons to function safely and effectively in unstructured community settings. This work presents a first step towards enabling adaptive plantarflexion and dorsiflexion assistance from data-driven torque estimation during post-stroke walking. We trained a multi-task Temporal Convolutional Network (TCN) using collected data from four post-stroke participants walking on a treadmill ($R^2$ of $0.74 \pm 0.13$). The model uses data from three inertial measurement units (IMU) and was pretrained on healthy walking data from 6 participants. We implemented a wearable prototype for our ankle torque estimation approach for exoskeleton control and demonstrated the viability of real-time sensing, estimation, and actuation with one post-stroke participant. 

**Abstract (ZH)**: 基于数据驱动的方法实现卒中后步行中适应性跖屈和背屈辅助研究 

---
# OpenScout v1.1 mobile robot: a case study on open hardware continuation 

**Title (ZH)**: 开源探险者v1.1移动机器人：开源硬件延续的案例研究 

**Authors**: Bartosz Krawczyk, Ahmed Elbary, Robbie Cato, Jagdish Patil, Kaung Myat, Anyeh Ndi-Tah, Nivetha Sakthivel, Mark Crampton, Gautham Das, Charles Fox  

**Link**: [PDF](https://arxiv.org/pdf/2508.00625)  

**Abstract**: OpenScout is an Open Source Hardware (OSH) mobile robot for research and industry. It is extended to v1.1 which includes simplified, cheaper and more powerful onboard compute hardware; a simulated ROS2 interface; and a Gazebo simulation. Changes, their rationale, project methodology, and results are reported as an OSH case study. 

**Abstract (ZH)**: OpenScout是面向研究和工业领域的开源硬件（OSH）移动机器人，已扩展至v1.1版本，包括简化、更便宜且更强大的机载计算硬件；模拟ROS2接口；以及Gazebo仿真。项目方法论、改动理由、结果等均作为OSH案例研究进行报告。 

---
# A control scheme for collaborative object transportation between a human and a quadruped robot using the MIGHTY suction cup 

**Title (ZH)**: 基于MIGHTY吸附杯的人与四足机器人协作物体运输的控制方案 

**Authors**: Konstantinos Plotas, Emmanouil Papadakis, Drosakis Drosakis, Panos Trahanias, Dimitrios Papageorgiou  

**Link**: [PDF](https://arxiv.org/pdf/2508.00584)  

**Abstract**: In this work, a control scheme for human-robot collaborative object transportation is proposed, considering a quadruped robot equipped with the MIGHTY suction cup that serves both as a gripper for holding the object and a force/torque sensor. The proposed control scheme is based on the notion of admittance control, and incorporates a variable damping term aiming towards increasing the controllability of the human and, at the same time, decreasing her/his effort. Furthermore, to ensure that the object is not detached from the suction cup during the collaboration, an additional control signal is proposed, which is based on a barrier artificial potential. The proposed control scheme is proven to be passive and its performance is demonstrated through experimental evaluations conducted using the Unitree Go1 robot equipped with the MIGHTY suction cup. 

**Abstract (ZH)**: 基于MIGHTY吸附杯的四足机器人协作物体运输控制方案 

---
# OmniUnet: A Multimodal Network for Unstructured Terrain Segmentation on Planetary Rovers Using RGB, Depth, and Thermal Imagery 

**Title (ZH)**: 全方位Unet：一种基于RGB、深度和热成像的行星探测车不规则地形 segmentation 网络 

**Authors**: Raul Castilla-Arquillo, Carlos Perez-del-Pulgar, Levin Gerdes, Alfonso Garcia-Cerezo, Miguel A. Olivares-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00580)  

**Abstract**: Robot navigation in unstructured environments requires multimodal perception systems that can support safe navigation. Multimodality enables the integration of complementary information collected by different sensors. However, this information must be processed by machine learning algorithms specifically designed to leverage heterogeneous data. Furthermore, it is necessary to identify which sensor modalities are most informative for navigation in the target environment. In Martian exploration, thermal imagery has proven valuable for assessing terrain safety due to differences in thermal behaviour between soil types. This work presents OmniUnet, a transformer-based neural network architecture for semantic segmentation using RGB, depth, and thermal (RGB-D-T) imagery. A custom multimodal sensor housing was developed using 3D printing and mounted on the Martian Rover Testbed for Autonomy (MaRTA) to collect a multimodal dataset in the Bardenas semi-desert in northern Spain. This location serves as a representative environment of the Martian surface, featuring terrain types such as sand, bedrock, and compact soil. A subset of this dataset was manually labeled to support supervised training of the network. The model was evaluated both quantitatively and qualitatively, achieving a pixel accuracy of 80.37% and demonstrating strong performance in segmenting complex unstructured terrain. Inference tests yielded an average prediction time of 673 ms on a resource-constrained computer (Jetson Orin Nano), confirming its suitability for on-robot deployment. The software implementation of the network and the labeled dataset have been made publicly available to support future research in multimodal terrain perception for planetary robotics. 

**Abstract (ZH)**: 火星探测中无结构环境中机器人的导航需要多模态感知系统以支持安全导航。多模态性使不同传感器收集的互补信息能够整合。然而，这些信息必须由专门设计的机器学习算法进行处理，以利用异质数据。此外，还必须确定在目标环境中哪些传感器模态对于导航最有信息价值。热成像在火星探索中由于不同土壤类型热行为的差异已被证明对评估地形安全性非常有价值。本文介绍了基于Transformer的OmniUnet神经网络架构，用于RGB、深度和热（RGB-D-T）成像的语义分割。使用3D打印开发了一种自定义多模态传感器外壳，并安装在火星自主漫游车测试平台（MaRTA）上，在西班牙北部的巴登纳半沙漠地区收集多模态数据集。该地点作为火星表面的代表环境，包含了沙地、岩床和紧实土壤等多种地形类型。手工对该数据集的一部分进行了标注，以支持网络的监督训练。该模型在定量和定性评估中均表现出色，像素准确率达到80.37%，在复杂无结构地形分割方面表现出强大性能。推理测试在资源受限的计算机（Jetson Orin Nano）上平均预测时间为673毫秒，证实了其在机器人上的部署适用性。该网络的软件实现和标注数据集已公开发布，以支持未来关于行星机器人多模态地形感知的研究。 

---
# HannesImitation: Grasping with the Hannes Prosthetic Hand via Imitation Learning 

**Title (ZH)**: Hannesimitation: 使用模仿学习控制Hannes假人手抓取 

**Authors**: Carlo Alessi, Federico Vasile, Federico Ceola, Giulia Pasquale, Nicolò Boccardo, Lorenzo Natale  

**Link**: [PDF](https://arxiv.org/pdf/2508.00491)  

**Abstract**: Recent advancements in control of prosthetic hands have focused on increasing autonomy through the use of cameras and other sensory inputs. These systems aim to reduce the cognitive load on the user by automatically controlling certain degrees of freedom. In robotics, imitation learning has emerged as a promising approach for learning grasping and complex manipulation tasks while simplifying data collection. Its application to the control of prosthetic hands remains, however, largely unexplored. Bridging this gap could enhance dexterity restoration and enable prosthetic devices to operate in more unconstrained scenarios, where tasks are learned from demonstrations rather than relying on manually annotated sequences. To this end, we present HannesImitationPolicy, an imitation learning-based method to control the Hannes prosthetic hand, enabling object grasping in unstructured environments. Moreover, we introduce the HannesImitationDataset comprising grasping demonstrations in table, shelf, and human-to-prosthesis handover scenarios. We leverage such data to train a single diffusion policy and deploy it on the prosthetic hand to predict the wrist orientation and hand closure for grasping. Experimental evaluation demonstrates successful grasps across diverse objects and conditions. Finally, we show that the policy outperforms a segmentation-based visual servo controller in unstructured scenarios. Additional material is provided on our project page: this https URL 

**Abstract (ZH)**: 最近在假手控制领域的进展集中在通过使用摄像头和其他感官输入来增加自主性。这些系统旨在通过自动控制某些自由度来减轻用户的认知负担。在机器人领域，模仿学习已成为一种有前途的方法，用于学习抓取和复杂操作任务，同时简化数据收集。然而，将其应用于假手控制的研究仍然相对较少。弥合这一差距可能增强灵巧性恢复，并使假手能够在更具约束条件的场景中操作，其中任务是通过示范而不是依赖手动标注序列来学习的。为此，我们提出了一种基于模仿学习的方法HannesImitationPolicy，用于控制Hannes假手，使其能够在非结构化环境中进行物体抓取。此外，我们还引入了HannesImitationDataset，该数据集包含在桌面、架子和人到假手交接场景中的抓取示范。我们利用这些数据训练一个扩散策略，并将其部署到假手上，以预测手腕姿态和手部闭合以进行抓取。实验评估表明，该策略在多种物体和条件下实现了成功的抓取。最后，我们展示了该策略在非结构化场景中优于基于分割的视觉伺服控制器。更多材料可在我们的项目页面获取：this https URL 

---
# SubCDM: Collective Decision-Making with a Swarm Subset 

**Title (ZH)**: 集体决策中的子群集决策Making: SubCDM with a Swarm Subset 

**Authors**: Samratul Fuady, Danesh Tarapore, Mohammad D. Soorati  

**Link**: [PDF](https://arxiv.org/pdf/2508.00467)  

**Abstract**: Collective decision-making is a key function of autonomous robot swarms, enabling them to reach a consensus on actions based on environmental features. Existing strategies require the participation of all robots in the decision-making process, which is resource-intensive and prevents the swarm from allocating the robots to any other tasks. We propose Subset-Based Collective Decision-Making (SubCDM), which enables decisions using only a swarm subset. The construction of the subset is dynamic and decentralized, relying solely on local information. Our method allows the swarm to adaptively determine the size of the subset for accurate decision-making, depending on the difficulty of reaching a consensus. Simulation results using one hundred robots show that our approach achieves accuracy comparable to using the entire swarm while reducing the number of robots required to perform collective decision-making, making it a resource-efficient solution for collective decision-making in swarm robotics. 

**Abstract (ZH)**: 基于子集的集体决策-making in Autonomous Robot Swarms via Subset-Based Collective Decision-Making (SubCDM) 

---
# On Learning Closed-Loop Probabilistic Multi-Agent Simulator 

**Title (ZH)**: 关于学习闭环概率多Agent模拟器 

**Authors**: Juanwu Lu, Rohit Gupta, Ahmadreza Moradipari, Kyungtae Han, Ruqi Zhang, Ziran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00384)  

**Abstract**: The rapid iteration of autonomous vehicle (AV) deployments leads to increasing needs for building realistic and scalable multi-agent traffic simulators for efficient evaluation. Recent advances in this area focus on closed-loop simulators that enable generating diverse and interactive scenarios. This paper introduces Neural Interactive Agents (NIVA), a probabilistic framework for multi-agent simulation driven by a hierarchical Bayesian model that enables closed-loop, observation-conditioned simulation through autoregressive sampling from a latent, finite mixture of Gaussian distributions. We demonstrate how NIVA unifies preexisting sequence-to-sequence trajectory prediction models and emerging closed-loop simulation models trained on Next-token Prediction (NTP) from a Bayesian inference perspective. Experiments on the Waymo Open Motion Dataset demonstrate that NIVA attains competitive performance compared to the existing method while providing embellishing control over intentions and driving styles. 

**Abstract (ZH)**: 自动驾驶汽车（AV）部署的快速迭代推动了对高效评估所需的实际且可扩展的多Agent交通模拟器的需求。这一领域的最新进展集中在可以生成多样化且交互式场景的闭环模拟器上。本文介绍了一种基于分层贝叶斯模型的神经交互Agent（NIVA），这是一种概率框架，通过自回归采样从潜在的有限混合高斯分布中实现闭环、基于观测的模拟。从贝叶斯推理的角度，我们展示NIVA如何统一现有的时间序列到时间序列轨迹预测模型和新兴的闭环模拟模型，并提供对意图和驾驶风格的增强控制。实验表明，NIVA在Waymo开放运动数据集上的性能与现有方法相当，同时提供了增强的控制。 

---
# A Whole-Body Motion Imitation Framework from Human Data for Full-Size Humanoid Robot 

**Title (ZH)**: 基于人类数据的全身运动模仿框架用于全尺寸人形机器人 

**Authors**: Zhenghan Chen, Haodong Zhang, Dongqi Wang, Jiyu Yu, Haocheng Xu, Yue Wang, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00362)  

**Abstract**: Motion imitation is a pivotal and effective approach for humanoid robots to achieve a more diverse range of complex and expressive movements, making their performances more human-like. However, the significant differences in kinematics and dynamics between humanoid robots and humans present a major challenge in accurately imitating motion while maintaining balance. In this paper, we propose a novel whole-body motion imitation framework for a full-size humanoid robot. The proposed method employs contact-aware whole-body motion retargeting to mimic human motion and provide initial values for reference trajectories, and the non-linear centroidal model predictive controller ensures the motion accuracy while maintaining balance and overcoming external disturbances in real time. The assistance of the whole-body controller allows for more precise torque control. Experiments have been conducted to imitate a variety of human motions both in simulation and in a real-world humanoid robot. These experiments demonstrate the capability of performing with accuracy and adaptability, which validates the effectiveness of our approach. 

**Abstract (ZH)**: 基于全身体态的仿人运动模仿框架 

---
# TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots 

**Title (ZH)**: TOP: 用于人类机器人稳定准确站立操作的时间优化策略 

**Authors**: Zhenghan Chen, Haocheng Xu, Haodong Zhang, Liang Zhang, He Li, Dongqi Wang, Jiyu Yu, Yifei Yang, Zhongxiang Zhou, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00355)  

**Abstract**: Humanoid robots have the potential capability to perform a diverse range of manipulation tasks, but this is based on a robust and precise standing controller. Existing methods are either ill-suited to precisely control high-dimensional upper-body joints, or difficult to ensure both robustness and accuracy, especially when upper-body motions are fast. This paper proposes a novel time optimization policy (TOP), to train a standing manipulation control model that ensures balance, precision, and time efficiency simultaneously, with the idea of adjusting the time trajectory of upper-body motions but not only strengthening the disturbance resistance of the lower-body. Our approach consists of three parts. Firstly, we utilize motion prior to represent upper-body motions to enhance the coordination ability between the upper and lower-body by training a variational autoencoder (VAE). Then we decouple the whole-body control into an upper-body PD controller for precision and a lower-body RL controller to enhance robust stability. Finally, we train TOP method in conjunction with the decoupled controller and VAE to reduce the balance burden resulting from fast upper-body motions that would destabilize the robot and exceed the capabilities of the lower-body RL policy. The effectiveness of the proposed approach is evaluated via both simulation and real world experiments, which demonstrate the superiority on standing manipulation tasks stably and accurately. The project page can be found at this https URL. 

**Abstract (ZH)**: 类人机器人具有执行多种操作任务的潜力，但这基于一个稳健且精确的站立控制器。现有方法要么不适合精确控制高维度上半身关节，要么难以同时保证稳健性和精确性，尤其是在上半身运动快速时。本文提出了一种新颖的时间优化策略（TOP），旨在训练一个同时确保平衡、精确性和时间效率的站立操作控制模型，通过调整上半身运动的时间轨迹，而不只是增强下半身的抗干扰能力。我们的方法包含三个部分：首先，利用运动先验来表示上半身运动，通过训练变分自编码器（VAE）增强上下半身的协调能力；然后，将全身控制分解为一个用于精确性的上半身PD控制器和一个用于增强鲁棒稳定的下半身RL控制器；最后，结合解耦控制器和VAE共同训练TOP方法，以降低因快速上半身运动导致的平衡负担，防止机器人失稳并超出下半身RL策略的能力范围。提出的这种方法通过仿真和现实世界的实验进行了评估，表明在稳定而准确的站立操作任务上具有优势。项目页面可访问 [此链接]。 

---
# Omni-Scan: Creating Visually-Accurate Digital Twin Object Models Using a Bimanual Robot with Handover and Gaussian Splat Merging 

**Title (ZH)**: 全手操掃描：使用交接和高斯点合并的人手双臂机器人创建视觉准确的数字双生对象模型 

**Authors**: Tianshuang Qiu, Zehan Ma, Karim El-Refai, Hiya Shah, Chung Min Kim, Justin Kerr, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2508.00354)  

**Abstract**: 3D Gaussian Splats (3DGSs) are 3D object models derived from multi-view images. Such "digital twins" are useful for simulations, virtual reality, marketing, robot policy fine-tuning, and part inspection. 3D object scanning usually requires multi-camera arrays, precise laser scanners, or robot wrist-mounted cameras, which have restricted workspaces. We propose Omni-Scan, a pipeline for producing high-quality 3D Gaussian Splat models using a bi-manual robot that grasps an object with one gripper and rotates the object with respect to a stationary camera. The object is then re-grasped by a second gripper to expose surfaces that were occluded by the first gripper. We present the Omni-Scan robot pipeline using DepthAny-thing, Segment Anything, as well as RAFT optical flow models to identify and isolate objects held by a robot gripper while removing the gripper and the background. We then modify the 3DGS training pipeline to support concatenated datasets with gripper occlusion, producing an omni-directional (360 degree view) model of the object. We apply Omni-Scan to part defect inspection, finding that it can identify visual or geometric defects in 12 different industrial and household objects with an average accuracy of 83%. Interactive videos of Omni-Scan 3DGS models can be found at this https URL 

**Abstract (ZH)**: 基于双臂机器人的360度三维高斯点云模型生成方法 

---
# TopoDiffuser: A Diffusion-Based Multimodal Trajectory Prediction Model with Topometric Maps 

**Title (ZH)**: 基于拓扑扩散的多模态轨迹预测模型——拓扑地图辅助方法 

**Authors**: Zehui Xu, Junhui Wang, Yongliang Shi, Chao Gao, Guyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.00303)  

**Abstract**: This paper introduces TopoDiffuser, a diffusion-based framework for multimodal trajectory prediction that incorporates topometric maps to generate accurate, diverse, and road-compliant future motion forecasts. By embedding structural cues from topometric maps into the denoising process of a conditional diffusion model, the proposed approach enables trajectory generation that naturally adheres to road geometry without relying on explicit constraints. A multimodal conditioning encoder fuses LiDAR observations, historical motion, and route information into a unified bird's-eye-view (BEV) representation. Extensive experiments on the KITTI benchmark demonstrate that TopoDiffuser outperforms state-of-the-art methods, while maintaining strong geometric consistency. Ablation studies further validate the contribution of each input modality, as well as the impact of denoising steps and the number of trajectory samples. To support future research, we publicly release our code at this https URL. 

**Abstract (ZH)**: 本文介绍了TopoDiffuser，这是一种利用拓扑地图进行多模态轨迹预测的扩散框架，能够生成准确、多样化且符合道路几何的未来运动预测。通过将拓扑地图中的结构线索嵌入到条件扩散模型的去噪过程中，所提出的方法能够在不依赖显式约束的情况下，自然地遵循道路几何进行轨迹生成。多模态条件编码器将LiDAR观测、历史运动和路径信息融合为统一的鸟瞰视图（BEV）表示。在KITTI基准上的 extensive 实验表明，TopoDiffuser 在保持良好的几何一致性的同时超越了现有最先进的方法。消融研究进一步验证了每个输入模态的贡献，以及去噪步骤和轨迹样本数量的影响。为了支持未来的研究，我们在以下网址公开发布了我们的代码：this https URL。 

---
# UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents 

**Title (ZH)**: UAV-ON: 一种基于无人机的开放世界目标导航基准数据集 

**Authors**: Jianqiang Xiao, Yuexuan Sun, Yixin Shao, Boxi Gan, Rongqiang Liu, Yanjing Wu, Weili Gua, Xiang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00288)  

**Abstract**: Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments. 

**Abstract (ZH)**: 基于物体目标的无人机空中导航：一个开放世界环境中的基准（UAV-ON） 

---
# Topology-Inspired Morphological Descriptor for Soft Continuum Robots 

**Title (ZH)**: 拓扑启发的形态描述符用于软连续机器人 

**Authors**: Zhiwei Wu, Siyi Wei, Jiahao Luo, Jinhui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00258)  

**Abstract**: This paper presents a topology-inspired morphological descriptor for soft continuum robots by combining a pseudo-rigid-body (PRB) model with Morse theory to achieve a quantitative characterization of robot morphologies. By counting critical points of directional projections, the proposed descriptor enables a discrete representation of multimodal configurations and facilitates morphological classification. Furthermore, we apply the descriptor to morphology control by formulating the target configuration as an optimization problem to compute actuation parameters that generate equilibrium shapes with desired topological features. The proposed framework provides a unified methodology for quantitative morphology description, classification, and control of soft continuum robots, with the potential to enhance their precision and adaptability in medical applications such as minimally invasive surgery and endovascular interventions. 

**Abstract (ZH)**: 基于拓扑启发的软连续机器人形态描述符：结合伪刚体模型与Morse理论实现机器人形态的定量表征与控制 

---
# CHILD (Controller for Humanoid Imitation and Live Demonstration): a Whole-Body Humanoid Teleoperation System 

**Title (ZH)**: CHILD (用于模仿和现场演示的人形控制器): 一个全身人形远程操作系统 

**Authors**: Noboru Myers, Obin Kwon, Sankalp Yamsani, Joohyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.00162)  

**Abstract**: Recent advances in teleoperation have demonstrated robots performing complex manipulation tasks. However, existing works rarely support whole-body joint-level teleoperation for humanoid robots, limiting the diversity of tasks that can be accomplished. This work presents Controller for Humanoid Imitation and Live Demonstration (CHILD), a compact reconfigurable teleoperation system that enables joint level control over humanoid robots. CHILD fits within a standard baby carrier, allowing the operator control over all four limbs, and supports both direct joint mapping for full-body control and loco-manipulation. Adaptive force feedback is incorporated to enhance operator experience and prevent unsafe joint movements. We validate the capabilities of this system by conducting loco-manipulation and full-body control examples on a humanoid robot and multiple dual-arm systems. Lastly, we open-source the design of the hardware promoting accessibility and reproducibility. Additional details and open-source information are available at our project website: this https URL. 

**Abstract (ZH)**: Recent Advances in Teleoperation Have Demonstrated Robots Performing Complex Manipulation Tasks. However, Existing Works Rarely Support Whole-Body Joint-Level Teleoperation for Humanoid Robots, Limiting the Diversity of Tasks That Can Be Accomplished. This Work Presents Controller for Humanoid Imitation and Live Demonstration (CHILD), a Compact Reconfigurable Teleoperation System That Enables Joint-Level Control Over Humanoid Robots. 

---
# XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation 

**Title (ZH)**: XRoboToolkit：一种跨平台机器人远程操作框架 

**Authors**: Zhigen Zhao, Liuchuan Yu, Ke Jing, Ning Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00097)  

**Abstract**: The rapid advancement of Vision-Language-Action models has created an urgent need for large-scale, high-quality robot demonstration datasets. Although teleoperation is the predominant method for data collection, current approaches suffer from limited scalability, complex setup procedures, and suboptimal data quality. This paper presents XRoboToolkit, a cross-platform framework for extended reality based robot teleoperation built on the OpenXR standard. The system features low-latency stereoscopic visual feedback, optimization-based inverse kinematics, and support for diverse tracking modalities including head, controller, hand, and auxiliary motion trackers. XRoboToolkit's modular architecture enables seamless integration across robotic platforms and simulation environments, spanning precision manipulators, mobile robots, and dexterous hands. We demonstrate the framework's effectiveness through precision manipulation tasks and validate data quality by training VLA models that exhibit robust autonomous performance. 

**Abstract (ZH)**: 基于扩展现实的跨平台机器人遥操作框架XRoboToolkit 

---
# IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation 

**Title (ZH)**: IGL-Nav: 增量三维高斯定位用于图像目标导航 

**Authors**: Wenxuan Guo, Xiuwei Xu, Hang Yin, Ziwei Wang, Jianjiang Feng, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00823)  

**Abstract**: Visual navigation with an image as goal is a fundamental and challenging problem. Conventional methods either rely on end-to-end RL learning or modular-based policy with topological graph or BEV map as memory, which cannot fully model the geometric relationship between the explored 3D environment and the goal image. In order to efficiently and accurately localize the goal image in 3D space, we build our navigation system upon the renderable 3D gaussian (3DGS) representation. However, due to the computational intensity of 3DGS optimization and the large search space of 6-DoF camera pose, directly leveraging 3DGS for image localization during agent exploration process is prohibitively inefficient. To this end, we propose IGL-Nav, an Incremental 3D Gaussian Localization framework for efficient and 3D-aware image-goal navigation. Specifically, we incrementally update the scene representation as new images arrive with feed-forward monocular prediction. Then we coarsely localize the goal by leveraging the geometric information for discrete space matching, which can be equivalent to efficient 3D convolution. When the agent is close to the goal, we finally solve the fine target pose with optimization via differentiable rendering. The proposed IGL-Nav outperforms existing state-of-the-art methods by a large margin across diverse experimental configurations. It can also handle the more challenging free-view image-goal setting and be deployed on real-world robotic platform using a cellphone to capture goal image at arbitrary pose. Project page: this https URL. 

**Abstract (ZH)**: 基于图像的目标三维视觉导航 

---
# Petri Net Modeling and Deadlock-Free Scheduling of Attachable Heterogeneous AGV Systems 

**Title (ZH)**: 可附着异构AGV系统的Petri网建模与死锁自由调度 

**Authors**: Boyu Li, Zhengchen Li, Weimin Wu, Mengchu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.00724)  

**Abstract**: The increasing demand for automation and flexibility drives the widespread adoption of heterogeneous automated guided vehicles (AGVs). This work intends to investigate a new scheduling problem in a material transportation system consisting of attachable heterogeneous AGVs, namely carriers and shuttles. They can flexibly attach to and detach from each other to cooperatively execute complex transportation tasks. While such collaboration enhances operational efficiency, the attachment-induced synchronization and interdependence render the scheduling coupled and susceptible to deadlock. To tackle this challenge, Petri nets are introduced to model AGV schedules, well describing the concurrent and sequential task execution and carrier-shuttle synchronization. Based on Petri net theory, a firing-driven decoding method is proposed, along with deadlock detection and prevention strategies to ensure deadlock-free schedules. Furthermore, a Petri net-based metaheuristic is developed in an adaptive large neighborhood search framework and incorporates an effective acceleration method to enhance computational efficiency. Finally, numerical experiments using real-world industrial data validate the effectiveness of the proposed algorithm against the scheduling policy applied in engineering practice, an exact solver, and four state-of-the-art metaheuristics. A sensitivity analysis is also conducted to provide managerial insights. 

**Abstract (ZH)**: 基于异构自动引导车的材料运输系统调度问题研究 

---
# Context-based Motion Retrieval using Open Vocabulary Methods for Autonomous Driving 

**Title (ZH)**: 基于上下文的运动检索：使用开放词汇方法应用于自主驾驶 

**Authors**: Stefan Englmeier, Max A. Büttner, Katharina Winter, Fabian B. Flohr  

**Link**: [PDF](https://arxiv.org/pdf/2508.00589)  

**Abstract**: Autonomous driving systems must operate reliably in safety-critical scenarios, particularly those involving unusual or complex behavior by Vulnerable Road Users (VRUs). Identifying these edge cases in driving datasets is essential for robust evaluation and generalization, but retrieving such rare human behavior scenarios within the long tail of large-scale datasets is challenging. To support targeted evaluation of autonomous driving systems in diverse, human-centered scenarios, we propose a novel context-aware motion retrieval framework. Our method combines Skinned Multi-Person Linear (SMPL)-based motion sequences and corresponding video frames before encoding them into a shared multimodal embedding space aligned with natural language. Our approach enables the scalable retrieval of human behavior and their context through text queries. This work also introduces our dataset WayMoCo, an extension of the Waymo Open Dataset. It contains automatically labeled motion and scene context descriptions derived from generated pseudo-ground-truth SMPL sequences and corresponding image data. Our approach outperforms state-of-the-art models by up to 27.5% accuracy in motion-context retrieval, when evaluated on the WayMoCo dataset. 

**Abstract (ZH)**: 自动驾驶系统必须在涉及脆弱道路使用者（VRUs）的异常或复杂行为的安全关键场景中可靠运行。识别驾驶数据集中这些边缘案例对于稳健的评估和泛化至关重要，但在大规模数据集的长尾中检索此类罕见的人类行为场景具有挑战性。为了支持在多样的、以人为中心的场景中对自动驾驶系统的针对性评估，我们提出了一种新的上下文感知运动检索框架。我们的方法结合了基于SMPL的人体运动序列及其对应的视频帧，并将它们编码到与自然语言对齐的多模态嵌入空间中。我们的方法通过文本查询实现了对人类行为及其上下文的大规模检索。此外，我们还介绍了我们的数据集WayMoCo，它是Waymo开放数据集的扩展，包含从生成的伪地面真实SMPL序列和相应的图像数据中自动生成的动作和场景上下文描述。与现有的最佳模型相比，我们的方法在WayMoCo数据集上将运动-上下文检索的准确率提高了高达27.5%。 

---
# Towards Efficient Certification of Maritime Remote Operation Centers 

**Title (ZH)**: 向海洋远程操作中心高效认证迈进 

**Authors**: Christian Neurohr, Marcel Saager, Lina Putze, Jan-Patrick Osterloh, Karina Rothemann, Hilko Wiards, Eckard Böde, Axel Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.00543)  

**Abstract**: Additional automation being build into ships implies a shift of crew from ship to shore. However, automated ships still have to be monitored and, in some situations, controlled remotely. These tasks are carried out by human operators located in shore-based remote operation centers. In this work, we present a concept for a hazard database that supports the safeguarding and certification of such remote operation centers. The concept is based on a categorization of hazard sources which we derive from a generic functional architecture. A subsequent preliminary suitability analysis unveils which methods for hazard analysis and risk assessment can adequately fill this hazard database. 

**Abstract (ZH)**: 自动化船舶进一步集成意味着船员的减少，从船上转移到岸上。然而，自动化船舶仍然需要远程监控，并在某些情况下需要远程控制。这些任务由位于岸基远程操作中心的人类操作员执行。本文提出一个理念，旨在支持这些远程操作中心的安全保障和认证。该理念基于从通用功能架构中推导出的危险源分类。随后的初步适用性分析揭示了哪些危险分析和风险评估方法能够适当地填充这种危险数据库。 

---
# Reducing the gap between general purpose data and aerial images in concentrated solar power plants 

**Title (ZH)**: 减少集中式太阳能电站通用数据与航拍图像之间的差距 

**Authors**: M.A. Pérez-Cutiño, J. Valverde, J. Capitán, J.M. Díaz-Báñez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00440)  

**Abstract**: In the context of Concentrated Solar Power (CSP) plants, aerial images captured by drones present a unique set of challenges. Unlike urban or natural landscapes commonly found in existing datasets, solar fields contain highly reflective surfaces, and domain-specific elements that are uncommon in traditional computer vision benchmarks. As a result, machine learning models trained on generic datasets struggle to generalize to this setting without extensive retraining and large volumes of annotated data. However, collecting and labeling such data is costly and time-consuming, making it impractical for rapid deployment in industrial applications.
To address this issue, we propose a novel approach: the creation of AerialCSP, a virtual dataset that simulates aerial imagery of CSP plants. By generating synthetic data that closely mimic real-world conditions, our objective is to facilitate pretraining of models before deployment, significantly reducing the need for extensive manual labeling. Our main contributions are threefold: (1) we introduce AerialCSP, a high-quality synthetic dataset for aerial inspection of CSP plants, providing annotated data for object detection and image segmentation; (2) we benchmark multiple models on AerialCSP, establishing a baseline for CSP-related vision tasks; and (3) we demonstrate that pretraining on AerialCSP significantly improves real-world fault detection, particularly for rare and small defects, reducing the need for extensive manual labeling. AerialCSP is made publicly available at this https URL. 

**Abstract (ZH)**: 在集中式太阳能发电（CSP）电站的背景下，由无人机捕获的航拍图像呈现出独特的挑战。与现有数据集中常见的城市或自然景观不同，太阳能田地包含高度反射的表面和在传统计算机视觉基准中罕见的领域特定元素。因此，机器学习模型在使用通用数据集训练后，在这种环境中推广需要大量的重新训练和标注数据。然而，收集和标注这些数据既耗时又昂贵，使其在工业应用中的快速部署变得不切实际。为解决这一问题，我们提出了一种新的方法：创建AerialCSP，一种模拟CSP电站航拍图像的虚拟数据集。通过生成能够近似模拟现实世界条件的合成数据，我们的目标是在部署前辅助模型的预训练，显著减少对大量手动标注的需要。我们的主要贡献包括三个方面：（1）引入AerialCSP，一种高质量的合成数据集，用于CSP电站的航拍检测，提供标注数据用于对象检测和图像分割；（2）在AerialCSP上对多种模型进行基准测试，建立CSP相关视觉任务的基础线；（3）证明在AerialCSP上的预训练极大地提高了真实世界故障检测的效果，特别是对于罕见和小型缺陷，减少了对大量手动标注的需要。AerialCSP在以下链接公开可用：[this https URL]。 

---
# Controllable Pedestrian Video Editing for Multi-View Driving Scenarios via Motion Sequence 

**Title (ZH)**: 多视图驾驶场景中的可控行人视频编辑基于运动序列 

**Authors**: Danzhen Fu, Jiagao Hu, Daiguo Zhou, Fei Wang, Zepeng Wang, Wenhua Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00299)  

**Abstract**: Pedestrian detection models in autonomous driving systems often lack robustness due to insufficient representation of dangerous pedestrian scenarios in training datasets. To address this limitation, we present a novel framework for controllable pedestrian video editing in multi-view driving scenarios by integrating video inpainting and human motion control techniques. Our approach begins by identifying pedestrian regions of interest across multiple camera views, expanding detection bounding boxes with a fixed ratio, and resizing and stitching these regions into a unified canvas while preserving cross-view spatial relationships. A binary mask is then applied to designate the editable area, within which pedestrian editing is guided by pose sequence control conditions. This enables flexible editing functionalities, including pedestrian insertion, replacement, and removal. Extensive experiments demonstrate that our framework achieves high-quality pedestrian editing with strong visual realism, spatiotemporal coherence, and cross-view consistency. These results establish the proposed method as a robust and versatile solution for multi-view pedestrian video generation, with broad potential for applications in data augmentation and scenario simulation in autonomous driving. 

**Abstract (ZH)**: 自主驾驶系统中的行人检测模型由于训练数据集中危险行人场景表示不足而导致鲁棒性不足。为此，我们提出了一种结合视频修补和人体运动控制技术的可控行人视频编辑新框架，适用于多视角驾驶场景。该方法首先在多个摄像头视角中识别行人区域，以固定比例扩展检测边框，并将这些区域调整大小并拼接成统一画布，同时保持跨视角空间关系。然后应用二值掩码指定可编辑区域，在该区域内根据姿态序列控制条件进行行人编辑，从而实现灵活的编辑功能，包括行人插入、替换和移除。大量实验结果表明，该框架实现了高质量的行人编辑，具有较强的视觉真实感、时空连贯性和跨视角一致性。这些结果确立了所提出方法作为多视角行人视频生成的稳健且多功能的解决方案，并在自主驾驶的数据增强和场景模拟中有广泛的应用潜力。 

---
# Data-Driven Motion Planning for Uncertain Nonlinear Systems 

**Title (ZH)**: 数据驱动的不确定非线性系统运动规划 

**Authors**: Babak Esmaeili, Hamidreza Modares, Stefano Di Cairano  

**Link**: [PDF](https://arxiv.org/pdf/2508.00154)  

**Abstract**: This paper proposes a data-driven motion-planning framework for nonlinear systems that constructs a sequence of overlapping invariant polytopes. Around each randomly sampled waypoint, the algorithm identifies a convex admissible region and solves data-driven linear-matrix-inequality problems to learn several ellipsoidal invariant sets together with their local state-feedback gains. The convex hull of these ellipsoids, still invariant under a piece-wise-affine controller obtained by interpolating the gains, is then approximated by a polytope. Safe transitions between nodes are ensured by verifying the intersection of consecutive convex-hull polytopes and introducing an intermediate node for a smooth transition. Control gains are interpolated in real time via simplex-based interpolation, keeping the state inside the invariant polytopes throughout the motion. Unlike traditional approaches that rely on system dynamics models, our method requires only data to compute safe regions and design state-feedback controllers. The approach is validated through simulations, demonstrating the effectiveness of the proposed method in achieving safe, dynamically feasible paths for complex nonlinear systems. 

**Abstract (ZH)**: 基于数据驱动的非线性系统运动规划框架：构建重叠不变多面体序列 

---
# The Monado SLAM Dataset for Egocentric Visual-Inertial Tracking 

**Title (ZH)**: Monado SLAM数据集：第一人称视觉-惯性追踪 

**Authors**: Mateo de Mayo, Daniel Cremers, Taihú Pire  

**Link**: [PDF](https://arxiv.org/pdf/2508.00088)  

**Abstract**: Humanoid robots and mixed reality headsets benefit from the use of head-mounted sensors for tracking. While advancements in visual-inertial odometry (VIO) and simultaneous localization and mapping (SLAM) have produced new and high-quality state-of-the-art tracking systems, we show that these are still unable to gracefully handle many of the challenging settings presented in the head-mounted use cases. Common scenarios like high-intensity motions, dynamic occlusions, long tracking sessions, low-textured areas, adverse lighting conditions, saturation of sensors, to name a few, continue to be covered poorly by existing datasets in the literature. In this way, systems may inadvertently overlook these essential real-world issues. To address this, we present the Monado SLAM dataset, a set of real sequences taken from multiple virtual reality headsets. We release the dataset under a permissive CC BY 4.0 license, to drive advancements in VIO/SLAM research and development. 

**Abstract (ZH)**: 头部位姿传感器在人形机器人和混合现实头显中的应用：Monado SLAM数据集促进视觉惯性里程计和同时定位与建图的研究 

---
