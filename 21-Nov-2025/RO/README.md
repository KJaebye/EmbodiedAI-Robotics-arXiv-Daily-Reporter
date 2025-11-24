# Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations 

**Title (ZH)**: 智能镜片赋能的灵巧操作：基于野外人类示范的多手指机器人 manipulation 

**Authors**: Irmak Guzey, Haozhi Qi, Julen Urain, Changhao Wang, Jessica Yin, Krishna Bodduluri, Mike Lambeta, Lerrel Pinto, Akshara Rai, Jitendra Malik, Tingfan Wu, Akash Sharma, Homanga Bharadhwaj  

**Link**: [PDF](https://arxiv.org/pdf/2511.16661)  

**Abstract**: Learning multi-fingered robot policies from humans performing daily tasks in natural environments has long been a grand goal in the robotics community. Achieving this would mark significant progress toward generalizable robot manipulation in human environments, as it would reduce the reliance on labor-intensive robot data collection. Despite substantial efforts, progress toward this goal has been bottle-necked by the embodiment gap between humans and robots, as well as by difficulties in extracting relevant contextual and motion cues that enable learning of autonomous policies from in-the-wild human videos. We claim that with simple yet sufficiently powerful hardware for obtaining human data and our proposed framework AINA, we are now one significant step closer to achieving this dream. AINA enables learning multi-fingered policies from data collected by anyone, anywhere, and in any environment using Aria Gen 2 glasses. These glasses are lightweight and portable, feature a high-resolution RGB camera, provide accurate on-board 3D head and hand poses, and offer a wide stereo view that can be leveraged for depth estimation of the scene. This setup enables the learning of 3D point-based policies for multi-fingered hands that are robust to background changes and can be deployed directly without requiring any robot data (including online corrections, reinforcement learning, or simulation). We compare our framework against prior human-to-robot policy learning approaches, ablate our design choices, and demonstrate results across nine everyday manipulation tasks. Robot rollouts are best viewed on our website: this https URL. 

**Abstract (ZH)**: 从自然环境中执行日常任务的人类操控多手指机器人政策的学习一直是机器人学领域的宏伟目标。实现这一目标将有助于机器人在人类环境中的通用操控取得重大进展，因为它可以减少对劳动密集型机器人数据采集的依赖。尽管进行了大量努力，但这一目标的进展一直受人类和机器人之间的实体差距以及从野外的人类视频中提取相关上下文和运动线索以学习自主策略的困难限制。我们主张，借助简单但足够强大的硬件获取人类数据以及我们提出的AINA框架，我们现在距离实现这一梦想更近了一步。AINA使任何人都可以在任何环境、任何地点使用Aria Gen 2眼镜收集数据以学习多手指政策成为可能。这些眼镜轻便且易于携带，配备高分辨率RGB相机，提供精准的车载3D头部和手部姿态，并具有可用于场景深度估计的宽立体视场。该设置使我们能够学习对背景变化具有鲁棒性的3D点基多手指手政策，并可以直接部署而无需任何机器人数据（包括在线修正、强化学习或模拟）。我们将我们的框架与先前的人到机策略学习方法进行比较，消除我们的设计选择，并在九种日常操作任务上展示结果。机器人演示最好通过我们的网站查看：this https URL。 

---
# InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy 

**Title (ZH)**: InternData-A1：首创高保真合成数据用于预训练通用策略 

**Authors**: Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16651)  

**Abstract**: Recent works explore how real and synthetic data contribute to Vision-Language-Action (VLA) models' generalization. While current VLA models have shown the strong effectiveness of large-scale real-robot pre-training, synthetic data has not previously demonstrated comparable capability at scale. This paper provides the first evidence that synthetic data alone can match the performance of the strongest $\pi$-dataset in pre-training a VLA model, revealing the substantial value of large-scale simulation. The resulting model also exhibits surprisingly zero-shot sim-to-real transfer on several challenging tasks. Our synthetic dataset, InternData-A1, contains over 630k trajectories and 7,433 hours across 4 embodiments, 18 skills, 70 tasks, and 227 scenes, covering rigid, articulated, deformable, and fluid-object manipulation. It is generated through a highly autonomous, fully decoupled, and compositional simulation pipeline that enables long-horizon skill composition, flexible task assembly, and heterogeneous embodiments with minimal manual tuning. Using the same architecture as $\pi_0$, we pre-train a model entirely on InternData-A1 and find that it matches the official $\pi_0$ across 49 simulation tasks, 5 real-world tasks, and 4 long-horizon dexterous tasks. We release the dataset and will open-source the generation pipeline to broaden access to large-scale robotic data and to lower the barrier to scalable data creation for embodied AI research. 

**Abstract (ZH)**: Recent Works Explore How Real and Synthetic Data Contribute to the Generalization of Vision-Language-Action (VLA) Models 

---
# MiMo-Embodied: X-Embodied Foundation Model Technical Report 

**Title (ZH)**: MiMo-Embodied: X-Embodied 基础模型技术报告 

**Authors**: Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, Yuchen Zhang, Jing Wu, Jinghui Lu, Chenxu Dang, Jiayi Guan, Jianhua Wu, Zhiyi Hou, Hanbing Li, Shumeng Xia, Mingliang Zhou, Yinan Zheng, Zihao Yue, Shuhao Gu, Hao Tian, Yuannan Shen, Jianwei Cui, Wen Zhang, Shaoqing Xu, Bing Wang, Haiyang Sun, Zeyu Zhu, Yuncheng Jiang, Zibin Guo, Chuhong Gong, Chaofan Zhang, Wenbo Ding, Kun Ma, Guang Chen, Rui Cai, Diyun Xiang, Heng Qu, Fuli Luo, Hangjun Ye, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16518)  

**Abstract**: We open-source MiMo-Embodied, the first cross-embodied foundation model to successfully integrate and achieve state-of-the-art performance in both Autonomous Driving and Embodied AI. MiMo-Embodied sets new records across 17 embodied AI benchmarks in Task Planning, Affordance Prediction and Spatial Understanding, while also excelling in 12 autonomous driving benchmarks across Environmental Perception, Status Prediction, and Driving Planning. Across these tasks, MiMo-Embodied significantly outperforms existing open-source, closed-source, and specialized baselines. Our results indicate that through multi-stage learning, curated data construction, and CoT/RL fine-tuning, these two domains exhibit strong positive transfer and mutually reinforce one another. We provide a detailed analysis of our model design and training methodologies to facilitate further research. Code and models are available at this https URL. 

**Abstract (ZH)**: 我们开源了MiMo-Embodied，这是首个成功将自动驾驶和具身AI两大领域整合并同时实现最佳性能的跨具身基础模型。MiMo-Embodied在17个具身AI基准任务（包括任务规划、可操作性预测和空间理解）上设立了新纪录，同时也在12个自动驾驶基准任务（涵盖环境感知、状态预测和驾驶规划）上表现优异。在这些任务中，MiMo-Embodied显著优于现有的开源、封闭源代码和专门的基础模型。我们的结果表明，通过多阶段学习、精心构建的数据构造以及CoT/RL微调，这两个领域表现出强烈的知识迁移并相互强化。我们提供了详细的模型设计和训练方法分析，以促进进一步的研究。代码和模型可在以下链接获取。 

---
# From Prompts to Printable Models: Support-Effective 3D Generation via Offset Direct Preference Optimization 

**Title (ZH)**: 从提示到可打印模型：基于偏移直接偏好优化的支持有效3D生成 

**Authors**: Chenming Wu, Xiaofan Li, Chengkai Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.16434)  

**Abstract**: The transition from digital 3D models to physical objects via 3D printing often requires support structures to prevent overhanging features from collapsing during the fabrication process. While current slicing technologies offer advanced support strategies, they focus on post-processing optimizations rather than addressing the underlying need for support-efficient design during the model generation phase. This paper introduces SEG (\textit{\underline{S}upport-\underline{E}ffective \underline{G}eneration}), a novel framework that integrates Direct Preference Optimization with an Offset (ODPO) into the 3D generation pipeline to directly optimize models for minimal support material usage. By incorporating support structure simulation into the training process, SEG encourages the generation of geometries that inherently require fewer supports, thus reducing material waste and production time. We demonstrate SEG's effectiveness through extensive experiments on two benchmark datasets, Thingi10k-Val and GPT-3DP-Val, showing that SEG significantly outperforms baseline models such as TRELLIS, DPO, and DRO in terms of support volume reduction and printability. Qualitative results further reveal that SEG maintains high fidelity to input prompts while minimizing the need for support structures. Our findings highlight the potential of SEG to transform 3D printing by directly optimizing models during the generative process, paving the way for more sustainable and efficient digital fabrication practices. 

**Abstract (ZH)**: SEG：一种直接优化生成支持结构材料使用的新型框架 

---
# LAOF: Robust Latent Action Learning with Optical Flow Constraints 

**Title (ZH)**: LAOF：基于光学流约束的鲁棒潜动作学习 

**Authors**: Xizhou Bu, Jiexi Lyu, Fulei Sun, Ruichen Yang, Zhiqiang Ma, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.16407)  

**Abstract**: Learning latent actions from large-scale videos is crucial for the pre-training of scalable embodied foundation models, yet existing methods often struggle with action-irrelevant distractors. Although incorporating action supervision can alleviate these distractions, its effectiveness is restricted by the scarcity of available action labels. Optical flow represents pixel-level motion between consecutive frames, naturally suppressing background elements and emphasizing moving objects. Motivated by this, we propose robust Latent Action learning with Optical Flow constraints, called LAOF, a pseudo-supervised framework that leverages the agent's optical flow as an action-driven signal to learn latent action representations robust to distractors. Experimental results show that the latent representations learned by LAOF outperform existing methods on downstream imitation learning and reinforcement learning tasks. This superior performance arises from optical flow constraints, which substantially stabilize training and improve the quality of latent representations under extremely label-scarce conditions, while remaining effective as the proportion of action labels increases to 10 percent. Importantly, even without action supervision, LAOF matches or surpasses action-supervised methods trained with 1 percent of action labels. 

**Abstract (ZH)**: 大规模视频中学习潜在动作对于可扩展的体感基础模型的预训练至关重要，但现有方法往往难以应对无关动作干扰。受此启发，我们提出了光学流约束下的鲁棒潜在动作学习方法（LAOF），这是一种伪监督框架，利用代理的光学流作为动作驱动信号，以在干扰下学习鲁棒的潜在动作表示。实验结果显示，LAOF 学习到的潜在表示在下游 imitation learning 和 reinforcement learning 任务中优于现有方法。这种优越性能源于光学流约束，它们在极端标签稀缺条件下显著稳定训练，并提高潜在表示的质量，即使动作标签比例增加到 10％ 时仍然有效。更重要的是，即使不使用动作监督，LAOF 也能匹配或超越使用 1％ 动作标签训练的动作监督方法。 

---
# Homogeneous Proportional-Integral-Derivative Controller in Mobile Robotic Manipulators 

**Title (ZH)**: 移动机器人 manipulator 中的 homogeneous proportional-integral-derivative 控制器 

**Authors**: Luis Luna, Isaac Chairez, Andrey Polyakov  

**Link**: [PDF](https://arxiv.org/pdf/2511.16406)  

**Abstract**: Mobile robotic manipulators (MRMs), which integrate mobility and manipulation capabilities, present significant control challenges due to their nonlinear dynamics, underactuation, and coupling between the base and manipulator subsystems. This paper proposes a novel homogeneous Proportional-Integral-Derivative (hPID) control strategy tailored for MRMs to achieve robust and coordinated motion control. Unlike classical PID controllers, the hPID controller leverages the mathematical framework of homogeneous control theory to systematically enhance the stability and convergence properties of the closed-loop system, even in the presence of dynamic uncertainties and external disturbances involved into a system in a homogeneous way. A homogeneous PID structure is designed, ensuring improved convergence of tracking errors through a graded homogeneity approach that generalizes traditional PID gains to nonlinear, state-dependent functions. Stability analysis is conducted using Lyapunov-based methods, demonstrating that the hPID controller guarantees global asymptotic stability and finite-time convergence under mild assumptions. Experimental results on a representative MRM model validate the effectiveness of the hPID controller in achieving high-precision trajectory tracking for both the mobile base and manipulator arm, outperforming conventional linear PID controllers in terms of response time, steady-state error, and robustness to model uncertainties. This research contributes a scalable and analytically grounded control framework for enhancing the autonomy and reliability of next-generation mobile manipulation systems in structured and unstructured environments. 

**Abstract (ZH)**: 移动机器人 manipulators (MRMs) 的协调控制：一种基于齐次控制理论的新型 hPID 控制策略 

---
# Robot Metacognition: Decision Making with Confidence for Tool Invention 

**Title (ZH)**: 机器人元认知：具有自信程度的工具发明决策making 

**Authors**: Ajith Anil Meera, Poppy Collis, Polina Arbuzova, Abián Torres, Paul F Kinghorn, Ricardo Sanz, Pablo Lanillos  

**Link**: [PDF](https://arxiv.org/pdf/2511.16390)  

**Abstract**: Robots today often miss a key ingredient of truly intelligent behavior: the ability to reflect on their own cognitive processes and decisions. In humans, this self-monitoring or metacognition is crucial for learning, decision making and problem solving. For instance, they can evaluate how confident they are in performing a task, thus regulating their own behavior and allocating proper resources. Taking inspiration from neuroscience, we propose a robot metacognition architecture centered on confidence (a second-order judgment on decisions) and we demonstrate it on the use case of autonomous tool invention. We propose the use of confidence as a metacognitive measure within the robot decision making scheme. Confidence-informed robots can evaluate the reliability of their decisions, improving their robustness during real-world physical deployment. This form of robotic metacognition emphasizes embodied action monitoring as a means to achieve better informed decisions. We also highlight potential applications and research directions for robot metacognition. 

**Abstract (ZH)**: 今天的机器人往往缺乏真正智能行为的一个关键要素：自我反思其认知过程和决策的能力。在人类中，这种自我监控或元认知对于学习、决策和问题解决至关重要。例如，他们可以评估自己执行任务的自信程度，从而调节自己的行为并分配适当的资源。借鉴神经科学的灵感，我们提出了一种以自信为核心（对决策的第二级判断）的机器人元认知架构，并在自主工具发明的应用场景中进行了演示。我们建议在机器人的决策方案中使用自信作为元认知指标。自信驱动的机器人能够评估自己决策的可靠性，从而在真实世界的物理部署中提高其鲁棒性。这种机器人的元认知形式强调通过体动监控来实现更明智的决策。我们还指出了机器人元认知的潜在应用和研究方向。 

---
# Flow-Aided Flight Through Dynamic Clutters From Point To Motion 

**Title (ZH)**: 流辅助通过动态杂波从点到运动的飞行 

**Authors**: Bowen Xu, Zexuan Yan, Minghao Lu, Xiyu Fan, Yi Luo, Youshen Lin, Zhiqiang Chen, Yeke Chen, Qiyuan Qiao, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16372)  

**Abstract**: Challenges in traversing dynamic clutters lie mainly in the efficient perception of the environmental dynamics and the generation of evasive behaviors considering obstacle movement. Previous solutions have made progress in explicitly modeling the dynamic obstacle motion for avoidance, but this key dependency of decision-making is time-consuming and unreliable in highly dynamic scenarios with occlusions. On the contrary, without introducing object detection, tracking, and prediction, we empower the reinforcement learning (RL) with single LiDAR sensing to realize an autonomous flight system directly from point to motion. For exteroception, a depth sensing distance map achieving fixed-shape, low-resolution, and detail-safe is encoded from raw point clouds, and an environment change sensing point flow is adopted as motion features extracted from multi-frame observations. These two are integrated into a lightweight and easy-to-learn representation of complex dynamic environments. For action generation, the behavior of avoiding dynamic threats in advance is implicitly driven by the proposed change-aware sensing representation, where the policy optimization is indicated by the relative motion modulated distance field. With the deployment-friendly sensing simulation and dynamics model-free acceleration control, the proposed system shows a superior success rate and adaptability to alternatives, and the policy derived from the simulator can drive a real-world quadrotor with safe maneuvers. 

**Abstract (ZH)**: 动态杂乱环境中自主飞行面临的挑战主要在于高效感知环境动态以及考虑障碍物运动生成规避行为。以往解决方案在明确建模动态障碍物运动以避免碰撞方面取得了进展，但在高度动态且存在遮挡的场景中，这一决策关键依赖性代价高昂且不可靠。相反，我们不引入物体检测、跟踪和预测，而是凭借单一激光雷达传感器实现从点云到运动的自主飞行系统，直接利用强化学习（RL）进行自主飞行。对于外部感知，采用固定形状、低分辨率且细节安全的距离图编码原始点云信息，并采用多帧观察中提取的环境变化感知点流作为运动特征。这两者被整合成一种轻量级且易于学习的复杂动态环境表示。对于行为生成，提出的感知变化感知表示隐式驱动提前规避动态威胁的行为，其中策略优化由相对运动调制距离场指示。通过部署友好的传感模拟和动力学模型驱动的加速控制，所提系统展示了优于替代方案的成功率和适应性，并且从模拟器中得出的策略可以驱动现实世界的四旋翼无人机进行安全机动。 

---
# Safe and Optimal Variable Impedance Control via Certified Reinforcement Learning 

**Title (ZH)**: 通过认证强化学习实现安全与最优的变阻抗控制 

**Authors**: Shreyas Kumar, Ravi Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2511.16330)  

**Abstract**: Reinforcement learning (RL) offers a powerful approach for robots to learn complex, collaborative skills by combining Dynamic Movement Primitives (DMPs) for motion and Variable Impedance Control (VIC) for compliant interaction. However, this model-free paradigm often risks instability and unsafe exploration due to the time-varying nature of impedance gains. This work introduces Certified Gaussian Manifold Sampling (C-GMS), a novel trajectory-centric RL framework that learns combined DMP and VIC policies while guaranteeing Lyapunov stability and actuator feasibility by construction. Our approach reframes policy exploration as sampling from a mathematically defined manifold of stable gain schedules. This ensures every policy rollout is guaranteed to be stable and physically realizable, thereby eliminating the need for reward penalties or post-hoc validation. Furthermore, we provide a theoretical guarantee that our approach ensures bounded tracking error even in the presence of bounded model errors and deployment-time uncertainties. We demonstrate the effectiveness of C-GMS in simulation and verify its efficacy on a real robot, paving the way for reliable autonomous interaction in complex environments. 

**Abstract (ZH)**: 强化学习(RL)通过结合动态运动片段(DMPs)进行运动控制和可变阻抗控制(VIC)进行 compliant 交互，提供了一种学习复杂协作技能的强大方法。然而，这种无模型的方法往往由于阻抗增益的时变性质而导致不稳定性和不安全的探索。本文提出了一种新的轨迹中心强化学习框架——认证高斯流形采样(C-GMS)，该框架能够在保证李亚普诺夫稳定性和执行器可行性的同时学习结合DMP和VIC的策略。我们通过采样数学定义的稳定增益调度流形来重新定义策略探索，确保每次策略 rollout 都是稳定的且物理上可实现的，从而消除了需要奖励惩罚或事后验证的需求。此外，我们提供了理论保证，即使在存在有界模型误差和部署时不确定性的情况下，我们的方法也能确保有界的跟踪误差。我们在仿真中验证了C-GMS的有效性，并在真实机器人上进行了验证，为其在复杂环境中的可靠自主交互铺平了道路。 

---
# InEKFormer: A Hybrid State Estimator for Humanoid Robots 

**Title (ZH)**: InEKFormer: 人型机器人的一种混合状态估计器 

**Authors**: Lasse Hohmeyer, Mihaela Popescu, Ivan Bergonzani, Dennis Mronga, Frank Kirchner  

**Link**: [PDF](https://arxiv.org/pdf/2511.16306)  

**Abstract**: Humanoid robots have great potential for a wide range of applications, including industrial and domestic use, healthcare, and search and rescue missions. However, bipedal locomotion in different environments is still a challenge when it comes to performing stable and dynamic movements. This is where state estimation plays a crucial role, providing fast and accurate feedback of the robot's floating base state to the motion controller. Although classical state estimation methods such as Kalman filters are widely used in robotics, they require expert knowledge to fine-tune the noise parameters. Due to recent advances in the field of machine learning, deep learning methods are increasingly used for state estimation tasks. In this work, we propose the InEKFormer, a novel hybrid state estimation method that incorporates an invariant extended Kalman filter (InEKF) and a Transformer network. We compare our method with the InEKF and the KalmanNet approaches on datasets obtained from the humanoid robot RH5. The results indicate the potential of Transformers in humanoid state estimation, but also highlight the need for robust autoregressive training in these high-dimensional problems. 

**Abstract (ZH)**: 类人机器人在工业和家庭应用、医疗保健以及搜救任务等领域具有广泛的应用潜力。然而，在不同环境下的双足行走仍然是一项挑战，尤其是在执行稳定和动态运动时。在这里，状态估计发挥着关键作用，为运动控制器提供快速而准确的机器人浮动基座状态反馈。虽然经典的卡尔曼滤波器等状态估计方法在机器人学中广泛应用，但它们需要专家知识来调整噪声参数。由于机器学习领域的最新进展，深度学习方法在状态估计任务中越来越受欢迎。在本文中，我们提出了一种名为InEKFormer的新型混合状态估计方法，该方法结合了不变扩展卡尔曼滤波器（InEKF）和Transformer网络。我们在来自类人机器人RH5的数据集上将我们的方法与InEKF和KalmanNet方法进行了比较。结果显示Transformer在类人状态估计中的潜力，但也指出了这些高维问题中鲁棒自回归训练的必要性。 

---
# Funabot-Upper: McKibben Actuated Haptic Suit Inducing Kinesthetic Perceptions in Trunk, Shoulder, Elbow, and Wrist 

**Title (ZH)**: Funabot-Upper: McKibben驱动的触感套装，诱导躯干、肩部、肘部和腕部的运动感知 

**Authors**: Haru Fukatsu, Ryoji Yasuda, Yuki Funabora, Shinji Doki  

**Link**: [PDF](https://arxiv.org/pdf/2511.16265)  

**Abstract**: This paper presents Funabot-Upper, a wearable haptic suit that enables users to perceive 14 upper-body motions, including those of the trunk, shoulder, elbow, and wrist. Inducing kinesthetic perception through wearable haptic devices has attracted attention, and various devices have been developed in the past. However, these have been limited to verifications on single body parts, and few have applied the same method to multiple body parts as well. In our previous study, we developed a technology that uses the contraction of artificial muscles to deform clothing in three dimensions. Using this technology, we developed a haptic suit that induces kinesthetic perception of 7 motions in multiple upper body. However, perceptual mixing caused by stimulating multiple human muscles has occurred between the shoulder and the elbow. In this paper, we established a new, simplified design policy and developed a novel haptic suit that induces kinesthetic perceptions in the trunk, shoulder, elbow, and wrist by stimulating joints and muscles independently. We experimentally demonstrated the induced kinesthetic perception and examined the relationship between stimulation and perceived kinesthetic perception under the new design policy. Experiments confirmed that Funabot-Upper successfully induces kinesthetic perception across multiple joints while reducing perceptual mixing observed in previous designs. The new suit improved recognition accuracy from 68.8% to 94.6% compared to the previous Funabot-Suit, demonstrating its superiority and potential for future haptic applications. 

**Abstract (ZH)**: 基于穿戴式触觉装置的Funabot-Upper：一种能够感知14种上肢运动的触觉外衣 

---
# How Robot Dogs See the Unseeable 

**Title (ZH)**: 机器人狗如何认知不可见之事 

**Authors**: Oliver Bimber, Karl Dietrich von Ellenrieder, Michael Haller, Rakesh John Amala Arokia Nathan, Gianni Lunardi, Marco Camurri, Mohamed Youssef, Santos Miguel Orozco Soto, Jeremy E. Niven  

**Link**: [PDF](https://arxiv.org/pdf/2511.16262)  

**Abstract**: Peering, a side-to-side motion used by animals to estimate distance through motion parallax, offers a powerful bio-inspired strategy to overcome a fundamental limitation in robotic vision: partial occlusion. Conventional robot cameras, with their small apertures and large depth of field, render both foreground obstacles and background objects in sharp focus, causing occluders to obscure critical scene information. This work establishes a formal connection between animal peering and synthetic aperture (SA) sensing from optical imaging. By having a robot execute a peering motion, its camera describes a wide synthetic aperture. Computational integration of the captured images synthesizes an image with an extremely shallow depth of field, effectively blurring out occluding elements while bringing the background into sharp focus. This efficient, wavelength-independent technique enables real-time, high-resolution perception across various spectral bands. We demonstrate that this approach not only restores basic scene understanding but also empowers advanced visual reasoning in large multimodal models, which fail with conventionally occluded imagery. Unlike feature-dependent multi-view 3D vision methods or active sensors like LiDAR, SA sensing via peering is robust to occlusion, computationally efficient, and immediately deployable on any mobile robot. This research bridges animal behavior and robotics, suggesting that peering motions for synthetic aperture sensing are a key to advanced scene understanding in complex, cluttered environments. 

**Abstract (ZH)**: 通过视差感知突破 robotic vision 部分遮挡限制的侧向窥视方法：合成孔径感知 

---
# FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models 

**Title (ZH)**: FT-NCFM：一种考虑影响因素的数据蒸馏框架以构建高效的多视角模型 

**Authors**: Kewei Chen, Yayu Long, Shuai Li, Mingsheng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16233)  

**Abstract**: The powerful generalization of Vision-Language-Action (VLA) models is bottlenecked by their heavy reliance on massive, redundant, and unevenly valued datasets, hindering their widespread application. Existing model-centric optimization paths, such as model compression (which often leads to performance degradation) or policy distillation (whose products are model-dependent and lack generality), fail to fundamentally address this data-level challenge. To this end, this paper introduces FT-NCFM, a fundamentally different, data-centric generative data distillation framework. Our framework employs a self-contained Fact-Tracing (FT) engine that combines causal attribution with programmatic contrastive verification to assess the intrinsic value of samples. Guided by these assessments, an adversarial NCFM process synthesizes a model-agnostic, information-dense, and reusable data asset. Experimental results on several mainstream VLA benchmarks show that models trained on just 5% of our distilled coreset achieve a success rate of 85-90% compared with training on the full dataset, while reducing training time by over 80%. Our work demonstrates that intelligent data distillation is a highly promising new path for building efficient, high-performance VLA models. 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型的强大泛化受限于其对大量冗余且价值不均数据集的依赖，阻碍了其广泛应用。现有的以模型为中心的优化路径，如模型压缩（常导致性能下降）或策略蒸馏（产物依赖于特定模型且缺乏通用性），未能从根本上解决这一数据层面的挑战。为此，本文引入了FT-NCFM，这是一种本质上不同的、以数据为中心的生成性数据蒸馏框架。我们的框架利用一个自包含的因果追溯（FT）引擎，结合因果归因与程序对比验证来评估样本的内在价值。在这些评估的指导下，对手过程合成一个模型无关、信息丰富且可重用的数据资产。在多个主流VLA基准测试上的实验结果显示，仅使用我们蒸馏的核心集5%的数据进行训练，模型的成功率达到了85-90%，同时将训练时间减少了80%以上。我们的工作证明，智能数据蒸馏是构建高效高性能VLA模型的一个极具前景的新途径。 

---
# DynaMimicGen: A Data Generation Framework for Robot Learning of Dynamic Tasks 

**Title (ZH)**: DynaMimicGen: 一种用于机器人动态任务学习的数据生成框架 

**Authors**: Vincenzo Pomponi, Paolo Franceschi, Stefano Baraldo, Loris Roveda, Oliver Avram, Luca Maria Gambardella, Anna Valente  

**Link**: [PDF](https://arxiv.org/pdf/2511.16223)  

**Abstract**: Learning robust manipulation policies typically requires large and diverse datasets, the collection of which is time-consuming, labor-intensive, and often impractical for dynamic environments. In this work, we introduce DynaMimicGen (D-MG), a scalable dataset generation framework that enables policy training from minimal human supervision while uniquely supporting dynamic task settings. Given only a few human demonstrations, D-MG first segments the demonstrations into meaningful sub-tasks, then leverages Dynamic Movement Primitives (DMPs) to adapt and generalize the demonstrated behaviors to novel and dynamically changing environments. Improving prior methods that rely on static assumptions or simplistic trajectory interpolation, D-MG produces smooth, realistic, and task-consistent Cartesian trajectories that adapt in real time to changes in object poses, robot states, or scene geometry during task execution. Our method supports different scenarios - including scene layouts, object instances, and robot configurations - making it suitable for both static and highly dynamic manipulation tasks. We show that robot agents trained via imitation learning on D-MG-generated data achieve strong performance across long-horizon and contact-rich benchmarks, including tasks like cube stacking and placing mugs in drawers, even under unpredictable environment changes. By eliminating the need for extensive human demonstrations and enabling generalization in dynamic settings, D-MG offers a powerful and efficient alternative to manual data collection, paving the way toward scalable, autonomous robot learning. 

**Abstract (ZH)**: 基于最小人类监督的动态仿真生成框架DynαMimicGen及其在动态任务设置下的应用 

---
# PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks 

**Title (ZH)**: PIPHEN：基于哈密顿能量网络的物理相互作用预测 

**Authors**: Kewei Chen, Yayu Long, Mingsheng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16200)  

**Abstract**: Multi-robot systems in complex physical collaborations face a "shared brain dilemma": transmitting high-dimensional multimedia data (e.g., video streams at ~30MB/s) creates severe bandwidth bottlenecks and decision-making latency. To address this, we propose PIPHEN, an innovative distributed physical cognition-control framework. Its core idea is to replace "raw data communication" with "semantic communication" by performing "semantic distillation" at the robot edge, reconstructing high-dimensional perceptual data into compact, structured physical representations. This idea is primarily realized through two key components: (1) a novel Physical Interaction Prediction Network (PIPN), derived from large model knowledge distillation, to generate this representation; and (2) a Hamiltonian Energy Network (HEN) controller, based on energy conservation, to precisely translate this representation into coordinated actions. Experiments show that, compared to baseline methods, PIPHEN can compress the information representation to less than 5% of the original data volume and reduce collaborative decision-making latency from 315ms to 76ms, while significantly improving task success rates. This work provides a fundamentally efficient paradigm for resolving the "shared brain dilemma" in resource-constrained multi-robot systems. 

**Abstract (ZH)**: 多机器人系统在复杂物理协作中面临的“共享大脑困境”：通过执行“语义蒸馏”在机器人边缘将高维感知数据重构为紧凑的结构化物理表示，提出一种创新的分布式物理认知-控制框架PIPHEN。 

---
# MagBotSim: Physics-Based Simulation and Reinforcement Learning Environments for Magnetic Robotics 

**Title (ZH)**: MagBotSim：基于物理的磁变形机器人模拟和强化学习环境 

**Authors**: Lara Bergmann, Cedric Grothues, Klaus Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2511.16158)  

**Abstract**: Magnetic levitation is about to revolutionize in-machine material flow in industrial automation. Such systems are flexibly configurable and can include a large number of independently actuated shuttles (movers) that dynamically rebalance production capacity. Beyond their capabilities for dynamic transportation, these systems possess the inherent yet unexploited potential to perform manipulation. By merging the fields of transportation and manipulation into a coordinated swarm of magnetic robots (MagBots), we enable manufacturing systems to achieve significantly higher efficiency, adaptability, and compactness. To support the development of intelligent algorithms for magnetic levitation systems, we introduce MagBotSim (Magnetic Robotics Simulation): a physics-based simulation for magnetic levitation systems. By framing magnetic levitation systems as robot swarms and providing a dedicated simulation, this work lays the foundation for next generation manufacturing systems powered by Magnetic Robotics. MagBotSim's documentation, videos, experiments, and code are available at: this https URL 

**Abstract (ZH)**: 磁悬浮即将革新工业自动化中的在机物料流动。此类系统可灵活配置，并可包括大量独立驱动的小车（搬运器），动态调整生产产能。除了动态运输能力，这些系统还具备未充分利用的操控潜力。通过将运输与操控领域合并为一个协调的磁机器人群（MagBots），我们使制造系统实现了显著更高的效率、适应性和紧凑性。为支持磁悬浮系统智能算法的发展，我们引入了MagBotSim（磁控机器人仿真）：一种基于物理的磁悬浮系统仿真工具。通过将磁悬浮系统视为机器人集群，并提供专门的仿真工具，本工作为由磁控机器人驱动的下一代制造系统奠定了基础。MagBotSim的文档、视频、实验和代码可在以下链接获取：this https URL。 

---
# Bi-AQUA: Bilateral Control-Based Imitation Learning for Underwater Robot Arms via Lighting-Aware Action Chunking with Transformers 

**Title (ZH)**: 双边控制基于的模仿学习：通过照明感知的动作切片与变换器技术的水下机器人臂控制 

**Authors**: Takeru Tsunoori, Masato Kobayashi, Yuki Uranishi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16050)  

**Abstract**: Underwater robotic manipulation is fundamentally challenged by extreme lighting variations, color distortion, and reduced visibility. We introduce Bi-AQUA, the first underwater bilateral control-based imitation learning framework that integrates lighting-aware visual processing for underwater robot arms. Bi-AQUA employs a hierarchical three-level lighting adaptation mechanism: a Lighting Encoder that extracts lighting representations from RGB images without manual annotation and is implicitly supervised by the imitation objective, FiLM modulation of visual backbone features for adaptive, lighting-aware feature extraction, and an explicit lighting token added to the transformer encoder input for task-aware conditioning. Experiments on a real-world underwater pick-and-place task under diverse static and dynamic lighting conditions show that Bi-AQUA achieves robust performance and substantially outperforms a bilateral baseline without lighting modeling. Ablation studies further confirm that all three lighting-aware components are critical. This work bridges terrestrial bilateral control-based imitation learning and underwater manipulation, enabling force-sensitive autonomous operation in challenging marine environments. For additional material, please check: this https URL 

**Abstract (ZH)**: 水下机器人操作基本受到极端光照变化、颜色失真和能见度降低的挑战。我们引入了Bi-AQUA，这是一种将光照意识视觉处理整合到水下机器人手臂中的第一种基于双边控制的模仿学习框架。Bi-AQUA 雇佣了一个分层的三级光照适应机制：一个光照编码器，从RGB图像中提取光照表示，无需人工标注，并由模仿目标隐式监督；基于FiLM的视觉主干特征调制，实现适应性的、光照意识的特征提取；以及添加到变换器编码器输入中的显式的光照标记，实现任务意识的条件控制。在不同静态和动态光照条件下的实际水下取放任务实验表明，Bi-AQUA 达到了稳健的性能，并显著优于没有光照建模的双边基线。进一步的消融研究证实，所有三个光照意识组件都是必不可少的。这项工作将陆地上的基于双边控制的模仿学习与水下操作相结合，使在具有挑战性的海洋环境中实现力感知的自主操作成为可能。 

---
# Semantic Glitch: Agency and Artistry in an Autonomous Pixel Cloud 

**Title (ZH)**: 语义漏洞：自主像素云中的代理与艺术istry 

**Authors**: Qing Zhang, Jing Huang, Mingyang Xu, Jun Rekimoto  

**Link**: [PDF](https://arxiv.org/pdf/2511.16048)  

**Abstract**: While mainstream robotics pursues metric precision and flawless performance, this paper explores the creative potential of a deliberately "lo-fi" approach. We present the "Semantic Glitch," a soft flying robotic art installation whose physical form, a 3D pixel style cloud, is a "physical glitch" derived from digital archaeology. We detail a novel autonomous pipeline that rejects conventional sensors like LiDAR and SLAM, relying solely on the qualitative, semantic understanding of a Multimodal Large Language Model to navigate. By authoring a bio-inspired personality for the robot through a natural language prompt, we create a "narrative mind" that complements the "weak," historically, loaded body. Our analysis begins with a 13-minute autonomous flight log, and a follow-up study statistically validates the framework's robustness for authoring quantifiably distinct personas. The combined analysis reveals emergent behaviors, from landmark-based navigation to a compelling "plan to execution" gap, and a character whose unpredictable, plausible behavior stems from a lack of precise proprioception. This demonstrates a lo-fi framework for creating imperfect companions whose success is measured in character over efficiency. 

**Abstract (ZH)**: 虽然主流机器人追求度量上的精确和完美性能，本文探讨了一种故意采用“低分辨率”方法的创造潜力。我们介绍了“语义错误”，这是一种软体飞行机器人艺术装置，其物理形态是源自数字考古的3D像素风格云，是一种“物理错误”。我们详细阐述了一种新颖的自主管道，该管道拒绝使用传统的传感器如LiDAR和SLAM，仅依赖多模态大语言模型的定性语义理解进行导航。通过通过自然语言提示为机器人编写生物启发式个性，我们创造了一个“叙事心智”，这与历史上负担沉重但功能较弱的身体相补充。我们的分析从一个13分钟的自主飞行日志开始，并进行后续研究以统计验证该框架在编写可量化不同的个性方面的鲁棒性。结合分析揭示了从基于地标导航到引人入胜的“计划到执行”缺口等一系列新兴行为，一个行为不可预测但合理的角色来自缺乏精确本体感受。这证明了一种低分辨率框架，用于创建在性格而非效率上成功的不完美同伴。 

---
# PushingBots: Collaborative Pushing via Neural Accelerated Combinatorial Hybrid Optimization 

**Title (ZH)**: 推盒机器人：基于神经加速组合混合优化的协作推盒 

**Authors**: Zili Tang, Ying Zhang, Meng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.15995)  

**Abstract**: Many robots are not equipped with a manipulator and many objects are not suitable for prehensile manipulation (such as large boxes and cylinders). In these cases, pushing is a simple yet effective non-prehensile skill for robots to interact with and further change the environment. Existing work often assumes a set of predefined pushing modes and fixed-shape objects. This work tackles the general problem of controlling a robotic fleet to push collaboratively numerous arbitrary objects to respective destinations, within complex environments of cluttered and movable obstacles. It incorporates several characteristic challenges for multi-robot systems such as online task coordination under large uncertainties of cost and duration, and for contact-rich tasks such as hybrid switching among different contact modes, and under-actuation due to constrained contact forces. The proposed method is based on combinatorial hybrid optimization over dynamic task assignments and hybrid execution via sequences of pushing modes and associated forces. It consists of three main components: (I) the decomposition, ordering and rolling assignment of pushing subtasks to robot subgroups; (II) the keyframe guided hybrid search to optimize the sequence of parameterized pushing modes for each subtask; (III) the hybrid control to execute these modes and transit among them. Last but not least, a diffusion-based accelerator is adopted to predict the keyframes and pushing modes that should be prioritized during hybrid search; and further improve planning efficiency. The framework is complete under mild assumptions. Its efficiency and effectiveness under different numbers of robots and general-shaped objects are validated extensively in simulations and hardware experiments, as well as generalizations to heterogeneous robots, planar assembly and 6D pushing. 

**Abstract (ZH)**: 一种基于组合混合优化的多机器人协作推动物体方法及其应用 

---
# The Role of Consequential and Functional Sound in Human-Robot Interaction: Toward Audio Augmented Reality Interfaces 

**Title (ZH)**: 功能音与后果音在人机交互中的作用：通往音频增强现实界面的研究 

**Authors**: Aliyah Smith, Monroe Kennedy III  

**Link**: [PDF](https://arxiv.org/pdf/2511.15956)  

**Abstract**: As robots become increasingly integrated into everyday environments, understanding how they communicate with humans is critical. Sound offers a powerful channel for interaction, encompassing both operational noises and intentionally designed auditory cues. In this study, we examined the effects of consequential and functional sounds on human perception and behavior, including a novel exploration of spatial sound through localization and handover tasks. Results show that consequential sounds of the Kinova Gen3 manipulator did not negatively affect perceptions, spatial localization is highly accurate for lateral cues but declines for frontal cues, and spatial sounds can simultaneously convey task-relevant information while promoting warmth and reducing discomfort. These findings highlight the potential of functional and transformative auditory design to enhance human-robot collaboration and inform future sound-based interaction strategies. 

**Abstract (ZH)**: 随着机器人越来越多地集成到日常环境中，理解它们与人类的沟通方式至关重要。声音提供了一种强大的互动通道，涵盖了操作噪声和故意设计的听觉提示。在本研究中，我们探讨了意义性和功能性声音对人类感知和行为的影响，包括通过定位和传递任务对空间声音的全新探索。研究结果显示，Kinova Gen3 manipulator的意义性声音并没有负面影响感知，侧向线索的空间定位非常准确但对前方线索的定位准确性下降，空间声音可以同时传递相关信息并促进温暖感、减少不适感。这些发现强调了功能性及变革性听觉设计的潜在价值，以提升人机协作，并指导未来基于声音的交互策略。 

---
# I've Changed My Mind: Robots Adapting to Changing Human Goals during Collaboration 

**Title (ZH)**: 我改变了主意：机器人在协作过程中适应变化的人类目标 

**Authors**: Debasmita Ghose, Oz Gitelson, Ryan Jin, Grace Abawe, Marynel Vazquez, Brian Scassellati  

**Link**: [PDF](https://arxiv.org/pdf/2511.15914)  

**Abstract**: For effective human-robot collaboration, a robot must align its actions with human goals, even as they change mid-task. Prior approaches often assume fixed goals, reducing goal prediction to a one-time inference. However, in real-world scenarios, humans frequently shift goals, making it challenging for robots to adapt without explicit communication. We propose a method for detecting goal changes by tracking multiple candidate action sequences and verifying their plausibility against a policy bank. Upon detecting a change, the robot refines its belief in relevant past actions and constructs Receding Horizon Planning (RHP) trees to actively select actions that assist the human while encouraging Differentiating Actions to reveal their updated goal. We evaluate our approach in a collaborative cooking environment with up to 30 unique recipes and compare it to three comparable human goal prediction algorithms. Our method outperforms all baselines, quickly converging to the correct goal after a switch, reducing task completion time, and improving collaboration efficiency. 

**Abstract (ZH)**: 有效的人机协作中，机器人必须在其动作与人类目标发生变化时保持一致，我们提出了一种通过跟踪多个候选动作序列并验证其可行性来检测目标变化的方法。检测到变化后，机器人对其相关过去的动作进行信念细化，并构建回退水平计划（RHP）树，以积极选择有助于人类的动作并鼓励披露其更新后的目标。我们在此类协作烹饪环境中评估了该方法，该环境包含多达30种独特的食谱，并将其与三种可比的人类目标预测算法进行了比较。我们的方法优于所有基线，能够在目标切换后迅速收敛到正确的目标，减少任务完成时间，提高协作效率。 

---
# Gimballed Rotor Mechanism for Omnidirectional Quadrotors 

**Title (ZH)**: Giancan 型旋翼机构Omnidirectional 四旋翼飞行器 

**Authors**: J. Cristobal, A. Z. Zain Aldeen, M. Izadi, R. Faieghi  

**Link**: [PDF](https://arxiv.org/pdf/2511.15909)  

**Abstract**: This paper presents the design of a gimballed rotor mechanism as a modular and efficient solution for constructing omnidirectional quadrotors. Unlike conventional quadrotors, which are underactuated, this class of quadrotors achieves full actuation, enabling independent motion in all six degrees of freedom. While existing omnidirectional quadrotor designs often require significant structural modifications, the proposed gimballed rotor system maintains a lightweight and easy-to-integrate design by incorporating servo motors within the rotor platforms, allowing independent tilting of each rotor without major alterations to the central structure of a quadrotor. To accommodate this unconventional design, we develop a new control allocation scheme in PX4 Autopilot and present successful flight tests, validating the effectiveness of the proposed approach. 

**Abstract (ZH)**: 本文提出了一种转台式旋翼机制的设计，作为一种模块化和高效的方法用于构建全向四旋翼。与传统四旋翼相比，这类四旋翼实现了全驱动，能够在六自由度上独立运动。虽然现有全向四旋翼设计往往需要较大的结构修改，但提出的设计通过在旋翼平台内集成伺服电机，保持了轻量化和易于集成的特性，允许每个旋翼独立倾斜而无需对四旋翼中心结构进行重大改动。为适应这种非传统设计，我们开发了一种新的控制分配方案并在PX4 自动飞行控制系统中实现，并通过成功的飞行测试验证了所提方法的有效性。 

---
# Green Resilience of Cyber-Physical Systems: Doctoral Dissertation 

**Title (ZH)**: cyber-物理系统绿色韧性：博士 dissertation 

**Authors**: Diaeddin Rimawi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16593)  

**Abstract**: Cyber-physical systems (CPS) combine computational and physical components. Online Collaborative AI System (OL-CAIS) is a type of CPS that learn online in collaboration with humans to achieve a common goal, which makes it vulnerable to disruptive events that degrade performance. Decision-makers must therefore restore performance while limiting energy impact, creating a trade-off between resilience and greenness. This research addresses how to balance these two properties in OL-CAIS. It aims to model resilience for automatic state detection, develop agent-based policies that optimize the greenness-resilience trade-off, and understand catastrophic forgetting to maintain performance consistency. We model OL-CAIS behavior through three operational states: steady, disruptive, and final. To support recovery during disruptions, we introduce the GResilience framework, which provides recovery strategies through multi-objective optimization (one-agent), game-theoretic decision-making (two-agent), and reinforcement learning (RL-agent). We also design a measurement framework to quantify resilience and greenness. Empirical evaluation uses real and simulated experiments with a collaborative robot learning object classification from human demonstrations. Results show that the resilience model captures performance transitions during disruptions, and that GResilience policies improve green recovery by shortening recovery time, stabilizing performance, and reducing human dependency. RL-agent policies achieve the strongest results, although with a marginal increase in CO2 emissions. We also observe catastrophic forgetting after repeated disruptions, while our policies help maintain steadiness. A comparison with containerized execution shows that containerization cuts CO2 emissions by half. Overall, this research provides models, metrics, and policies that ensure the green recovery of OL-CAIS. 

**Abstract (ZH)**: 基于物理系统的在线协作AI系统中平衡韧性和绿色性的研究 

---
# The Shawshank Redemption of Embodied AI: Understanding and Benchmarking Indirect Environmental Jailbreaks 

**Title (ZH)**: 实体人工智能的 Shawshank 解救：理解与基准测试间接环境监狱逃脱 

**Authors**: Chunyang Li, Zifeng Kang, Junwei Zhang, Zhuo Ma, Anda Cheng, Xinghua Li, Jianfeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.16347)  

**Abstract**: The adoption of Vision-Language Models (VLMs) in embodied AI agents, while being effective, brings safety concerns such as jailbreaking. Prior work have explored the possibility of directly jailbreaking the embodied agents through elaborated multi-modal prompts. However, no prior work has studied or even reported indirect jailbreaks in embodied AI, where a black-box attacker induces a jailbreak without issuing direct prompts to the embodied agent. In this paper, we propose, for the first time, indirect environmental jailbreak (IEJ), a novel attack to jailbreak embodied AI via indirect prompt injected into the environment, such as malicious instructions written on a wall. Our key insight is that embodied AI does not ''think twice'' about the instructions provided by the environment -- a blind trust that attackers can exploit to jailbreak the embodied agent. We further design and implement open-source prototypes of two fully-automated frameworks: SHAWSHANK, the first automatic attack generation framework for the proposed attack IEJ; and SHAWSHANK-FORGE, the first automatic benchmark generation framework for IEJ. Then, using SHAWSHANK-FORGE, we automatically construct SHAWSHANK-BENCH, the first benchmark for indirectly jailbreaking embodied agents. Together, our two frameworks and one benchmark answer the questions of what content can be used for malicious IEJ instructions, where they should be placed, and how IEJ can be systematically evaluated. Evaluation results show that SHAWSHANK outperforms eleven existing methods across 3,957 task-scene combinations and compromises all six tested VLMs. Furthermore, current defenses only partially mitigate our attack, and we have responsibly disclosed our findings to all affected VLM vendors. 

**Abstract (ZH)**: Vision-Language模型在体感人工智能代理中的应用虽有效，但带来了如 Jailbreaking 等安全问题。前人工作探讨了通过复杂的多模态提示直接 Jailbreaking 体感代理的可能性，但尚未研究或报告间接 Jailbreak，即黑盒攻击者诱导 Jailbreak 而不直接向体感代理发出指令的情况。本文首次提出了间接环境 Jailbreak (IEJ)，通过将恶意指令等间接提示注入环境来 Jailbreak 体感 AI 的新颖攻击方式。我们的核心洞察是，体感 AI 对环境提供的指令毫不怀疑，攻击者可以利用这种信任来进行 Jailbreak。我们进一步设计并实现了两个全自动化框架的开源原型：SHAWSHANK，首个用于提出这种攻击 IEJ 的自动化攻击生成框架；SHAWSHANK-FORGE，首个用于生成 IEJ 基准的自动化基准生成框架。然后，使用 SHAWSHANK-FORGE 自动构建了 SHAWSHANK-BENCH，首个用于间接 Jailbreak 体感代理的基准。两个框架和一个基准回答了可用于恶意 IEJ 指令的内容、放置位置以及如何系统地评估 IEJ 的问题。评估结果显示，SHAWSHANK 在 3,957 任务-场景组合中优于现有方法中的 11 种，并攻击了所有六个测试的 VLMs。此外，当前防御措施只能部分缓解我们的攻击，我们已负责任地向所有受影响的 VLM 供应商披露了我们的发现。 

---
# LEGO-SLAM: Language-Embedded Gaussian Optimization SLAM 

**Title (ZH)**: LEGO-SLAM：嵌入语言的高斯优化SLAM 

**Authors**: Sibaek Lee, Seongbo Ha, Kyeongsu Kang, Joonyeol Choi, Seungjun Tak, Hyeonwoo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16144)  

**Abstract**: Recent advances in 3D Gaussian Splatting (3DGS) have enabled Simultaneous Localization and Mapping (SLAM) systems to build photorealistic maps. However, these maps lack the open-vocabulary semantic understanding required for advanced robotic interaction. Integrating language features into SLAM remains a significant challenge, as storing high-dimensional features demands excessive memory and rendering overhead, while existing methods with static models lack adaptability for novel environments. To address these limitations, we propose LEGO-SLAM (Language-Embedded Gaussian Optimization SLAM), the first framework to achieve real-time, open-vocabulary mapping within a 3DGS-based SLAM system. At the core of our method is a scene-adaptive encoder-decoder that distills high-dimensional language embeddings into a compact 16-dimensional feature space. This design reduces the memory per Gaussian and accelerates rendering, enabling real-time performance. Unlike static approaches, our encoder adapts online to unseen scenes. These compact features also enable a language-guided pruning strategy that identifies semantic redundancy, reducing the map's Gaussian count by over 60\% while maintaining rendering quality. Furthermore, we introduce a language-based loop detection approach that reuses these mapping features, eliminating the need for a separate detection model. Extensive experiments demonstrate that LEGO-SLAM achieves competitive mapping quality and tracking accuracy, all while providing open-vocabulary capabilities at 15 FPS. 

**Abstract (ZH)**: Recent Advances in Language-Embedded Gaussian Optimization SLAM for Real-Time Open-Vocabulary Mapping 

---
# Heterogeneous Stroke: Using Unique Vibration Cues to Improve the Wrist-Worn Spatiotemporal Tactile Display 

**Title (ZH)**: 异质性触觉拓扑：利用独特的振动线索改进腕戴式时空触觉显示 

**Authors**: Taejun Kim, Youngbo Aram Shim, Geehyuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.16133)  

**Abstract**: Beyond a simple notification of incoming calls or messages, more complex information such as alphabets and digits can be delivered through spatiotemporal tactile patterns (STPs) on a wrist-worn tactile display (WTD) with multiple tactors. However, owing to the limited skin area and spatial acuity of the wrist, frequent confusions occur between closely located tactors, resulting in a low recognition accuracy. Furthermore, the accuracies reported in previous studies have mostly been measured for a specific posture and could further decrease with free arm postures in real life. Herein, we present Heterogeneous Stroke, a design concept for improving the recognition accuracy of STPs on a WTD. By assigning unique vibrotactile stimuli to each tactor, the confusion between tactors can be reduced. Through our implementation of Heterogeneous Stroke, the alphanumeric characters could be delivered with high accuracy (93.8% for 26 alphabets and 92.4% for 10 digits) across different arm postures. 

**Abstract (ZH)**: 超越简单的来电或消息通知，腕戴触觉显示设备（WTD）上的多个振动器可以通过时空触觉模式（STPs）传递字母和数字等更复杂的信息。然而，由于手腕有限的皮肤面积和空间分辨率，靠近的振动器之间容易发生混淆，导致较低的识别准确率。此外，先前研究中报告的准确率多是在特定姿势下测量的，在实际自由手臂状态下可能会进一步下降。在此，我们提出了一种异构笔画设计概念，以提高腕戴触觉显示设备上时空触觉模式的识别准确率。通过为每个振动器分配独特的振动刺激，可以减少振动器之间的混淆。通过我们的异构笔画实现，字母和数字可以跨不同手臂姿势以高准确率（26个字母为93.8%，10个数字为92.4%）传递。 

---
# Towards a Safer and Sustainable Manufacturing Process: Material classification in Laser Cutting Using Deep Learning 

**Title (ZH)**: 基于深度学习的激光切割中材料分类：朝着更安全和可持续的制造过程努力 

**Authors**: Mohamed Abdallah Salem, Hamdy Ahmed Ashur, Ahmed Elshinnawy  

**Link**: [PDF](https://arxiv.org/pdf/2511.16026)  

**Abstract**: Laser cutting is a widely adopted technology in material processing across various industries, but it generates a significant amount of dust, smoke, and aerosols during operation, posing a risk to both the environment and workers' health. Speckle sensing has emerged as a promising method to monitor the cutting process and identify material types in real-time. This paper proposes a material classification technique using a speckle pattern of the material's surface based on deep learning to monitor and control the laser cutting process. The proposed method involves training a convolutional neural network (CNN) on a dataset of laser speckle patterns to recognize distinct material types for safe and efficient cutting. Previous methods for material classification using speckle sensing may face issues when the color of the laser used to produce the speckle pattern is changed. Experiments conducted in this study demonstrate that the proposed method achieves high accuracy in material classification, even when the laser color is changed. The model achieved an accuracy of 98.30 % on the training set and 96.88% on the validation set. Furthermore, the model was evaluated on a set of 3000 new images for 30 different materials, achieving an F1-score of 0.9643. The proposed method provides a robust and accurate solution for material-aware laser cutting using speckle sensing. 

**Abstract (ZH)**: 基于 speckle 模式和深度学习的材料分类方法在激光切割过程监测与控制中的应用 

---
