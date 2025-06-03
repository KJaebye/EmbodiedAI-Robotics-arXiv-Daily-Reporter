# Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning 

**Title (ZH)**: 快入慢思：一种结合快速操作与缓慢推理的双系统基础模型 

**Authors**: Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Renrui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01953)  

**Abstract**: Generalized policy and execution efficiency constitute the two critical challenges in robotic manipulation. While recent foundation policies benefit from the common-sense reasoning capabilities of internet-scale pretrained vision-language models (VLMs), they often suffer from low execution frequency. To mitigate this dilemma, dual-system approaches, inspired by Kahneman's theory, have been proposed to leverage a VLM-based System 2 model handling high-level reasoning and a separate System 1 action model ensuring real-time control. However, existing designs maintain both systems as separate models, limiting System 1 from fully leveraging the rich pretrained knowledge from the VLM-based System 2. In this work, we propose Fast-in-Slow (FiS), a unified dual-system vision-language-action (VLA) model that embeds the System 1 execution module within the VLM-based System 2 by partially sharing parameters. This innovative paradigm not only enables high-frequency execution in System 1 but also facilitates coordination between the reasoning and execution components within a single foundation model of System 2. Given their fundamentally distinct roles within FiS-VLA, we design the two systems to incorporate heterogeneous modality inputs alongside asynchronous operating frequencies, enabling both fast and precise manipulation. To enable coordination between the two systems, a dual-aware co-training strategy is proposed that equips System 1 with action generation capabilities while preserving System 2's contextual reasoning representation. For evaluation, FiS-VLA outperforms previous state-of-the-art methods by 8% in simulation and 11% in real-world tasks in terms of average success rate, while achieving a 117.7 Hz control frequency with action chunk set to eight. Project web page: this http URL. 

**Abstract (ZH)**: 广义的政策和执行效率是机器人操作面临的两大关键挑战。尽管近期的基础政策得益于互联网规模预训练视觉-语言模型（VLMs）的常识推理能力，但通常执行频率较低。为缓解这一问题，受Kahneman理论启发的双系统方法提出了利用VLM为基础的System 2模型处理高层推理，以及独立的System 1动作模型确保实时控制。然而，现有设计将两个系统保持为独立模型，限制了System 1从VLM为基础的System 2的丰富预训练知识中充分利用。在本工作中，我们提出了Fast-in-Slow（FiS），一种统一的双系统视觉-语言-动作（VLA）模型，通过部分共享参数将System 1执行模块嵌入到VLM为基础的System 2中。这一创新范式不仅使System 1能够实现高频执行，还促进了System 2内推理和执行组件之间的协调。鉴于FiS-VLA中的两个系统具有根本不同的角色，我们设计两个系统集成异质模态输入和异步操作频率，实现即快又准的操纵。为了在两个系统之间实现协调，提出了一种双系统意识的协同训练策略，使System 1具备动作生成能力，同时保留System 2的上下文推理表示。在评估中，FiS-VLA在仿真任务中平均成功率上比之前的方法高出8%，在真实世界任务中高出11%，且动作块集设置为八个时，实现了117.7 Hz的控制频率。项目网页：this http URL。 

---
# ADEPT: Adaptive Diffusion Environment for Policy Transfer Sim-to-Real 

**Title (ZH)**: ADEPT: 自适应扩散环境用于政策传输的模拟到现实转化 

**Authors**: Youwei Yu, Junhong Xu, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01759)  

**Abstract**: Model-free reinforcement learning has emerged as a powerful method for developing robust robot control policies capable of navigating through complex and unstructured environments. The effectiveness of these methods hinges on two essential elements: (1) the use of massively parallel physics simulations to expedite policy training, and (2) an environment generator tasked with crafting sufficiently challenging yet attainable environments to facilitate continuous policy improvement. Existing methods of outdoor environment generation often rely on heuristics constrained by a set of parameters, limiting the diversity and realism. In this work, we introduce ADEPT, a novel \textbf{A}daptive \textbf{D}iffusion \textbf{E}nvironment for \textbf{P}olicy \textbf{T}ransfer in the zero-shot sim-to-real fashion that leverages Denoising Diffusion Probabilistic Models to dynamically expand existing training environments by adding more diverse and complex environments adaptive to the current policy. ADEPT guides the diffusion model's generation process through initial noise optimization, blending noise-corrupted environments from existing training environments weighted by the policy's performance in each corresponding environment. By manipulating the noise corruption level, ADEPT seamlessly transitions between generating similar environments for policy fine-tuning and novel ones to expand training diversity. To benchmark ADEPT in off-road navigation, we propose a fast and effective multi-layer map representation for wild environment generation. Our experiments show that the policy trained by ADEPT outperforms both procedural generated and natural environments, along with popular navigation methods. 

**Abstract (ZH)**: 无需生成标题，以下是翻译内容：

无模型强化学习已成为开发能够在复杂和未结构化环境中导航的稳健机器人控制策略的强大方法。这些方法的有效性依赖于两个关键要素：（1）使用大规模并行物理模拟加速策略训练；（2）负责生成足够具有挑战性但又可行的环境以促进连续策略改进的环境生成器。现有的户外环境生成方法通常依赖受参数集约束的经验规则，这限制了环境的多样性和逼真度。在此工作中，我们提出了ADEPT，一种新颖的自适应扩散环境，用于零样本模拟到现实的策略转移，它利用去噪扩散概率模型动态扩展现有训练环境，添加更多多样和复杂、适应当前策略的环境。ADEPT通过初始噪声优化引导扩散模型的生成过程，通过根据每个环境中的政策表现加权融合现有训练环境中的噪声污染环境。通过调整噪声污染水平，ADEPT可以无缝地在为策略微调生成相似环境和为扩展训练多样性生成新颖环境之间过渡。为了在无路导航中测试ADEPT，我们提出了一种快速有效的多层地图表示方法，用于自然环境生成。实验结果显示，由ADEPT训练的策略在性能上优于程序生成和自然环境，以及流行的导航方法。 

---
# Learning with pyCub: A New Simulation and Exercise Framework for Humanoid Robotics 

**Title (ZH)**: 基于 pyCub 的新型人形机器人模拟与练习框架 

**Authors**: Lukas Rustler, Matej Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.01756)  

**Abstract**: We present pyCub, an open-source physics-based simulation of the humanoid robot iCub, along with exercises to teach students the basics of humanoid robotics. Compared to existing iCub similators (iCub SIM, iCub Gazebo), which require C++ code and YARP as middleware, pyCub works without YARP and with Python code. The complete robot with all articulations has been simulated, with two cameras in the eyes and the unique sensitive skin of the iCub comprising 4000 receptors on its body surface. The exercises range from basic control of the robot in velocity, joint, and Cartesian space to more complex tasks like gazing, grasping, or reactive control. The whole framework is written and controlled with Python, thus allowing to be used even by people with small or almost no programming practice. The exercises can be scaled to different difficulty levels. We tested the framework in two runs of a course on humanoid robotics. The simulation, exercises, documentation, Docker images, and example videos are publicly available at this https URL. 

**Abstract (ZH)**: 我们介绍了基于物理的iCub人形机器人开源模拟器pyCub及其相关练习，用以教授人形机器人基础。相比现有的iCub模拟器（iCub SIM, iCub Gazebo），pyCub无需YARP中间件且使用Python代码。整个机器人包括所有关节以及iCub特有的眼球中的两个相机和身体表面的4000个敏感受体。练习内容从基本的速度控制、关节控制以及笛卡尔空间控制，到复杂的任务如注视、抓取或反应控制。整个框架使用Python编写和控制，因此即便是编程经验较少的人也可以使用。练习可根据难度级别进行调整。我们在人形机器人课程的两次运行中测试了该框架。模拟器、练习、文档、Docker镜像和示例视频均可在该网址公开获取。 

---
# WoMAP: World Models For Embodied Open-Vocabulary Object Localization 

**Title (ZH)**: WoMAP：世界模型在具身开放词汇对象定位中的应用 

**Authors**: Tenny Yin, Zhiting Mei, Tao Sun, Lihan Zha, Emily Zhou, Jeremy Bao, Miyu Yamane, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.01600)  

**Abstract**: Language-instructed active object localization is a critical challenge for robots, requiring efficient exploration of partially observable environments. However, state-of-the-art approaches either struggle to generalize beyond demonstration datasets (e.g., imitation learning methods) or fail to generate physically grounded actions (e.g., VLMs). To address these limitations, we introduce WoMAP (World Models for Active Perception): a recipe for training open-vocabulary object localization policies that: (i) uses a Gaussian Splatting-based real-to-sim-to-real pipeline for scalable data generation without the need for expert demonstrations, (ii) distills dense rewards signals from open-vocabulary object detectors, and (iii) leverages a latent world model for dynamics and rewards prediction to ground high-level action proposals at inference time. Rigorous simulation and hardware experiments demonstrate WoMAP's superior performance in a broad range of zero-shot object localization tasks, with more than 9x and 2x higher success rates compared to VLM and diffusion policy baselines, respectively. Further, we show that WoMAP achieves strong generalization and sim-to-real transfer on a TidyBot. 

**Abstract (ZH)**: 基于语言指示的主动物体定位是机器人面临的關鍵挑戰，需要高效探索部分可观测环境。然而，最先进的方法要么难以泛化到演示数据集之外（例如，模仿学习方法），要么生成不出物理上合理的动作（例如，VLMs）。为此，我们提出了WoMAP（World Models for Active Perception）：一种用于训练开放词汇物体定位策略的方法，包括：(i) 使用基于高斯点积的实时到模拟再到实时的数据生成管道，无需专家演示；(ii) 从开放词汇物体检测器中提取密集奖励信号；(iii) 利用潜在世界模型进行动力学和奖励预测，在推理时使高层动作提案具有物理意义。严格的仿真和硬件实验表明，WoMAP在多种零样本物体定位任务中的性能优于VLM和扩散策略基线，成功率分别高出9倍和2倍。此外，我们展示了WoMAP在TidyBot上实现了强大的泛化能力和模拟到现实的迁移。 

---
# FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens 

**Title (ZH)**: FreqPolicy: 频率自回归visuomotor策略与连续_token_表示 

**Authors**: Yiming Zhong, Yumeng Liu, Chuyang Xiao, Zemin Yang, Youzhuo Wang, Yufei Zhu, Ye Shi, Yujing Sun, Xinge Zhu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.01583)  

**Abstract**: Learning effective visuomotor policies for robotic manipulation is challenging, as it requires generating precise actions while maintaining computational efficiency. Existing methods remain unsatisfactory due to inherent limitations in the essential action representation and the basic network architectures. We observe that representing actions in the frequency domain captures the structured nature of motion more effectively: low-frequency components reflect global movement patterns, while high-frequency components encode fine local details. Additionally, robotic manipulation tasks of varying complexity demand different levels of modeling precision across these frequency bands. Motivated by this, we propose a novel paradigm for visuomotor policy learning that progressively models hierarchical frequency components. To further enhance precision, we introduce continuous latent representations that maintain smoothness and continuity in the action space. Extensive experiments across diverse 2D and 3D robotic manipulation benchmarks demonstrate that our approach outperforms existing methods in both accuracy and efficiency, showcasing the potential of a frequency-domain autoregressive framework with continuous tokens for generalized robotic manipulation. 

**Abstract (ZH)**: 学习有效的视觉运动策略以进行机器人操作具有挑战性，因为它要求在保持计算效率的同时生成精确的动作。现有方法由于在基本动作表示和基础网络架构方面的固有限制，仍未达到满意的效果。我们观察到，在频域中表示动作能更有效地捕捉运动的结构特征：低频分量反映全局运动模式，而高频分量编码细微的局部细节。此外，不同复杂度的机器人操作任务要求在这些频带中建模不同的精度水平。受此启发，我们提出了一种新颖的视觉运动策略学习范式，逐步建模分层的频域成分。为进一步提高精度，我们引入了连续的潜在表示，以在动作空间中保持平滑性和连续性。在多种2D和3D机器人操作基准上的广泛实验表明，我们的方法在准确性和效率上均优于现有方法，展示了频域自回归框架与连续令牌在通用机器人操作中的潜力。 

---
# Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots 

**Title (ZH)**: humanoid机器人分层意图感知表达性运动生成 

**Authors**: Lingfan Bao, Yan Pan, Tianhu Peng, Chengxu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01563)  

**Abstract**: Effective human-robot interaction requires robots to identify human intentions and generate expressive, socially appropriate motions in real-time. Existing approaches often rely on fixed motion libraries or computationally expensive generative models. We propose a hierarchical framework that combines intention-aware reasoning via in-context learning (ICL) with real-time motion generation using diffusion models. Our system introduces structured prompting with confidence scoring, fallback behaviors, and social context awareness to enable intention refinement and adaptive response. Leveraging large-scale motion datasets and efficient latent-space denoising, the framework generates diverse, physically plausible gestures suitable for dynamic humanoid interactions. Experimental validation on a physical platform demonstrates the robustness and social alignment of our method in realistic scenarios. 

**Abstract (ZH)**: 有效的人机交互要求机器人识别人类意图并在实时生成表达性、社会适当的运动。现有方法通常依赖于固定的运动库或计算成本高的生成模型。我们提出了一种分层框架，该框架结合了上下文学习（ICL）意图感知推理与基于扩散模型的实时运动生成。我们的系统引入了具有置信评分、备用行为和社会上下文意识的结构化提示，以实现意图细化和适应性响应。利用大规模运动数据集和高效的潜在空间去噪，该框架生成适用于动态类人交互的多样化且物理合理的手势。在物理平台上的实验验证表明，该方法在现实场景中的稳健性和社会一致性。 

---
# Sparse Imagination for Efficient Visual World Model Planning 

**Title (ZH)**: 稀疏想象以实现高效的视觉世界模型规划 

**Authors**: Junha Chun, Youngjoon Jeong, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01392)  

**Abstract**: World model based planning has significantly improved decision-making in complex environments by enabling agents to simulate future states and make informed choices. However, ensuring the prediction accuracy of world models often demands substantial computational resources, posing a major challenge for real-time applications. This computational burden is particularly restrictive in robotics, where resources are severely constrained. To address this limitation, we propose a Sparse Imagination for Efficient Visual World Model Planning, which enhances computational efficiency by reducing the number of tokens processed during forward prediction. Our method leverages a sparsely trained vision-based world model based on transformers with randomized grouped attention strategy, allowing the model to adaptively adjust the number of tokens processed based on the computational resource. By enabling sparse imagination (rollout), our approach significantly accelerates planning while maintaining high control fidelity. Experimental results demonstrate that sparse imagination preserves task performance while dramatically improving inference efficiency, paving the way for the deployment of world models in real-time decision-making scenarios. 

**Abstract (ZH)**: 基于 Worlds 模型的稀疏想象高效视觉世界模型规划 

---
# Generating Diverse Challenging Terrains for Legged Robots Using Quality-Diversity Algorithm 

**Title (ZH)**: 使用质量多样性算法生成多样且具有挑战性的地形用于腿式机器人 

**Authors**: Arthur Esquerre-Pourtère, Minsoo Kim, Jaeheung Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.01362)  

**Abstract**: While legged robots have achieved significant advancements in recent years, ensuring the robustness of their controllers on unstructured terrains remains challenging. It requires generating diverse and challenging unstructured terrains to test the robot and discover its vulnerabilities. This topic remains underexplored in the literature. This paper presents a Quality-Diversity framework to generate diverse and challenging terrains that uncover weaknesses in legged robot controllers. Our method, applied to both simulated bipedal and quadruped robots, produces an archive of terrains optimized to challenge the controller in different ways. Quantitative and qualitative analyses show that the generated archive effectively contains terrains that the robots struggled to traverse, presenting different failure modes. Interesting results were observed, including failure cases that were not necessarily expected. Experiments show that the generated terrains can also be used to improve RL-based controllers. 

**Abstract (ZH)**: 基于质量多样性框架生成挑战腿式机器人控制器的多样化地形 

---
# HoMeR: Learning In-the-Wild Mobile Manipulation via Hybrid Imitation and Whole-Body Control 

**Title (ZH)**: HoMeR: 结合部分示教和全身控制的野生环境下移动操作学习 

**Authors**: Priya Sundaresan, Rhea Malhotra, Phillip Miao, Jingyun Yang, Jimmy Wu, Hengyuan Hu, Rika Antonova, Francis Engelmann, Dorsa Sadigh, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01185)  

**Abstract**: We introduce HoMeR, an imitation learning framework for mobile manipulation that combines whole-body control with hybrid action modes that handle both long-range and fine-grained motion, enabling effective performance on realistic in-the-wild tasks. At its core is a fast, kinematics-based whole-body controller that maps desired end-effector poses to coordinated motion across the mobile base and arm. Within this reduced end-effector action space, HoMeR learns to switch between absolute pose predictions for long-range movement and relative pose predictions for fine-grained manipulation, offloading low-level coordination to the controller and focusing learning on task-level decisions. We deploy HoMeR on a holonomic mobile manipulator with a 7-DoF arm in a real home. We compare HoMeR to baselines without hybrid actions or whole-body control across 3 simulated and 3 real household tasks such as opening cabinets, sweeping trash, and rearranging pillows. Across tasks, HoMeR achieves an overall success rate of 79.17% using just 20 demonstrations per task, outperforming the next best baseline by 29.17 on average. HoMeR is also compatible with vision-language models and can leverage their internet-scale priors to better generalize to novel object appearances, layouts, and cluttered scenes. In summary, HoMeR moves beyond tabletop settings and demonstrates a scalable path toward sample-efficient, generalizable manipulation in everyday indoor spaces. Code, videos, and supplementary material are available at: this http URL 

**Abstract (ZH)**: HoMeR：一种结合全身控制和混合动作模式的移动 manipulation 模拟学习框架 

---
# Humanoid World Models: Open World Foundation Models for Humanoid Robotics 

**Title (ZH)**: 类人世界模型：面向类人机器人的人类世模型 

**Authors**: Muhammad Qasim Ali, Aditya Sridhar, Shahbuland Matiana, Alex Wong, Mohammad Al-Sharman  

**Link**: [PDF](https://arxiv.org/pdf/2506.01182)  

**Abstract**: Humanoid robots have the potential to perform complex tasks in human centered environments but require robust predictive models to reason about the outcomes of their actions. We introduce Humanoid World Models (HWM) a family of lightweight open source video based models that forecast future egocentric observations conditioned on actions. We train two types of generative models Masked Transformers and FlowMatching on 100 hours of humanoid demonstrations. Additionally we explore architectural variants with different attention mechanisms and parameter sharing strategies. Our parameter sharing techniques reduce model size by 33 to 53 with minimal impact on performance or visual fidelity. HWM is designed to be trained and deployed in practical academic and small lab settings such as 1 to 2 GPUs. 

**Abstract (ZH)**: humanoid 机器人在人类中心化的环境中具有执行复杂任务的潜力，但需要稳健的预测模型来推理其行为结果。我们介绍了基于动作条件预测未来第一人称观察的轻量级开源视频模型 Humanoid 世界模型（HWM）。我们在100小时的 humanoid 示范数据上训练了两种生成模型——Masked Transformers 和 FlowMatching。此外，我们还探索了具有不同注意力机制和参数共享策略的架构变体。我们的参数共享技术将模型大小减少33%至53%，同时对性能和视觉保真度的影响极小。HWM 旨在适应1到2块GPU的实用学术和小型实验室环境进行训练和部署。 

---
# Standing Tall: Robust Fall Prediction for Bipedal Robots 

**Title (ZH)**: 挺立前行：双足机器人稳健跌倒预测 

**Authors**: Gokul Prabhakaran, Jessy W. Grizzle, M. Eva Mungai  

**Link**: [PDF](https://arxiv.org/pdf/2506.01141)  

**Abstract**: This paper extends the fall prediction algorithm from Mungai et al.(2024) to a real-time/online setting, implemented in both hardware and simulation. This yields results comparable to the offline version, maintaining a zero false positive rate, sufficient lead time, and accurate lead time prediction. Additionally, it achieves a high recovery rate. The paper also evaluates the fall prediction algorithm against omnidirectional faults and introduces an improved algorithm capable of reliably predicting falls and lead times across a wider range of faults in full-sized robots. Compared to Mungai et al.(2024), the proposed algorithm performs significantly better across all metrics, such as false positive rate, lead time, accuracy, and response time, demonstrating the algorithm's efficacy for real-time fall prediction in bipedal robots. 

**Abstract (ZH)**: 这篇论文将Mungai等人（2024）的跌倒预测算法扩展到实时/在线设置，并在硬件和仿真中实现。这在保持零误报率、足够提前时间以及准确的提前时间预测的同时，产生了与离线版本相当的结果，并且实现了较高的恢复率。此外，该论文还评估了跌倒预测算法在全方位故障下的表现，并引入了一个改进的算法，能够在更大范围的故障下可靠地预测跌倒和提前时间，适用于全尺寸机器人。与Mungai等人（2024）相比，所提出的算法在误报率、提前时间、准确性和响应时间等所有指标上表现显著更好，证明了该算法在双足机器人实时跌倒预测中的有效性。 

---
# $\text{TREX}^2$: Dual-Reconstruction Framework for Teleoperated-Robot with EXtended Reality 

**Title (ZH)**: TREX²: 拓展现实下的双重建构框架用于远程操作机器人 

**Authors**: Ziliang Zhang, Cong Liu, Hyoseung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01135)  

**Abstract**: Robot teleoperation with extended reality (XR teleoperation) enables intuitive interaction by allowing remote robots to mimic user motions with real-time 3D feedback. However, existing systems face significant motion-to-motion (M2M) latency--the delay between the user's latest motion and the corresponding robot feedback--leading to high teleoperation error and mission completion time. This issue stems from the system's exclusive reliance on network communication, making it highly vulnerable to network degradation. To address these challenges, we introduce $\text{TREX}^2$, the first end-to-end, fully open-sourced XR teleoperation framework that decouples robot control and XR visualization from network dependencies. $\text{TREX}^2$ leverages local sensing data to reconstruct delayed or missing information of the counterpart, thereby significantly reducing network-induced issues. This approach allows both the XR and robot to run concurrently with network transmission while maintaining high robot planning accuracy. $\text{TREX}^2$ also features contention-aware scheduling to mitigate GPU contention and bandwidth-adaptive point cloud scaling to cope with limited bandwidth. We implement $\text{TREX}^2$ across three hardware settings, including simulated and physical robots, and evaluate it on 9,500 real-world teleoperation trials from the RoboSet dataset \cite{kumar2024robohive}, covering single- and multi-step missions. Compared to state-of-the-art XR teleoperation frameworks, $\text{TREX}^2$ reduces teleoperation error by up to 69.8\% on WLAN and 73.1\% on cellular networks with only 6.7\% maximum runtime overhead. It also improves completion time by up to 47.7\%, enabling smoother teleoperation. A real-world case study on ten stationary and mobile missions further shows $\text{TREX}^2$ achieves up to 37.7\% faster completion while lowering average teleoperation error by up to 57.2\%. 

**Abstract (ZH)**: XR增强现实遥操作框架TREX²：面向端到端开源的减少网络依赖的遥操作技术 

---
# iRonCub 3: The Jet-Powered Flying Humanoid Robot 

**Title (ZH)**: iRonCub 3：喷气动力飞行类人机器人 

**Authors**: Davide Gorbani, Hosameldin Awadalla Omer Mohamed, Giuseppe L'Erario, Gabriele Nava, Punith Reddy Vanteddu, Shabarish Purushothaman Pillai, Antonello Paolino, Fabio Bergonti, Saverio Taliani, Alessandro Croci, Nicholas James Tremaroli, Silvio Traversaro, Bruno Vittorio Trombetta, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2506.01125)  

**Abstract**: This article presents iRonCub 3, a jet-powered humanoid robot, and its first flight experiments. Unlike traditional aerial vehicles, iRonCub 3 aims to achieve flight using a full-body humanoid form, which poses unique challenges in control, estimation, and system integration. We highlight the robot's current mechanical and software architecture, including its propulsion system, control framework, and experimental infrastructure. The control and estimation framework is first validated in simulation by performing a takeoff and tracking a reference trajectory. Then, we demonstrate, for the first time, a liftoff of a jet-powered humanoid robot - an initial but significant step toward aerial humanoid mobility. Also, we detail how the experimental area around a jet-powered humanoid robot should be designed in order to deal with a level of complexity that is substantially superior than indoor humanoid robot experiments. 

**Abstract (ZH)**: 本文介绍了iRonCub 3，一种喷气动力人形机器人及其首次飞行实验。不同于传统的航空器，iRonCub 3旨在通过全身人形形式实现飞行，这在控制、估计和系统集成方面提出了独特的挑战。我们强调了该机器人的当前机械和软件架构，包括其推进系统、控制框架和实验基础设施。控制和估计框架首先在仿真中通过执行起飞和跟踪参考轨迹进行了验证。然后，我们首次展示了喷气动力人形机器人离地飞行的过程——这是向空中人形机器人移动迈出的重要一步。此外，我们还详细介绍了如何设计围绕喷气动力人形机器人的实验区域，以应对远超室内外人形机器人实验的复杂性。 

---
# STATE-NAV: Stability-Aware Traversability Estimation for Bipedal Navigation on Rough Terrain 

**Title (ZH)**: STATE-NAV: 考虑稳定性的粗糙地形 bipedal 导航可通行性估计 

**Authors**: Ziwon Yoon, Lawrence Y. Zhu, Lu Gan, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01046)  

**Abstract**: Bipedal robots have advantages in maneuvering human-centered environments, but face greater failure risk compared to other stable mobile plarforms such as wheeled or quadrupedal robots. While learning-based traversability has been widely studied for these platforms, bipedal traversability has instead relied on manually designed rules with limited consideration of locomotion stability on rough terrain. In this work, we present the first learning-based traversability estimation and risk-sensitive navigation framework for bipedal robots operating in diverse, uneven environments. 

**Abstract (ZH)**: 基于学习的双足机器人在多样化不平坦环境中的通过性估测与风险敏感导航框架 

---
# Robust and Safe Multi-Agent Reinforcement Learning Framework with Communication for Autonomous Vehicles 

**Title (ZH)**: 具备通信的鲁棒且安全的多智能体强化学习框架：自主车辆领域 

**Authors**: Keshawn Smith, Zhili Zhang, H M Sabbir Ahmad, Ehsan Sabouni, Maniak Mondal, Song Han, Wenchao Li, Fei Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00982)  

**Abstract**: Deep multi-agent reinforcement learning (MARL) has been demonstrated effectively in simulations for many multi-robot problems. For autonomous vehicles, the development of vehicle-to-vehicle (V2V) communication technologies provide opportunities to further enhance safety of the system. However, zero-shot transfer of simulator-trained MARL policies to hardware dynamic systems remains challenging, and how to leverage communication and shared information for MARL has limited demonstrations on hardware. This problem is challenged by discrepancies between simulated and physical states, system state and model uncertainties, practical shared information design, and the need for safety guarantees in both simulation and hardware. This paper introduces RSR-RSMARL, a novel Robust and Safe MARL framework that supports Real-Sim-Real (RSR) policy adaptation for multi-agent systems with communication among agents, with both simulation and hardware demonstrations. RSR-RSMARL leverages state (includes shared state information among agents) and action representations considering real system complexities for MARL formulation. The MARL policy is trained with robust MARL algorithm to enable zero-shot transfer to hardware considering the sim-to-real gap. A safety shield module using Control Barrier Functions (CBFs) provides safety guarantee for each individual agent. Experiment results on F1/10th-scale autonomous vehicles with V2V communication demonstrate the ability of RSR-RSMARL framework to enhance driving safety and coordination across multiple configurations. These findings emphasize the importance of jointly designing robust policy representations and modular safety architectures to enable scalable, generalizable RSR transfer in multi-agent autonomy. 

**Abstract (ZH)**: 一种适用于多agent系统的Robust和Safe MARL框架：基于通信的Real-Sim-Real策略适应 

---
# DriveMind: A Dual-VLM based Reinforcement Learning Framework for Autonomous Driving 

**Title (ZH)**: DriveMind: 一种基于双多模态预训练语言模型的自主驾驶强化学习框架 

**Authors**: Dawood Wasif, Terrence J Moore, Chandan K Reddy, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2506.00819)  

**Abstract**: End-to-end autonomous driving systems map sensor data directly to control commands, but remain opaque, lack interpretability, and offer no formal safety guarantees. While recent vision-language-guided reinforcement learning (RL) methods introduce semantic feedback, they often rely on static prompts and fixed objectives, limiting adaptability to dynamic driving scenes. We present DriveMind, a unified semantic reward framework that integrates: (i) a contrastive Vision-Language Model (VLM) encoder for stepwise semantic anchoring; (ii) a novelty-triggered VLM encoder-decoder, fine-tuned via chain-of-thought (CoT) distillation, for dynamic prompt generation upon semantic drift; (iii) a hierarchical safety module enforcing kinematic constraints (e.g., speed, lane centering, stability); and (iv) a compact predictive world model to reward alignment with anticipated ideal states. DriveMind achieves 19.4 +/- 2.3 km/h average speed, 0.98 +/- 0.03 route completion, and near-zero collisions in CARLA Town 2, outperforming baselines by over 4% in success rate. Its semantic reward generalizes zero-shot to real dash-cam data with minimal distributional shift, demonstrating robust cross-domain alignment and potential for real-world deployment. 

**Abstract (ZH)**: 统一语义奖励框架DriveMind：面向动态驾驶场景的端到端自主驾驶系统 

---
# AWML: An Open-Source ML-based Robotics Perception Framework to Deploy for ROS-based Autonomous Driving Software 

**Title (ZH)**: AWML: 一种基于ROS的自主驾驶软件的开源机器学习导向的机器人感知框架 

**Authors**: Satoshi Tanaka, Samrat Thapa, Kok Seang Tan, Amadeusz Szymko, Lobos Kenzo, Koji Minoda, Shintaro Tomie, Kotaro Uetake, Guolong Zhang, Isamu Yamashita, Takamasa Horibe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00645)  

**Abstract**: In recent years, machine learning technologies have played an important role in robotics, particularly in the development of autonomous robots and self-driving vehicles. As the industry matures, robotics frameworks like ROS 2 have been developed and provides a broad range of applications from research to production. In this work, we introduce AWML, a framework designed to support MLOps for robotics. AWML provides a machine learning infrastructure for autonomous driving, supporting not only the deployment of trained models to robotic systems, but also an active learning pipeline that incorporates auto-labeling, semi-auto-labeling, and data mining techniques. 

**Abstract (ZH)**: 近年来，机器学习技术在机器人领域发挥了重要作用，特别是在自主机器人和自动驾驶车辆的发展中。随着行业的成熟，像ROS 2这样的机器人框架被开发出来，并提供了从研究到生产的广泛应用。在本工作中，我们介绍了AWML，这是一个用于机器人领域的MLOps框架。AWML提供了一种机器学习基础设施，不仅支持将训练好的模型部署到机器人系统中，还提供了一个集成自动标注、半自动标注和数据挖掘技术的主动学习管道。 

---
# Evaluating Robot Policies in a World Model 

**Title (ZH)**: 在世界模型中评估机器人策略 

**Authors**: Julian Quevedo, Percy Liang, Sherry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00613)  

**Abstract**: Robotics has broad applications from automating house chores to taking care of patients. However, evaluating robot control policies is challenging, as real-world testing is expensive, while handcrafted simulations often fail to accurately reflect real-world conditions, resulting in poor correlation between simulated evaluation and real-world outcomes. In this work, we investigate World-model-based Policy Evaluation (WPE). We first train an action-conditioned video generation model as a proxy to real-world environments. To enable efficient rollouts of hundreds of interactive steps while mitigating error accumulation in the world model, we propose an inference scheme which we call Blockwise-Autoregressive Diffusion Transformer with adjustable context and decoding horizon lengths. To ensure that the world model indeed follows action input, we propose metrics based on the agreement between the ground truth video and generated video conditioned on the same sequence of actions to evaluate the world model. We then use the world model for policy evaluation by performing Monte Carlo rollouts in the world model while employing a vision-language model (VLM) as a reward function. Interestingly, we found that WPE tends to underestimate the policy values for in-distribution actions and overestimate policy values for out-of-distribution actions. Nevertheless, WPE preserves the relative rankings of different policies. In emulating real robot executions, WPE achieves high fidelity in mimicing robot arm movements as in real videos, while emulating highly realistic object interaction remains challenging. Despite this limitation, we show that a world model can serve as a starting point for evaluating robot policies before real-world deployment. 

**Abstract (ZH)**: 基于世界模型的策略评估：机器人控制策略评估的新方法 

---
# Constrained Stein Variational Gradient Descent for Robot Perception, Planning, and Identification 

**Title (ZH)**: 受约束的Stein变分梯度下降方法在机器人感知、规划和识别中的应用 

**Authors**: Griffin Tabor, Tucker Hermans  

**Link**: [PDF](https://arxiv.org/pdf/2506.00589)  

**Abstract**: Many core problems in robotics can be framed as constrained optimization problems. Often on these problems, the robotic system has uncertainty, or it would be advantageous to identify multiple high quality feasible solutions. To enable this, we present two novel frameworks for applying principles of constrained optimization to the new variational inference algorithm Stein variational gradient descent. Our general framework supports multiple types of constrained optimizers and can handle arbitrary constraints. We demonstrate on a variety of problems that we are able to learn to approximate distributions without violating constraints. Specifically, we show that we can build distributions of: robot motion plans that exactly avoid collisions, robot arm joint angles on the SE(3) manifold with exact table placement constraints, and object poses from point clouds with table placement constraints. 

**Abstract (ZH)**: 许多机器人领域的核心问题可以框架为约束优化问题。在这种问题上，机器人系统通常存在不确定性，或者识别多个高质量的可行解是很有优势的。为了实现这一点，我们提出了将约束优化原理应用于新的变分推理算法Stein variational gradient descent的两种新型框架。我们的通用框架支持多种类型的约束优化器，并能处理任意类型的约束。我们通过多种问题的演示表明，能够学习到不违反约束的近似分布。具体来说，我们展示了如何构建满足以下条件的分布：机器人运动计划完全避免碰撞、SE(3)流形上的机器人手臂关节角度带精确桌面放置约束、以及带有桌面放置约束的点云物体姿态分布。 

---
# Using Diffusion Ensembles to Estimate Uncertainty for End-to-End Autonomous Driving 

**Title (ZH)**: 使用扩散集成估计端到端自动驾驶中的不确定性 

**Authors**: Florian Wintel, Sigmund H. Høeg, Gabriel Kiss, Frank Lindseth  

**Link**: [PDF](https://arxiv.org/pdf/2506.00560)  

**Abstract**: End-to-end planning systems for autonomous driving are improving rapidly, especially in closed-loop simulation environments like CARLA. Many such driving systems either do not consider uncertainty as part of the plan itself, or obtain it by using specialized representations that do not generalize. In this paper, we propose EnDfuser, an end-to-end driving system that uses a diffusion model as the trajectory planner. EnDfuser effectively leverages complex perception information like fused camera and LiDAR features, through combining attention pooling and trajectory planning into a single diffusion transformer module. Instead of committing to a single plan, EnDfuser produces a distribution of candidate trajectories (128 for our case) from a single perception frame through ensemble diffusion. By observing the full set of candidate trajectories, EnDfuser provides interpretability for uncertain, multi-modal future trajectory spaces, where there are multiple plausible options. EnDfuser achieves a competitive driving score of 70.1 on the Longest6 benchmark in CARLA with minimal concessions on inference speed. Our findings suggest that ensemble diffusion, used as a drop-in replacement for traditional point-estimate trajectory planning modules, can help improve the safety of driving decisions by modeling the uncertainty of the posterior trajectory distribution. 

**Abstract (ZH)**: 端到端融合系统在自主驾驶中的快速进步，尤其是在CARLA等闭环模拟环境中。许多此类驾驶系统要么不将不确定性作为计划的一部分，要么使用特殊表示法而不具备泛化能力。本文提出了一种名为EnDfuser的端到端驾驶系统，该系统使用扩散模型作为轨迹规划器。EnDfuser有效利用了融合的感知信息，如融合的相机和LiDAR特征，通过将注意力池化和轨迹规划结合到一个单一的扩散变换器模块中来实现。EnDfuser不是仅生成一个确定性计划，而是通过集合扩散从单个感知帧生成候选轨迹分布（例如，128条候选轨迹）。通过观察候选轨迹的完整集合，EnDfuser提供了对具有多个可能选项的不确定性和多模态未来轨迹空间的可解释性。EnDfuser在CARLA的Longest6基准测试中取得了70.1的竞争力驾驶分数，同时在推理速度上几乎没有妥协。我们的研究发现表明，将集合扩散用作传统点估计轨迹规划模块的即插即用替代品，可以有助于通过建模后轨迹分布的不确定性来提高驾驶决策的安全性。 

---
# Disturbance-Aware Adaptive Compensation in Hybrid Force-Position Locomotion Policy for Legged Robots 

**Title (ZH)**: 带扰动感知自适应补偿的腿式机器人混合力位运动政策 

**Authors**: Yang Zhang, Buqing Nie, Zhanxiang Cao, Yangqing Fu, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00472)  

**Abstract**: Reinforcement Learning (RL)-based methods have significantly improved the locomotion performance of legged robots. However, these motion policies face significant challenges when deployed in the real world. Robots operating in uncertain environments struggle to adapt to payload variations and external disturbances, resulting in severe degradation of motion performance. In this work, we propose a novel Hybrid Force-Position Locomotion Policy (HFPLP) learning framework, where the action space of the policy is defined as a combination of target joint positions and feedforward torques, enabling the robot to rapidly respond to payload variations and external disturbances. In addition, the proposed Disturbance-Aware Adaptive Compensation (DAAC) provides compensation actions in the torque space based on external disturbance estimation, enhancing the robot's adaptability to dynamic environmental changes. We validate our approach in both simulation and real-world deployment, demonstrating that it outperforms existing methods in carrying payloads and resisting disturbances. 

**Abstract (ZH)**: 基于强化学习的混合力-位置运动策略框架在提升腿式机器人运动性能方面的研究：扰动感知自适应补偿方法 

---
# Diffusion Models for Increasing Accuracy in Olfaction Sensors and Datasets 

**Title (ZH)**: 扩散模型在提高气味传感器和数据集准确性中的应用 

**Authors**: Kordel K. France, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00455)  

**Abstract**: Robotic odour source localization (OSL) is a critical capability for autonomous systems operating in complex environments. However, current OSL methods often suffer from ambiguities, particularly when robots misattribute odours to incorrect objects due to limitations in olfactory datasets and sensor resolutions. To address this challenge, we introduce a novel machine learning method using diffusion-based molecular generation to enhance odour localization accuracy that can be used by itself or with automated olfactory dataset construction pipelines with vision-language models (VLMs) This generative process of our diffusion model expands the chemical space beyond the limitations of both current olfactory datasets and the training data of VLMs, enabling the identification of potential odourant molecules not previously documented. The generated molecules can then be more accurately validated using advanced olfactory sensors which emulate human olfactory recognition through electronic sensor arrays. By integrating visual analysis, language processing, and molecular generation, our framework enhances the ability of olfaction-vision models on robots to accurately associate odours with their correct sources, thereby improving navigation and decision-making in environments where olfactory cues are essential. Our methodology represents a foundational advancement in the field of robotic olfaction, offering a scalable solution to the challenges posed by limited olfactory data and sensor ambiguities. 

**Abstract (ZH)**: 基于扩散过程分子生成的机器人气味来源定位方法 

---
# LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks 

**Title (ZH)**: LoHoVLA：统一的视觉-语言-行动模型用于长期 horizon 汰务 

**Authors**: Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00411)  

**Abstract**: Real-world embodied agents face long-horizon tasks, characterized by high-level goals demanding multi-step solutions beyond single actions. Successfully navigating these requires both high-level task planning (i.e., decomposing goals into sub-tasks) and low-level motion control (i.e., generating precise robot actions). While existing vision language action (VLA) models and hierarchical architectures offer potential in embodied tasks, the former often falter in planning, and the latter can suffer from coordination issues, both hampering performance. We introduce a new unified VLA framework for long-horizon tasks, dubbed LoHoVLA, to overcome these limitations. LoHoVLA leverages a large pretrained vision language model (VLM) as the backbone to jointly generate language and action tokens for sub-task generation and robot action prediction, respectively. This shared representation promotes better generalization across tasks. Additionally, LoHoVLA embraces a hierarchical closed-loop control mechanism to mitigate errors originating from both high-level planning and low-level control. To train LoHoVLA, we introduce LoHoSet, a dataset built on the Ravens simulator, containing 20 long-horizon tasks, each with 1,000 expert demonstrations composed of visual observations, linguistic goals, sub-tasks, and robot actions. Experimental results show that LoHoVLA significantly surpasses both hierarchical and standard VLA approaches on long-horizon embodied tasks in the Ravens simulator. These findings underscore the promise of unified architectures for advancing generalizable embodied intelligence. 

**Abstract (ZH)**: LoHoVLA：用于长时_horizon任务的统一视觉语言行动框架 

---
# Music-driven Robot Swarm Painting 

**Title (ZH)**: 音乐驱动的机器人 swarm 绘画 

**Authors**: Jingde Cheng, Gennaro Notomista  

**Link**: [PDF](https://arxiv.org/pdf/2506.00326)  

**Abstract**: This paper proposes a novel control framework for robotic swarms capable of turning a musical input into a painting. The approach connects the two artistic domains, music and painting, leveraging their respective connections to fundamental emotions. The robotic units of the swarm are controlled in a coordinated fashion using a heterogeneous coverage policy to control the motion of the robots which continuously release traces of color in the environment. The results of extensive simulations performed starting from different musical inputs and with different color equipments are reported. Finally, the proposed framework has been implemented on real robots equipped with LED lights and capable of light-painting. 

**Abstract (ZH)**: 本文提出了一种新颖的控制框架，能够将 musical 输入转化为绘画作品，该框架将音乐和绘画这两种艺术领域连接起来，利用它们与基本情感的关联。集群中的机器人单元以协调的方式受控，并使用异构覆盖策略控制机器人的运动，使其在环境中不断释放颜色痕迹。从不同的音乐输入和不同颜色设备出发进行的大量仿真结果被报道。最后，提出的框架已在配备 LED 灯光并能够进行光绘的实体机器人上实现。 

---
# Learning Aerodynamics for the Control of Flying Humanoid Robots 

**Title (ZH)**: 学习气动特性以控制飞行类人机器人 

**Authors**: Antonello Paolino, Gabriele Nava, Fabio Di Natale, Fabio Bergonti, Punith Reddy Vanteddu, Donato Grassi, Luca Riccobene, Alex Zanotti, Renato Tognaccini, Gianluca Iaccarino, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2506.00305)  

**Abstract**: Robots with multi-modal locomotion are an active research field due to their versatility in diverse environments. In this context, additional actuation can provide humanoid robots with aerial capabilities. Flying humanoid robots face challenges in modeling and control, particularly with aerodynamic forces. This paper addresses these challenges from a technological and scientific standpoint. The technological contribution includes the mechanical design of iRonCub-Mk1, a jet-powered humanoid robot, optimized for jet engine integration, and hardware modifications for wind tunnel experiments on humanoid robots for precise aerodynamic forces and surface pressure measurements. The scientific contribution offers a comprehensive approach to model and control aerodynamic forces using classical and learning techniques. Computational Fluid Dynamics (CFD) simulations calculate aerodynamic forces, validated through wind tunnel experiments on iRonCub-Mk1. An automated CFD framework expands the aerodynamic dataset, enabling the training of a Deep Neural Network and a linear regression model. These models are integrated into a simulator for designing aerodynamic-aware controllers, validated through flight simulations and balancing experiments on the iRonCub-Mk1 physical prototype. 

**Abstract (ZH)**: 具有多模态运动的机器人是活跃的研究领域，由于其在多种环境中的灵活性。在这种背景下，额外的动力装置可以为类人机器人提供飞行能力。飞行类人机器人在建模和控制方面面临挑战，尤其是在气动力方面。本文从技术和科学的角度解决了这些挑战。技术贡献包括iRonCub-Mk1机械设计，这是一种配备喷气发动机的类人机器人，优化了喷气发动机的集成，并进行了硬件修改以在类人机器人上进行风洞实验，以精确测量气动力和表面压力。科学贡献提供了一种全面的气动力建模和控制方法，结合了经典技术和学习技术。通过计算流体动力学（CFD）模拟计算气动力，并通过iRonCub-Mk1的风洞实验进行验证。自动化的CFD框架扩展了气动力数据集，使深度神经网络和线性回归模型的训练成为可能。这些模型被集成到一个模拟器中，用于设计气动力感知控制器，并通过飞行模拟和iRonCub-Mk1物理原型的平衡实验进行了验证。 

---
# RoboMoRe: LLM-based Robot Co-design via Joint Optimization of Morphology and Reward 

**Title (ZH)**: RoboMoRe：基于LLM的机器人联合形态与奖励优化协同设计 

**Authors**: Jiawei Fang, Yuxuan Sun, Chengtian Ma, Qiuyu Lu, Lining Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00276)  

**Abstract**: Robot co-design, jointly optimizing morphology and control policy, remains a longstanding challenge in the robotics community, where many promising robots have been developed. However, a key limitation lies in its tendency to converge to sub-optimal designs due to the use of fixed reward functions, which fail to explore the diverse motion modes suitable for different morphologies. Here we propose RoboMoRe, a large language model (LLM)-driven framework that integrates morphology and reward shaping for co-optimization within the robot co-design loop. RoboMoRe performs a dual-stage optimization: in the coarse optimization stage, an LLM-based diversity reflection mechanism generates both diverse and high-quality morphology-reward pairs and efficiently explores their distribution. In the fine optimization stage, top candidates are iteratively refined through alternating LLM-guided reward and morphology gradient updates. RoboMoRe can optimize both efficient robot morphologies and their suited motion behaviors through reward shaping. Results demonstrate that without any task-specific prompting or predefined reward/morphology templates, RoboMoRe significantly outperforms human-engineered designs and competing methods across eight different tasks. 

**Abstract (ZH)**: 机器人协同设计中的形态与控制策略联合优化仍是一个长期挑战，尽管已经开发了许多有前景的机器人，但其固定奖励函数的使用导致了亚最优设计的收敛，未能探索适合不同形态的多样化运动模式。为此，我们提出了一种以大型语言模型驱动的框架RoboMoRe，该框架将形态与奖励塑造结合起来，在机器人协同设计循环中共同优化。RoboMoRe执行两阶段优化：在粗优化阶段，基于大型语言模型的多样性反射机制生成多样且高质量的形态-奖励配对，并高效探索它们的分布；在细优化阶段，通过交替的基于大型语言模型的奖励和形态梯度更新，逐步优化顶级候选方案。RoboMoRe能够通过奖励塑造优化高效的机器人形态及其相应的运动行为。结果表明，无需任何特定任务的提示或预定义的奖励/形态模板，RoboMoRe在八个不同任务上显著优于人工设计和竞争方法。 

---
# Understanding while Exploring: Semantics-driven Active Mapping 

**Title (ZH)**: 探索中的理解：语义驱动的主动建图 

**Authors**: Liyan Chen, Huangying Zhan, Hairong Yin, Yi Xu, Philippos Mordohai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00225)  

**Abstract**: Effective robotic autonomy in unknown environments demands proactive exploration and precise understanding of both geometry and semantics. In this paper, we propose ActiveSGM, an active semantic mapping framework designed to predict the informativeness of potential observations before execution. Built upon a 3D Gaussian Splatting (3DGS) mapping backbone, our approach employs semantic and geometric uncertainty quantification, coupled with a sparse semantic representation, to guide exploration. By enabling robots to strategically select the most beneficial viewpoints, ActiveSGM efficiently enhances mapping completeness, accuracy, and robustness to noisy semantic data, ultimately supporting more adaptive scene exploration. Our experiments on the Replica and Matterport3D datasets highlight the effectiveness of ActiveSGM in active semantic mapping tasks. 

**Abstract (ZH)**: 有效的自主机器人在未知环境中的探索需要主动探索和对几何和语义的精确理解。本文提出了一种名为ActiveSGM的主动语义映射框架，该框架能够在执行前预测潜在观察的信息量。基于3D高斯点云（3DGS）映射骨干网络，该方法结合语义和几何不确定性量化以及稀疏语义表示来引导探索。通过使机器人能够战略性地选择最具益处的视角，ActiveSGM有效地增强了映射的完整性和准确性，并提高了对噪声语义数据的鲁棒性，最终支持更适应的场景探索。我们在Replica和Matterport3D数据集上的实验展示了ActiveSGM在主动语义映射任务中的有效性。 

---
# Interactive Imitation Learning for Dexterous Robotic Manipulation: Challenges and Perspectives -- A Survey 

**Title (ZH)**: Dexterous 机器人操作的交互式模仿学习：挑战与展望——综述 

**Authors**: Edgar Welte, Rania Rayyes  

**Link**: [PDF](https://arxiv.org/pdf/2506.00098)  

**Abstract**: Dexterous manipulation is a crucial yet highly complex challenge in humanoid robotics, demanding precise, adaptable, and sample-efficient learning methods. As humanoid robots are usually designed to operate in human-centric environments and interact with everyday objects, mastering dexterous manipulation is critical for real-world deployment. Traditional approaches, such as reinforcement learning and imitation learning, have made significant strides, but they often struggle due to the unique challenges of real-world dexterous manipulation, including high-dimensional control, limited training data, and covariate shift. This survey provides a comprehensive overview of these challenges and reviews existing learning-based methods for dexterous manipulation, spanning imitation learning, reinforcement learning, and hybrid approaches. A promising yet underexplored direction is interactive imitation learning, where human feedback actively refines a robot's behavior during training. While interactive imitation learning has shown success in various robotic tasks, its application to dexterous manipulation remains limited. To address this gap, we examine current interactive imitation learning techniques applied to other robotic tasks and discuss how these methods can be adapted to enhance dexterous manipulation. By synthesizing state-of-the-art research, this paper highlights key challenges, identifies gaps in current methodologies, and outlines potential directions for leveraging interactive imitation learning to improve dexterous robotic skills. 

**Abstract (ZH)**: 灵巧 manipulation 是类人机器人研究中一个关键但极具挑战性的课题，要求精确、适应性强且样本高效的算法。由于类人机器人通常设计用于人类中心环境并操作日常物体，因此掌握灵巧 manipulation 对于实际部署至关重要。传统方法，如强化学习和模仿学习，虽取得了显著进展，但在现实世界的灵巧 manipulation 中面临的高维控制、有限训练数据和协变量偏移等挑战常常难以克服。本文综述了这些挑战，并回顾了现有的基于学习的灵巧 manipulation 方法，涵盖模仿学习、强化学习及其混合方法。一个有前景但尚未充分探索的方向是互动模仿学习，其中人类反馈在训练过程中主动优化机器人的行为。虽然互动模仿学习在各种机器人任务中取得了成功，但在灵巧 manipulation 中的应用仍有限。为解决这一差距，本文讨论了当前应用于其他机器人任务的互动模仿学习技术，并探讨了如何调整这些方法以增强灵巧 manipulation。通过综合最新的研究成果，本文指出了关键挑战，识别了当前方法中的不足，并概述了利用互动模仿学习提高机器人灵巧技能的潜在方向。 

---
# Navigation of a Three-Link Microswimmer via Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的三连杆微型游泳器导航 

**Authors**: Yuyang Lai, Sina Heydari, On Shun Pak, Yi Man  

**Link**: [PDF](https://arxiv.org/pdf/2506.00084)  

**Abstract**: Motile microorganisms develop effective swimming gaits to adapt to complex biological environments. Translating this adaptability to smart microrobots presents significant challenges in motion planning and stroke design. In this work, we explore the use of reinforcement learning (RL) to develop stroke patterns for targeted navigation in a three-link swimmer model at low Reynolds numbers. Specifically, we design two RL-based strategies: one focusing on maximizing velocity (Velocity-Focused Strategy) and another balancing velocity with energy consumption (Energy-Aware Strategy). Our results demonstrate how the use of different reward functions influences the resulting stroke patterns developed via RL, which are compared with those obtained from traditional optimization methods. Furthermore, we showcase the capability of the RL-powered swimmer in adapting its stroke patterns in performing different navigation tasks, including tracing complex trajectories and pursuing moving targets. Taken together, this work highlights the potential of reinforcement learning as a versatile tool for designing efficient and adaptive microswimmers capable of sophisticated maneuvers in complex environments. 

**Abstract (ZH)**: 可游动微生物通过发展有效的游泳姿态来适应复杂的生物环境。将这种适应性移植到智能微机器人中在运动规划和摆动设计方面提出了重大挑战。在本工作中，我们探索使用强化学习（RL）来为低雷诺数下的三链接游泳者模型开发定向导航的摆动模式。具体地，我们设计了两种基于RL的策略：一种专注于最大化速度（速度导向策略）和另一种在速度与能耗之间寻求平衡（能量感知策略）。我们的结果表明，不同的奖励函数如何影响通过RL产生的摆动模式，并将这些结果与传统优化方法获得的结果进行比较。此外，我们展示了基于RL的游泳者在执行不同导航任务（包括跟踪复杂轨迹和追逐移动目标）时调整其摆动模式的能力。总体而言，本工作强调了强化学习作为设计高效且适应性强的微游泳者工具的潜力，这些微游泳者能够在复杂环境中执行复杂的操控动作。 

---
# Hi-Dyna Graph: Hierarchical Dynamic Scene Graph for Robotic Autonomy in Human-Centric Environments 

**Title (ZH)**: Hi-Dyna 图：以人为本环境中机器人自主性的分层动态场景图 

**Authors**: Jiawei Hou, Xiangyang Xue, Taiping Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00083)  

**Abstract**: Autonomous operation of service robotics in human-centric scenes remains challenging due to the need for understanding of changing environments and context-aware decision-making. While existing approaches like topological maps offer efficient spatial priors, they fail to model transient object relationships, whereas dense neural representations (e.g., NeRF) incur prohibitive computational costs. Inspired by the hierarchical scene representation and video scene graph generation works, we propose Hi-Dyna Graph, a hierarchical dynamic scene graph architecture that integrates persistent global layouts with localized dynamic semantics for embodied robotic autonomy. Our framework constructs a global topological graph from posed RGB-D inputs, encoding room-scale connectivity and large static objects (e.g., furniture), while environmental and egocentric cameras populate dynamic subgraphs with object position relations and human-object interaction patterns. A hybrid architecture is conducted by anchoring these subgraphs to the global topology using semantic and spatial constraints, enabling seamless updates as the environment evolves. An agent powered by large language models (LLMs) is employed to interpret the unified graph, infer latent task triggers, and generate executable instructions grounded in robotic affordances. We conduct complex experiments to demonstrate Hi-Dyna Grap's superior scene representation effectiveness. Real-world deployments validate the system's practicality with a mobile manipulator: robotics autonomously complete complex tasks with no further training or complex rewarding in a dynamic scene as cafeteria assistant. See this https URL for video demonstration and more details. 

**Abstract (ZH)**: 基于人类中心场景的服务机器人自主操作仍具有挑战性，因为需要理解和进行情境aware决策。现有的方法如拓扑地图虽然提供了有效的空间先验知识，但无法建模瞬时对象关系，而密集神经表示（如NeRF）则会产生高昂的计算成本。受分层场景表示和视频场景图生成工作的启发，我们提出了一种分层动态场景图架构Hi-Dyna Graph，该架构结合了持久的全局布局和局部动态语义，以实现嵌入式机器人的自主性。我们的框架从摆拍的RGB-D输入构建全局拓扑图，编码房间尺度的连接性和大型静态对象（如家具），同时环境和第一人称相机填充动态子图，包含对象位置关系和人机交互模式。通过结合语义和空间约束将这些子图锚定到全局拓扑结构中，实现环境演变时的无缝更新。由大规模语言模型驱动的代理用于解释统一的图，推断潜在任务触发，并生成基于机器人操作能力的可执行指令。通过复杂实验展示了Hi-Dyna Graph在场景表示有效性上的优越性。实际部署验证了系统的实用性，使用移动 manipulator：机器人在动态场景中作为自助餐厅助手自主完成复杂任务，无需进一步训练或复杂奖励。更多详情请参见此链接。 

---
# Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics 

**Title (ZH)**: Robot-R1：强化学习在增强机器人 embodied reasoning 中的应用 

**Authors**: Dongyoung Kim, Sumin Park, Huiwon Jang, Jinwoo Shin, Jaehyung Kim, Younggyo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00070)  

**Abstract**: Large Vision-Language Models (LVLMs) have recently shown great promise in advancing robotics by combining embodied reasoning with robot control. A common approach involves training on embodied reasoning tasks related to robot control using Supervised Fine-Tuning (SFT). However, SFT datasets are often heuristically constructed and not explicitly optimized for improving robot control. Furthermore, SFT often leads to issues such as catastrophic forgetting and reduced generalization performance. To address these limitations, we introduce Robot-R1, a novel framework that leverages reinforcement learning to enhance embodied reasoning specifically for robot control. Robot-R1 learns to predict the next keypoint state required for task completion, conditioned on the current scene image and environment metadata derived from expert demonstrations. Inspired by the DeepSeek-R1 learning approach, Robot-R1 samples reasoning-based responses and reinforces those that lead to more accurate predictions. Our experiments show that models trained with Robot-R1 outperform SFT methods on embodied reasoning tasks. Despite having only 7B parameters, Robot-R1 even surpasses GPT-4o on reasoning tasks related to low-level action control, such as spatial and primitive movement reasoning. 

**Abstract (ZH)**: 大型多模态模型（LVLMs）通过结合本体推理与机器人控制，在推动机器人技术方面展现出了巨大的潜力。一种常见方法是使用监督微调（SFT）在与机器人控制相关的本体推理任务上进行训练。然而，SFT数据集通常是基于启发式构建的，并未明确优化以提高机器人控制性能。此外，SFT还常常导致灾难性遗忘和泛化性能降低等问题。为解决这些问题，我们提出了Robot-R1新型框架，该框架利用强化学习来增强特定于机器人控制的本体推理。Robot-R1通过当前场景图像和从专家示范中派生的环境元数据条件，学习预测完成任务所需的下一个关键点状态。受DeepSeek-R1学习方法的启发，Robot-R1采样基于推理的响应，并强化那些能够产生更准确预测的回答。我们的实验结果表明，使用Robot-R1训练的模型在本体推理任务上的表现优于SFT方法。即使只有7B参数，Robot-R1在低层级动作控制相关的推理任务（如空间和基本运动推理）上也超越了GPT-4o。 

---
# From Motion to Behavior: Hierarchical Modeling of Humanoid Generative Behavior Control 

**Title (ZH)**: 从运动到行为：类人生成行为控制的层次模型 

**Authors**: Jusheng Zhang, Jinzhou Tang, Sidi Liu, Mingyan Li, Sheng Zhang, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00043)  

**Abstract**: Human motion generative modeling or synthesis aims to characterize complicated human motions of daily activities in diverse real-world environments. However, current research predominantly focuses on either low-level, short-period motions or high-level action planning, without taking into account the hierarchical goal-oriented nature of human activities. In this work, we take a step forward from human motion generation to human behavior modeling, which is inspired by cognitive science. We present a unified framework, dubbed Generative Behavior Control (GBC), to model diverse human motions driven by various high-level intentions by aligning motions with hierarchical behavior plans generated by large language models (LLMs). Our insight is that human motions can be jointly controlled by task and motion planning in robotics, but guided by LLMs to achieve improved motion diversity and physical fidelity. Meanwhile, to overcome the limitations of existing benchmarks, i.e., lack of behavioral plans, we propose GBC-100K dataset annotated with a hierarchical granularity of semantic and motion plans driven by target goals. Our experiments demonstrate that GBC can generate more diverse and purposeful high-quality human motions with 10* longer horizons compared with existing methods when trained on GBC-100K, laying a foundation for future research on behavioral modeling of human motions. Our dataset and source code will be made publicly available. 

**Abstract (ZH)**: 基于生成的行为控制 modeling human motions via large language models and hierarchical behavior planning 

---
# Buoyant Choreographies: Harmonies of Light, Sound, and Human Connection 

**Title (ZH)**: 浮力 choreography：光、音与人联结的和谐 

**Authors**: Dennis Hong, Yusuke Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2506.00021)  

**Abstract**: BALLU, the Buoyancy Assisted Lightweight Legged Unit, is a unique legged robot with a helium balloon body and articulated legs \fig{fig:fig1}. Since it is buoyant-assisted, BALLU is inherently stable, never falling over, while being able to walk, jump, and interact safely with people. The BALLU art installation builds on this playful platform to express fluidity, serendipity, and connection. It transforms robotic motion into an artistic visual and acoustic experience, merging technology and creativity into a dynamic, interactive display. This exhibition intentionally does not have a physical boundary for the robots, emphasizing the harmony of the technologies and humanity. This work significantly extends BALLU's existing permanent exhibition in the Seoul Robotics & Artificial Intelligence Museum, Seoul RAIM (this https URL), emphasizing the harmony of robotics and humanity through visual, acoustic, and physical expression. 

**Abstract (ZH)**: BALLU，一种带有氦气球身体和可动腿的浮力辅助轻型腿足单元，是一种独特的腿足机器人 \fig{fig:fig1}。由于它是浮力辅助的， BALLU 原本就具备稳定性能，不会摔倒，同时能够行走、跳跃，并且能够安全地与人互动。BALLU 艺术装置在此基础上构建了一个充满趣味的平台，表达流动性、偶然性和连接性。它将机器人的运动转变为一种艺术性的视觉和听觉体验，将技术和创意融为一体，形成一个动态、互动的展示。此次展览故意没有为机器人设置物理边界，强调技术和人类的和谐共生。这项工作显著扩展了BALLU在首尔机器人与人工智能博物馆（Seoul RAIM）的永久展览（链接：这个 https URL），通过视觉、听觉和身体上的表达，强调了机器人与人类的和谐共生。 

---
# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics 

**Title (ZH)**: SmolVLA：一种经济高效的人机协作模型 

**Authors**: Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, Remi Cadene  

**Link**: [PDF](https://arxiv.org/pdf/2506.01844)  

**Abstract**: Vision-language models (VLMs) pretrained on large-scale multimodal datasets encode rich visual and linguistic knowledge, making them a strong foundation for robotics. Rather than training robotic policies from scratch, recent approaches adapt VLMs into vision-language-action (VLA) models that enable natural language-driven perception and control. However, existing VLAs are typically massive--often with billions of parameters--leading to high training costs and limited real-world deployability. Moreover, they rely on academic and industrial datasets, overlooking the growing availability of community-collected data from affordable robotic platforms. In this work, we present SmolVLA, a small, efficient, and community-driven VLA that drastically reduces both training and inference costs, while retaining competitive performance. SmolVLA is designed to be trained on a single GPU and deployed on consumer-grade GPUs or even CPUs. To further improve responsiveness, we introduce an asynchronous inference stack decoupling perception and action prediction from action execution, allowing higher control rates with chunked action generation. Despite its compact size, SmolVLA achieves performance comparable to VLAs that are 10x larger. We evaluate SmolVLA on a range of both simulated as well as real-world robotic benchmarks and release all code, pretrained models, and training data. 

**Abstract (ZH)**: 基于视觉-语言的小型高效社区驱动机器人模型：SmolVLA 

---
# Provably Safe Reinforcement Learning from Analytic Gradients 

**Title (ZH)**: 可验证安全的强化学习：基于分析梯度的方法 

**Authors**: Tim Walter, Hannah Markgraf, Jonathan Külz, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.01665)  

**Abstract**: Deploying autonomous robots in safety-critical applications requires safety guarantees. Provably safe reinforcement learning is an active field of research which aims to provide such guarantees using safeguards. These safeguards should be integrated during training to prevent a large sim-to-real gap. While there are several approaches for safeguarding sampling-based reinforcement learning, analytic gradient-based reinforcement learning often achieves superior performance and sample efficiency. However, there is no safeguarding approach for this learning paradigm yet. Our work addresses this gap by developing the first effective safeguard for analytic gradient-based reinforcement learning. We analyse existing, differentiable safeguards, adapt them through modified mappings and gradient formulations, and integrate them with a state-of-the-art learning algorithm and a differentiable simulation. We evaluate how different safeguards affect policy optimisation using numerical experiments on two classical control tasks. The results demonstrate safeguarded training without compromising performance. 

**Abstract (ZH)**: 在安全性关键应用中部署自主机器人需要安全性保证。可证明安全的强化学习是研究的一个活跃领域，旨在通过安全机制提供此类保证。这些安全机制应在训练期间集成以防止仿真到现实世界的差距。尽管已有几种针对基于采样强化学习的保护方法，但分析梯度基强化学习通常能实现更好的性能和样本效率。然而，目前尚无针对这种学习范式的保护方法。我们通过开发首个有效的分析梯度基强化学习保护方法来填补这一空白。我们分析现有可微分保护方法，通过修改映射和梯度公式对它们进行调整，并将它们与最先进的学习算法和可微分模拟集成。我们通过在两个经典控制任务上的数值实验评估不同保护方法对策略优化的影响。结果表明，在不牺牲性能的情况下实现了受保护的训练。 

---
# General agents need world models 

**Title (ZH)**: 通用代理需要世界模型 

**Authors**: Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt  

**Link**: [PDF](https://arxiv.org/pdf/2506.01622)  

**Abstract**: Are world models a necessary ingredient for flexible, goal-directed behaviour, or is model-free learning sufficient? We provide a formal answer to this question, showing that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. We show that this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models. This has a number of consequences: from developing safe and general agents, to bounding agent capabilities in complex environments, and providing new algorithms for eliciting world models from agents. 

**Abstract (ZH)**: 世界模型是实现灵活、目标导向行为的必要成分，还是无模型学习足夠？我们提供了对该问题的形式化回答，表明任何能够泛化到多步目标导向任务的代理都必须学到了其环境的预测模型。我们证明了可以从代理的策略中提取这种模型，并且提高代理的性能或其可实现目标的复杂性需要学习越来越准确的世界模型。这具有多个后果：从开发安全且通用的代理，到限制代理在复杂环境中的能力，以及为从代理中引出世界模型提供新的算法。 

---
# Trajectory First: A Curriculum for Discovering Diverse Policies 

**Title (ZH)**: 先轨迹：一种发现多样化策略的课程学习方法 

**Authors**: Cornelius V. Braun, Sayantan Auddy, Marc Toussaint  

**Link**: [PDF](https://arxiv.org/pdf/2506.01568)  

**Abstract**: Being able to solve a task in diverse ways makes agents more robust to task variations and less prone to local optima. In this context, constrained diversity optimization has emerged as a powerful reinforcement learning (RL) framework to train a diverse set of agents in parallel. However, existing constrained-diversity RL methods often under-explore in complex tasks such as robotic manipulation, leading to a lack in policy diversity. To improve diversity optimization in RL, we therefore propose a curriculum that first explores at the trajectory level before learning step-based policies. In our empirical evaluation, we provide novel insights into the shortcoming of skill-based diversity optimization, and demonstrate empirically that our curriculum improves the diversity of the learned skills. 

**Abstract (ZH)**: 具备多种解决任务的方法可以使智能体更 robust 并减少陷入局部最优解的可能性。在这种背景下，约束多样性优化已成为一种强大的强化学习（RL）框架，用于并行训练一组多样性的智能体。然而，现有的约束多样性 RL 方法往往在诸如机器人 manipulation 等复杂任务中探索不足，导致策略多样性不足。为了改进 RL 中的多样性优化，我们提出了一种课程学习方法，首先在轨迹层面探索，然后再学习基于步骤的策略。在我们的实验评估中，我们提供了关于技能基多样性优化缺陷的新见解，并通过实验证明我们的课程学习方法可以提升学习技能的多样性。 

---
# Position: Olfaction Standardization is Essential for the Advancement of Embodied Artificial Intelligence 

**Title (ZH)**: 位置：嗅觉标准化对于推进具身人工智能至关重要 

**Authors**: Kordel K. France, Rohith Peddi, Nik Dennler, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00398)  

**Abstract**: Despite extraordinary progress in artificial intelligence (AI), modern systems remain incomplete representations of human cognition. Vision, audition, and language have received disproportionate attention due to well-defined benchmarks, standardized datasets, and consensus-driven scientific foundations. In contrast, olfaction - a high-bandwidth, evolutionarily critical sense - has been largely overlooked. This omission presents a foundational gap in the construction of truly embodied and ethically aligned super-human intelligence. We argue that the exclusion of olfactory perception from AI architectures is not due to irrelevance but to structural challenges: unresolved scientific theories of smell, heterogeneous sensor technologies, lack of standardized olfactory datasets, absence of AI-oriented benchmarks, and difficulty in evaluating sub-perceptual signal processing. These obstacles have hindered the development of machine olfaction despite its tight coupling with memory, emotion, and contextual reasoning in biological systems. In this position paper, we assert that meaningful progress toward general and embodied intelligence requires serious investment in olfactory research by the AI community. We call for cross-disciplinary collaboration - spanning neuroscience, robotics, machine learning, and ethics - to formalize olfactory benchmarks, develop multimodal datasets, and define the sensory capabilities necessary for machines to understand, navigate, and act within human environments. Recognizing olfaction as a core modality is essential not only for scientific completeness, but for building AI systems that are ethically grounded in the full scope of the human experience. 

**Abstract (ZH)**: 尽管在人工智能（AI）方面取得了非凡的进步，现代系统仍然是人类认知的不完整表现。由于有明确的基准、标准化的数据集和共识驱动的科学基础，视觉、听觉和语言受到了不成比例的关注。相比之下，作为演化上至关重要的感官之一的嗅觉，却遭到忽视。这种缺失在构建真正具身和伦理对齐的超人类智能时造成了基础性的缺口。我们认为，将嗅觉感知排除在AI架构之外并非由于无足轻重，而是由于结构上的挑战：未解决的嗅觉科学理论、异构传感器技术、缺乏标准化的嗅觉数据集、AI导向的基准缺失以及亚感知信号处理的评估难度。这些障碍阻碍了机器嗅觉的发展，尽管它在生物系统中与记忆、情绪和情境推理紧密耦合。在这篇立场论文中，我们主张，为了取得通识性和具身性智能的实质性进展，人工智能社区需要在嗅觉研究上进行认真投资。我们呼吁跨学科合作——从神经科学、机器人学、机器学习和伦理学等领域——以正式化嗅觉基准、开发多模态数据集，并定义机器理解、导航和在人类环境中行动所需的感官能力。认识嗅觉作为核心模态的重要性不仅对于科学的完整性至关重要，而且对于构建在人类完整体验范围内伦理基础稳固的AI系统也至关重要。 

---
# MotionPersona: Characteristics-aware Locomotion Control 

**Title (ZH)**: MotionPersona：特征感知的运动控制 

**Authors**: Mingyi Shi, Wei Liu, Jidong Mei, Wangpok Tse, Rui Chen, Xuelin Chen, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2506.00173)  

**Abstract**: We present MotionPersona, a novel real-time character controller that allows users to characterize a character by specifying attributes such as physical traits, mental states, and demographics, and projects these properties into the generated motions for animating the character. In contrast to existing deep learning-based controllers, which typically produce homogeneous animations tailored to a single, predefined character, MotionPersona accounts for the impact of various traits on human motion as observed in the real world. To achieve this, we develop a block autoregressive motion diffusion model conditioned on SMPLX parameters, textual prompts, and user-defined locomotion control signals. We also curate a comprehensive dataset featuring a wide range of locomotion types and actor traits to enable the training of this characteristic-aware controller. Unlike prior work, MotionPersona is the first method capable of generating motion that faithfully reflects user-specified characteristics (e.g., an elderly person's shuffling gait) while responding in real time to dynamic control inputs. Additionally, we introduce a few-shot characterization technique as a complementary conditioning mechanism, enabling customization via short motion clips when language prompts fall short. Through extensive experiments, we demonstrate that MotionPersona outperforms existing methods in characteristics-aware locomotion control, achieving superior motion quality and diversity. Results, code, and demo can be found at: this https URL. 

**Abstract (ZH)**: MotionPersona：一种新型实时角色控制器 

---
# Autonomous Behavior and Whole-Brain Dynamics Emerge in Embodied Zebrafish Agents with Model-based Intrinsic Motivation 

**Title (ZH)**: 基于模型的内在动机驱动的 embodied 斑马鱼代理中涌现自主行为与全脑动态 

**Authors**: Reece Keller, Alyn Tornell, Felix Pei, Xaq Pitkow, Leo Kozachkov, Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00138)  

**Abstract**: Autonomy is a hallmark of animal intelligence, enabling adaptive and intelligent behavior in complex environments without relying on external reward or task structure. Existing reinforcement learning approaches to exploration in sparse reward and reward-free environments, including class of methods known as intrinsic motivation, exhibit inconsistent exploration patterns and thus fail to produce robust autonomous behaviors observed in animals. Moreover, systems neuroscience has largely overlooked the neural basis of autonomy, focusing instead on experimental paradigms where animals are motivated by external reward rather than engaging in unconstrained, naturalistic and task-independent behavior. To bridge these gaps, we introduce a novel model-based intrinsic drive explicitly designed to capture robust autonomous exploration observed in animals. Our method (3M-Progress) motivates naturalistic behavior by tracking divergence between the agent's current world model and an ethological prior. We demonstrate that artificial embodied agents trained with 3M-Progress capture the explainable variance in behavioral patterns and whole-brain neural-glial dynamics recorded from autonomously-behaving larval zebrafish, introducing the first goal-driven, population-level model of neural-glial computation. Our findings establish a computational framework connecting model-based intrinsic motivation to naturalistic behavior, providing a foundation for building artificial agents with animal-like autonomy. 

**Abstract (ZH)**: 自主性是动物智能的标志，使动物能够在复杂的环境中表现出适应性和智能行为，而不依赖于外部奖励或任务结构。现有的稀疏奖励和无奖励环境中探索的强化学习方法，包括内在动机这类方法，表现出不一致的探索模式，因此无法产生在动物中观察到的稳健自主行为。此外，系统神经科学大多忽略了自主性的神经基础，而是将焦点放在动物被外部奖励驱动的实验范式上，而不是自然无约束的任务独立行为。为填补这些空白，我们引入了一个新的基于模型的内在驱动力，明确设计用于捕捉动物中观察到的稳健自主探索。我们的方法（3M-Progress）通过追踪智能体当前世界模型与生态学先验之间的差异来激励自然行为。我们证明，使用3M-Progress训练的合成躯体化代理能够捕捉自主行为的动态模式和来自自主行为的涡偶鱼幼虫的全脑神经-胶质动力学中的可解释变方，首次提出了目标驱动的神经-胶质计算的群体级模型。我们的发现建立了将基于模型的内在动机与自然行为联系起来的计算框架，为构建具有类似动物自主性的智能代理提供了基础。 

---
# Visual Embodied Brain: Let Multimodal Large Language Models See, Think, and Control in Spaces 

**Title (ZH)**: 视觉 bodyswarm 脑: 让多模态大型语言模型在空间中观察、思考和控制 

**Authors**: Gen Luo, Ganlin Yang, Ziyang Gong, Guanzhou Chen, Haonan Duan, Erfei Cui, Ronglei Tong, Zhi Hou, Tianyi Zhang, Zhe Chen, Shenglong Ye, Lewei Lu, Jingbo Wang, Wenhai Wang, Jifeng Dai, Yu Qiao, Rongrong Ji, Xizhou Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00123)  

**Abstract**: The remarkable progress of Multimodal Large Language Models (MLLMs) has attracted increasing attention to extend them to physical entities like legged robot. This typically requires MLLMs to not only grasp multimodal understanding abilities, but also integrate visual-spatial reasoning and physical interaction capabilities. Nevertheless,existing methods struggle to unify these capabilities due to their fundamental this http URL this paper, we present the Visual Embodied Brain (VeBrain), a unified framework for perception, reasoning, and control in real world. VeBrain reformulates robotic control into common text-based MLLM tasks in the 2D visual space, thus unifying the objectives and mapping spaces of different tasks. Then, a novel robotic adapter is proposed to convert textual control signals from MLLMs to motion policies of real robots. From the data perspective, we further introduce VeBrain-600k, a high-quality instruction dataset encompassing various capabilities of VeBrain. In VeBrain-600k, we take hundreds of hours to collect, curate and annotate the data, and adopt multimodal chain-of-thought(CoT) to mix the different capabilities into a single conversation. Extensive experiments on 13 multimodal benchmarks and 5 spatial intelligence benchmarks demonstrate the superior performance of VeBrain to existing MLLMs like Qwen2.5-VL. When deployed to legged robots and robotic arms, VeBrain shows strong adaptability, flexibility, and compositional capabilities compared to existing methods. For example, compared to Qwen2.5-VL, VeBrain not only achieves substantial gains on MMVet by +5.6%, but also excels in legged robot tasks with +50% average gains. 

**Abstract (ZH)**: Multimodal Large Language Models for Physical Entities: The Visual Embodied Brain (VeBrain)Unified Framework for Perception, Reasoning, and Control 

---
# Human sensory-musculoskeletal modeling and control of whole-body movements 

**Title (ZH)**: 人类感觉-运动系统建模与全身运动控制 

**Authors**: Chenhui Zuo, Guohao Lin, Chen Zhang, Shanning Zhuang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00071)  

**Abstract**: Coordinated human movement depends on the integration of multisensory inputs, sensorimotor transformation, and motor execution, as well as sensory feedback resulting from body-environment interaction. Building dynamic models of the sensory-musculoskeletal system is essential for understanding movement control and investigating human behaviours. Here, we report a human sensory-musculoskeletal model, termed SMS-Human, that integrates precise anatomical representations of bones, joints, and muscle-tendon units with multimodal sensory inputs involving visual, vestibular, proprioceptive, and tactile components. A stage-wise hierarchical deep reinforcement learning framework was developed to address the inherent challenges of high-dimensional control in musculoskeletal systems with integrated multisensory information. Using this framework, we demonstrated the simulation of three representative movement tasks, including bipedal locomotion, vision-guided object manipulation, and human-machine interaction during bicycling. Our results showed a close resemblance between natural and simulated human motor behaviours. The simulation also revealed musculoskeletal dynamics that could not be directly measured. This work sheds deeper insights into the sensorimotor dynamics of human movements, facilitates quantitative understanding of human behaviours in interactive contexts, and informs the design of systems with embodied intelligence. 

**Abstract (ZH)**: 协调的人类运动取决于多感官输入的整合、传感器运动转换以及运动执行，并且依赖于身体与环境相互作用所产生的感觉反馈。构建感觉-肌骨系统的动态模型是理解运动控制和探索人类行为的基础。在这里，我们报告了一个名为SMS-Human的人类感觉-肌骨模型，该模型整合了精确的骨骼、关节和肌腱单位的解剖学表示以及涉及视觉、前庭、本体感觉和触觉的多模态感觉输入。我们开发了一种分阶段层次化的深度强化学习框架，以解决集成多感官信息的肌骨系统中固有的高维控制难题。利用这一框架，我们展示了三种代表性运动任务的模拟，包括双足行走、视觉引导下的物体操作以及骑自行车时的人机交互。我们的结果表明，自然的人类运动行为与模拟行为相似。模拟还揭示了无法直接测量的肌骨系统动力学。这项工作深入探讨了人类运动的传感器运动动力学，促进了在交互上下文中对人类行为的量化理解，并为具有体态智能系统的开发提供了指导。 

---
# RoboEgo System Card: An Omnimodal Model with Native Full Duplexity 

**Title (ZH)**: RoboEgo系统卡：一种具备原生全双工能力的多模态模型 

**Authors**: Yiqun Yao, Xiang Li, Xin Jiang, Xuezhi Fang, Naitong Yu, Aixin Sun, Yequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01934)  

**Abstract**: Humans naturally process real-world multimodal information in a full-duplex manner. In artificial intelligence, replicating this capability is essential for advancing model development and deployment, particularly in embodied contexts. The development of multimodal models faces two primary challenges: (1) effectively handling more than three modalities-such as vision, audio, and text; and (2) delivering full-duplex responses to rapidly evolving human instructions. To facilitate research on models that support both omnimodal processing and full duplexity, we present RoboEgo (alias: FLM-Ego), a unified model system designed to address both challenges. RoboEgo incorporates a backbone architecture and algorithms that natively support full duplexity, achieving a theoretical duplex latency of 80 ms. In streaming visually grounded conversations under real-world conditions, RoboEgo exhibits superior responsiveness and speech naturalness, while maintaining comparable content qualities to state-of-the-art semi-duplex omnimodal models-a feat previously considered unattainable by native full-duplex systems. 

**Abstract (ZH)**: 人类自然地以全双工方式处理真实世界多模态信息。在人工智能领域，复制这种能力对于推进模型开发和部署，尤其是在体感知情境中，至关重要。多模态模型的开发面临两大主要挑战：（1）有效处理超过三种模态的信息，如视觉、音频和文本；（2）提供针对快速变化的人类指令的全双工响应。为了促进同时支持全方位处理和全双工能力的研究，我们提出了RoboEgo（别名：FLM-Ego），一个旨在解决这两个挑战的统一模型系统。RoboEgo整合了一个支持全双工的骨干架构和算法，理论上的双工延迟为80 ms。在真实的实时视觉支撑对话中，RoboEgo表现出卓越的响应能力和口语自然度，同时保持与当前最先进的半双工全方位模型相近的内容质量，这是原生全双工系统此前被认为无法实现的。 

---
# Social Cooperation in Conversational AI Agents 

**Title (ZH)**: 对话式人工智能代理中的社会协作 

**Authors**: Mustafa Mert Çelikok, Saptarashmi Bandyopadhyay, Robert Loftin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01624)  

**Abstract**: The development of AI agents based on large, open-domain language models (LLMs) has paved the way for the development of general-purpose AI assistants that can support human in tasks such as writing, coding, graphic design, and scientific research. A major challenge with such agents is that, by necessity, they are trained by observing relatively short-term interactions with humans. Such models can fail to generalize to long-term interactions, for example, interactions where a user has repeatedly corrected mistakes on the part of the agent. In this work, we argue that these challenges can be overcome by explicitly modeling humans' social intelligence, that is, their ability to build and maintain long-term relationships with other agents whose behavior cannot always be predicted. By mathematically modeling the strategies humans use to communicate and reason about one another over long periods of time, we may be able to derive new game theoretic objectives against which LLMs and future AI agents may be optimized. 

**Abstract (ZH)**: 基于大型开放域语言模型（LLMs）的AI代理的发展为开发能够支持人类在写作、编程、图形设计和科学研究等任务方面的通用AI助手铺平了道路。这类代理的一个主要挑战是，它们必须通过观察与人类相对短暂的互动来训练，这可能导致它们难以将模型泛化到长时间的互动中，例如用户反复纠正代理错误的情况。在本工作中，我们argue可以通过明确建模人类的社会智能，即他们与行为难以预测的其他代理建立并维持长期关系的能力，来克服这些挑战。通过数学建模人类在长时间内相互交流和推理的策略，我们或许能够推导出新的博弈论目标，以优化LLMs和未来AI代理。 

---
# Agentic Episodic Control 

**Title (ZH)**: 代理性事件控制 

**Authors**: Xidong Yang, Wenhao Li, Junjie Sheng, Chuyun Shen, Yun Hua, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01442)  

**Abstract**: Reinforcement learning (RL) has driven breakthroughs in AI, from game-play to scientific discovery and AI alignment. However, its broader applicability remains limited by challenges such as low data efficiency and poor generalizability. Recent advances suggest that large language models, with their rich world knowledge and reasoning capabilities, could complement RL by enabling semantic state modeling and task-agnostic planning. In this work, we propose the Agentic Episodic Control (AEC), a novel architecture that integrates RL with LLMs to enhance decision-making. The AEC can leverage a large language model (LLM) to map the observations into language-grounded embeddings, which further can be stored in an episodic memory for rapid retrieval of high-value experiences. Simultaneously, a World-Graph working memory module is utilized to capture structured environmental dynamics in order to enhance relational reasoning. Furthermore, a lightweight critical state detector dynamically arbitrates between the episodic memory recall and the world-model-guided exploration. On the whole, by combining the trial-and-error learning scheme with LLM-derived semantic priors, the proposed AEC can improve both data efficiency and generalizability in reinforcement learning. In experiments on BabyAI-Text benchmark tasks, AEC demonstrates substantial improvements over existing baselines, especially on complex and generalization tasks like FindObj, where it outperforms the best baseline by up to 76%. The proposed AEC framework bridges the strengths of numeric reinforcement learning and symbolic reasoning, which provides a pathway toward more adaptable and sample-efficient agents. 

**Abstract (ZH)**: 基于大型语言模型的强化学习代理人 episodic 控制 (AEC) 架构 

---
# FinRobot: Generative Business Process AI Agents for Enterprise Resource Planning in Finance 

**Title (ZH)**: FinRobot: 生成式商务过程AI代理在金融企业资源规划中应用 

**Authors**: Hongyang Yang, Likun Lin, Yang She, Xinyu Liao, Jiaoyang Wang, Runjia Zhang, Yuquan Mo, Christina Dan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01423)  

**Abstract**: Enterprise Resource Planning (ERP) systems serve as the digital backbone of modern financial institutions, yet they continue to rely on static, rule-based workflows that limit adaptability, scalability, and intelligence. As business operations grow more complex and data-rich, conventional ERP platforms struggle to integrate structured and unstructured data in real time and to accommodate dynamic, cross-functional workflows.
In this paper, we present the first AI-native, agent-based framework for ERP systems, introducing a novel architecture of Generative Business Process AI Agents (GBPAs) that bring autonomy, reasoning, and dynamic optimization to enterprise workflows. The proposed system integrates generative AI with business process modeling and multi-agent orchestration, enabling end-to-end automation of complex tasks such as budget planning, financial reporting, and wire transfer processing. Unlike traditional workflow engines, GBPAs interpret user intent, synthesize workflows in real time, and coordinate specialized sub-agents for modular task execution. We validate the framework through case studies in bank wire transfers and employee reimbursements, two representative financial workflows with distinct complexity and data modalities. Results show that GBPAs achieve up to 40% reduction in processing time, 94% drop in error rate, and improved regulatory compliance by enabling parallelism, risk control insertion, and semantic reasoning. These findings highlight the potential of GBPAs to bridge the gap between generative AI capabilities and enterprise-grade automation, laying the groundwork for the next generation of intelligent ERP systems. 

**Abstract (ZH)**: 基于AI的企业资源规划系统：生成式商业过程智能代理的架构 

---
# AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning 

**Title (ZH)**: AgentCPM-GUI: 构建基于强化微调的移动用途代理 

**Authors**: Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, Jie Xie, Wei Zhou, Wang Xu, Yuanheng Zhang, Zhou Su, Zhongwu Zhai, Xiaoming Liu, Yudong Mei, Jianming Xu, Hongyan Tian, Chongyi Wang, Chi Chen, Yuan Yao, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01391)  

**Abstract**: The recent progress of large language model agents has opened new possibilities for automating tasks through graphical user interfaces (GUIs), especially in mobile environments where intelligent interaction can greatly enhance usability. However, practical deployment of such agents remains constrained by several key challenges. Existing training data is often noisy and lack semantic diversity, which hinders the learning of precise grounding and planning. Models trained purely by imitation tend to overfit to seen interface patterns and fail to generalize in unfamiliar scenarios. Moreover, most prior work focuses on English interfaces while overlooks the growing diversity of non-English applications such as those in the Chinese mobile ecosystem. In this work, we present AgentCPM-GUI, an 8B-parameter GUI agent built for robust and efficient on-device GUI interaction. Our training pipeline includes grounding-aware pre-training to enhance perception, supervised fine-tuning on high-quality Chinese and English trajectories to imitate human-like actions, and reinforcement fine-tuning with GRPO to improve reasoning capability. We also introduce a compact action space that reduces output length and supports low-latency execution on mobile devices. AgentCPM-GUI achieves state-of-the-art performance on five public benchmarks and a new Chinese GUI benchmark called CAGUI, reaching $96.9\%$ Type-Match and $91.3\%$ Exact-Match. To facilitate reproducibility and further research, we publicly release all code, model checkpoint, and evaluation data. 

**Abstract (ZH)**: 大型语言模型代理在图形用户界面中的Recent进展及其在移动环境中的智能交互应用：AgentCPM-GUI的研究 

---
# Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner 

**Title (ZH)**: 克服多步复杂性在多模态理论思维推理中的障碍：一个可扩展的贝叶斯规划者 

**Authors**: Chunhui Zhang, Zhongyu Ouyang, Kwonjoon Lee, Nakul Agarwal, Sean Dae Houlihan, Soroush Vosoughi, Shao-Yuan Lo  

**Link**: [PDF](https://arxiv.org/pdf/2506.01301)  

**Abstract**: Theory-of-Mind (ToM) enables humans to infer mental states-such as beliefs, desires, and intentions-forming the foundation of social cognition. However, existing computational ToM methods rely on structured workflows with ToM-specific priors or deep model fine-tuning, which struggle with scalability in multimodal environments and fail to generalize as task complexity increases. To address these limitations, we propose a scalable Bayesian ToM planner that decomposes ToM reasoning into stepwise Bayesian updates. Our framework introduces weak-to-strong control, allowing smaller language models (LMs) to specialize in ToM-specific likelihood estimation and transfer their reasoning behaviors to larger LMs (7B to 405B) for integration with social and world knowledge. This synergistic approach aligns large-model inference of human mental states with Bayesian principles. Extensive experiments show that our method achieves a 4.6% accuracy improvement over state-of-the-art techniques on multimodal ToM benchmarks, including challenging unseen scenarios, thereby establishing a new standard for modeling human mental states in complex environments. 

**Abstract (ZH)**: Theory-of-Mind (ToM) 理论使人类能够推断信念、欲望和意图等心理状态，奠定社会认知的基础。然而，现有的计算ToM方法依赖于有ToM特定先验或深度模型微调的结构化工作流程，在多模态环境中难以扩展，并且随着任务复杂性的增加而难以泛化。为解决这些限制，我们提出了一种可扩展的贝叶斯ToM规划器，将ToM推理分解为逐步的贝叶斯更新。该框架引入了从弱到强的控制，允许较小的语言模型（LMs）专门从事ToM特定的似然估计，并将其实现的推理行为转移到更大的LMs（从7B到405B），以便将社会和世界知识集成进去。这种协同方法将大型模型对人类心理状态的推理与贝叶斯原则对齐。广泛的实验结果显示，我们的方法在包括具有挑战性的未见过场景的多模态ToM基准测试上，较最先进的技术在准确率上提高了4.6%，从而建立了在复杂环境中建模人类心理状态的新标准。 

---
# GraphPad: Inference-Time 3D Scene Graph Updates for Embodied Question Answering 

**Title (ZH)**: GraphPad: 语义理解时的3D场景图更新方法在体帧问答中的应用 

**Authors**: Muhammad Qasim Ali, Saeejith Nair, Alexander Wong, Yuchen Cui, Yuhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01174)  

**Abstract**: Structured scene representations are a core component of embodied agents, helping to consolidate raw sensory streams into readable, modular, and searchable formats. Due to their high computational overhead, many approaches build such representations in advance of the task. However, when the task specifications change, such static approaches become inadequate as they may miss key objects, spatial relations, and details. We introduce GraphPad, a modifiable structured memory that an agent can tailor to the needs of the task through API calls. It comprises a mutable scene graph representing the environment, a navigation log indexing frame-by-frame content, and a scratchpad for task-specific notes. Together, GraphPad serves as a dynamic workspace that remains complete, current, and aligned with the agent's immediate understanding of the scene and its task. On the OpenEQA benchmark, GraphPad attains 55.3%, a +3.0% increase over an image-only baseline using the same vision-language model, while operating with five times fewer input frames. These results show that allowing online, language-driven refinement of 3-D memory yields more informative representations without extra training or data collection. 

**Abstract (ZH)**: 基于图的可修改结构化记忆：一种通过API调用适应任务需求的动态工作空间 

---
# SuperRL: Reinforcement Learning with Supervision to Boost Language Model Reasoning 

**Title (ZH)**: SuperRL：带有监督的强化学习以增强语言模型推理能力 

**Authors**: Yihao Liu, Shuocheng Li, Lang Cao, Yuhang Xie, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01096)  

**Abstract**: Large language models are increasingly used for complex reasoning tasks where high-quality offline data such as expert-annotated solutions and distilled reasoning traces are often available. However, in environments with sparse rewards, reinforcement learning struggles to sample successful trajectories, leading to inefficient learning. At the same time, these offline trajectories that represent correct reasoning paths are not utilized by standard on-policy reinforcement learning methods. To address this limitation, we propose SuperRL, a unified training framework that adaptively incorporates offline supervision into reinforcement learning. SuperRL introduces an Adaptive Switch to detect sparse reward conditions and activates a Hybrid Actor when necessary. The Hybrid Actor integrates policy gradient and supervised learning objectives at the loss level, enabling the model to benefit from accurate offline reasoning signals while maintaining the exploratory capacity of reinforcement learning. Experiments on a range of reasoning benchmarks show that SuperRL consistently outperforms standard reinforcement learning by improving sample efficiency, generalization, and robustness under sparse rewards. 

**Abstract (ZH)**: SuperRL：统一训练框架下的自适应离线监督强化学习 

---
# Aligning VLM Assistants with Personalized Situated Cognition 

**Title (ZH)**: 个性化情境认知中的VLM辅助系统对齐 

**Authors**: Yongqi Li, Shen Zhou, Xiaohu Li, Xin Miao, Jintao Wen, Mayi Xu, Jianhao Chen, Birong Pan, Hankun Kang, Yuanyuan Zhu, Ming Zhong, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.00930)  

**Abstract**: Vision-language models (VLMs) aligned with general human objectives, such as being harmless and hallucination-free, have become valuable assistants of humans in managing visual tasks. However, people with diversified backgrounds have different cognition even in the same situation. Consequently, they may have personalized expectations for VLM assistants. This highlights the urgent need to align VLM assistants with personalized situated cognition for real-world assistance. To study this problem, we first simplify it by characterizing individuals based on the sociological concept of Role-Set. Then, we propose to evaluate the individuals' actions to examine whether the personalized alignment is achieved. Further, we construct a benchmark named PCogAlignBench, which includes 18k instances and 20 individuals with different Role-Sets. Finally, we present a framework called PCogAlign, which constructs a cognition-aware and action-based reward model for personalized alignment. Experimental results and human evaluations demonstrate the reliability of the PCogAlignBench and the effectiveness of our proposed PCogAlign. We will open-source the constructed benchmark and code at this https URL. 

**Abstract (ZH)**: Vision-language模型（VLMs）与普遍人类目标对齐，如无害和无幻觉，已成为人类在管理视觉任务中有价值的助理。然而，具有不同背景的人在同一情境下可能有不同的认知，因此他们可能对VLM助理有个性化的期望。这突显了在真实世界协助中对VLM助理进行个性化情境对齐的迫切需求。为研究这一问题，我们首先根据社会学概念Role-Set对该问题进行简化，基于此对个体进行刻画，然后提出通过评估个体的行为来检验个性化对齐是否实现。进一步地，我们构建了一个基准PCogAlignBench，其中包括18000个实例和20名具有不同Role-Set的个体。最后，我们提出了一种名为PCogAlign的框架，该框架构建了一个基于认知和行为的个性化对齐奖励模型。实验结果和人工评估证明了PCogAlignBench的可靠性和我们提出的PCogAlign的有效性。我们将在此网址公开所构建的基准和代码：this https URL。 

---
# Toward a Theory of Agents as Tool-Use Decision-Makers 

**Title (ZH)**: 面向代理作为工具使用决策者的理论研究 

**Authors**: Hongru Wang, Cheng Qian, Manling Li, Jiahao Qiu, Boyang Xue, Mengdi Wang, Heng Ji, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.00886)  

**Abstract**: As Large Language Models (LLMs) evolve into increasingly autonomous agents, fundamental questions about their epistemic foundations remain unresolved: What defines an agent? How should it make decisions? And what objectives should guide its behavior? In this position paper, we argue that true autonomy requires agents to be grounded in a coherent epistemic framework that governs what they know, what they need to know, and how to acquire that knowledge efficiently. We propose a unified theory that treats internal reasoning and external actions as equivalent epistemic tools, enabling agents to systematically coordinate introspection and interaction. Building on this framework, we advocate for aligning an agent's tool use decision-making boundary with its knowledge boundary, thereby minimizing unnecessary tool use and maximizing epistemic efficiency. This perspective shifts the design of agents from mere action executors to knowledge-driven intelligence systems, offering a principled path toward building foundation agents capable of adaptive, efficient, and goal-directed behavior. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）演变为日益自主的代理，关于其知识基础的基本问题仍未解决：什么是代理？它应如何做出决策？其行为应由什么目标指导？在本文中，我们argue，真正的自主性要求代理具备一套连贯的知识框架，以规范其认知、所需认知及其知识获取的效率。我们提出了一种统一理论，将内部推理和外部行动视为等价的知识工具，使代理能够系统地协调内省和互动。基于此框架，我们提倡将代理的工具使用决策边界与其知识边界对齐，从而减少不必要的工具使用，最大化知识效率。这种观点将代理的设计从单纯的行动执行者转变为以知识为导向的智能系统，提供了构建能够实现自适应、高效和目标导向行为的基础代理的原理性路径。 

---
# Predicting Empirical AI Research Outcomes with Language Models 

**Title (ZH)**: 使用语言模型预测 empirical AI 研究成果 

**Authors**: Jiaxin Wen, Chenglei Si, Yueh-han Chen, He He, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00794)  

**Abstract**: Many promising-looking ideas in AI research fail to deliver, but their validation takes substantial human labor and compute. Predicting an idea's chance of success is thus crucial for accelerating empirical AI research, a skill that even expert researchers can only acquire through substantial experience. We build the first benchmark for this task and compare LMs with human experts. Concretely, given two research ideas (e.g., two jailbreaking methods), we aim to predict which will perform better on a set of benchmarks. We scrape ideas and experimental results from conference papers, yielding 1,585 human-verified idea pairs published after our base model's cut-off date for testing, and 6,000 pairs for training. We then develop a system that combines a fine-tuned GPT-4.1 with a paper retrieval agent, and we recruit 25 human experts to compare with. In the NLP domain, our system beats human experts by a large margin (64.4% v.s. 48.9%). On the full test set, our system achieves 77% accuracy, while off-the-shelf frontier LMs like o3 perform no better than random guessing, even with the same retrieval augmentation. We verify that our system does not exploit superficial features like idea complexity through extensive human-written and LM-designed robustness tests. Finally, we evaluate our system on unpublished novel ideas, including ideas generated by an AI ideation agent. Our system achieves 63.6% accuracy, demonstrating its potential as a reward model for improving idea generation models. Altogether, our results outline a promising new direction for LMs to accelerate empirical AI research. 

**Abstract (ZH)**: AI研究中想法验证的第一个基准及其与人类专家的比较 

---
# A "Wenlu" Brain System for Multimodal Cognition and Embodied Decision-Making: A Secure New Architecture for Deep Integration of Foundation Models and Domain Knowledge 

**Title (ZH)**: “文溯”脑系统：多模态认知与 embodied 决策的新安全架构——基础模型与领域知识的深度集成 

**Authors**: Liang Geng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00570)  

**Abstract**: With the rapid penetration of artificial intelligence across industries and scenarios, a key challenge in building the next-generation intelligent core lies in effectively integrating the language understanding capabilities of foundation models with domain-specific knowledge bases in complex real-world applications. This paper proposes a multimodal cognition and embodied decision-making brain system, ``Wenlu", designed to enable secure fusion of private knowledge and public models, unified processing of multimodal data such as images and speech, and closed-loop decision-making from cognition to automatic generation of hardware-level code. The system introduces a brain-inspired memory tagging and replay mechanism, seamlessly integrating user-private data, industry-specific knowledge, and general-purpose language models. It provides precise and efficient multimodal services for enterprise decision support, medical analysis, autonomous driving, robotic control, and more. Compared with existing solutions, ``Wenlu" demonstrates significant advantages in multimodal processing, privacy security, end-to-end hardware control code generation, self-learning, and sustainable updates, thus laying a solid foundation for constructing the next-generation intelligent core. 

**Abstract (ZH)**: 随着人工智能在各行各业和各种场景中的快速渗透，构建下一代智能核心的关键挑战在于有效地将基础模型的语言理解能力与复杂现实应用中的领域特定知识库集成起来。“ Wenlu”是一种多模态认知和实体决策脑系统，旨在实现私人知识和公共模型的安全融合、多模态数据（如图像和语音）的一体化处理以及从认知到硬件级代码自动生成的闭环决策。该系统引入了一种受大脑启发的内存标记和回放机制，无缝集成用户私人数据、行业特定知识和通用语言模型。它为企业的决策支持、医疗分析、自动驾驶、机器人控制等领域提供精确高效的多模态服务。与现有解决方案相比，“ Wenlu”在多模态处理、隐私安全性、端到端硬件控制代码生成、自我学习和可持续更新方面展现出显著优势，从而为构建下一代智能核心奠定了坚实基础。 

---
# World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks 

**Title (ZH)**: 认知代理的 WORLD MODELS：转换未来网络中的边缘智能 

**Authors**: Changyuan Zhao, Ruichen Zhang, Jiacheng Wang, Gaosheng Zhao, Dusit Niyato, Geng Sun, Shiwen Mao, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.00417)  

**Abstract**: World models are emerging as a transformative paradigm in artificial intelligence, enabling agents to construct internal representations of their environments for predictive reasoning, planning, and decision-making. By learning latent dynamics, world models provide a sample-efficient framework that is especially valuable in data-constrained or safety-critical scenarios. In this paper, we present a comprehensive overview of world models, highlighting their architecture, training paradigms, and applications across prediction, generation, planning, and causal reasoning. We compare and distinguish world models from related concepts such as digital twins, the metaverse, and foundation models, clarifying their unique role as embedded cognitive engines for autonomous agents. We further propose Wireless Dreamer, a novel world model-based reinforcement learning framework tailored for wireless edge intelligence optimization, particularly in low-altitude wireless networks (LAWNs). Through a weather-aware UAV trajectory planning case study, we demonstrate the effectiveness of our framework in improving learning efficiency and decision quality. 

**Abstract (ZH)**: 世界模型已在人工智能领域 emerges 为一种变革性的范式，使代理能够构建其环境的内部表示以进行预测推理、规划和决策。通过学习潜在动力学，世界模型提供了一种样本高效的框架，特别是在数据受限或安全性关键的情景下特别有价值。在本文中，我们概述了世界模型的全面概况，强调其架构、训练范式及其在预测、生成、规划和因果推理中的应用。我们将世界模型与相关概念如数字孪生、元宇宙和基础模型进行比较和区分，阐明其作为嵌入式认知引擎的独特角色，为自主代理服务。我们进一步提出了基于世界模型的无线梦者（Wireless Dreamer）强化学习框架，该框架特别针对低空无线网络（LAWNs）的无线边缘智能优化。通过一个具备气象感知的无人机轨迹规划案例研究，我们展示了该框架在提高学习效率和决策质量方面的作用。 

---
# Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents 

**Title (ZH)**: Dyna-Think: 结合推理、行动与世界模型模拟的AI代理方法 

**Authors**: Xiao Yu, Baolin Peng, Ruize Xu, Michel Galley, Hao Cheng, Suman Nath, Jianfeng Gao, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00320)  

**Abstract**: Recent progress in reasoning with large language models (LLMs), such as DeepSeek-R1, demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities. 

**Abstract (ZH)**: Recent progress in reasoning with large language models (LLMs) such as DeepSeek-R1 demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities. 

---
# Agnostic Reinforcement Learning: Foundations and Algorithms 

**Title (ZH)**: agnostic强化学习：基础与算法 

**Authors**: Gene Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01884)  

**Abstract**: Reinforcement Learning (RL) has demonstrated tremendous empirical success across numerous challenging domains. However, we lack a strong theoretical understanding of the statistical complexity of RL in environments with large state spaces, where function approximation is required for sample-efficient learning. This thesis addresses this gap by rigorously examining the statistical complexity of RL with function approximation from a learning theoretic perspective. Departing from a long history of prior work, we consider the weakest form of function approximation, called agnostic policy learning, in which the learner seeks to find the best policy in a given class $\Pi$, with no guarantee that $\Pi$ contains an optimal policy for the underlying task.
We systematically explore agnostic policy learning along three key axes: environment access -- how a learner collects data from the environment; coverage conditions -- intrinsic properties of the underlying MDP measuring the expansiveness of state-occupancy measures for policies in the class $\Pi$, and representational conditions -- structural assumptions on the class $\Pi$ itself. Within this comprehensive framework, we (1) design new learning algorithms with theoretical guarantees and (2) characterize fundamental performance bounds of any algorithm. Our results reveal significant statistical separations that highlight the power and limitations of agnostic policy learning. 

**Abstract (ZH)**: 强化学习（RL）在众多具有挑战性的领域中展现出了巨大的经验成功。然而，我们对在具有大规模状态空间的环境中需要函数逼近以实现高效样本学习的RL的统计复杂性缺乏强有力的理论理解。本文从学习理论的角度严格探讨了函数逼近环境下RL的统计复杂性，不同于以往工作的长期研究，我们考虑了最弱形式的函数逼近——即不保证函数类包含最优策略的无偏策略学习。我们系统地从环境访问、覆盖条件和表示条件三个关键维度探索了无偏策略学习，并在这一综合框架中（1）设计了具有理论保证的新学习算法，（2）界定了任何算法的基本性能界。我们的结果揭示了无偏策略学习的重要统计分离，突显了其能力和局限性。 

---
# EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation 

**Title (ZH)**: EvolveNav：自我提升的嵌入式推理用于基于LLM的视觉语言导航 

**Authors**: Bingqian Lin, Yunshuang Nie, Khun Loun Zai, Ziming Wei, Mingfei Han, Rongtao Xu, Minzhe Niu, Jianhua Han, Liang Lin, Cewu Lu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01551)  

**Abstract**: Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at this https URL. 

**Abstract (ZH)**: 构建能够遵循自然语言指令进行导航的Vision-Language Navigation (VLN) 代理是人机交互应用中的一个长期目标。最近的研究揭示了通过训练开源大型语言模型（LLMs）来释放模型的推理能力，以改进导航并同时减轻LLMs训练语料库与VLN任务之间的领域差距的潜力。然而，这些方法主要采用直接输入-输出映射的范式，导致映射学习困难和导航决策难以解释。链式思考（CoT）训练是一种有望同时提高导航决策准确性和可解释性的方法，但由于导航任务的复杂性，使得完美的CoT标签不可用，且可能导致仅通过纯CoT监督微调产生过拟合。本文提出了一种新的自我改进的嵌入式推理框架，以提升基于LLMs的Vision-Language导航性能，名为EvolveNav。EvolveNav包含两个阶段：（1）形式化的CoT监督微调，我们使用形式化的CoT标签训练模型，以激活模型的导航推理能力并提高推理速度；（2）自我反思的后训练，模型通过迭代使用其自身推理输出作为自我丰富的CoT标签来增强监督多样性。还引入了一个自我反思的辅助任务，通过与错误的推理模式对比来促进正确推理模式的学习。在流行的VLN基准测试上的实验结果证明了EvolveNav相较于之前的基于LLMs的VLN方法的优势。代码可在此处访问。 

---
# Agentic AI and Multiagentic: Are We Reinventing the Wheel? 

**Title (ZH)**: 代理AI与多代理系统：我们是否在重造轮子？ 

**Authors**: V.Botti  

**Link**: [PDF](https://arxiv.org/pdf/2506.01463)  

**Abstract**: The terms Agentic AI and Multiagentic AI have recently gained popularity in discussions on generative artificial intelligence, often used to describe autonomous software agents and systems composed of such agents. However, the use of these terms confuses these buzzwords with well-established concepts in AI literature: intelligent agents and multi-agent systems. This article offers a critical analysis of this conceptual misuse. We review the theoretical origins of "agentic" in the social sciences (Bandura, 1986) and philosophical notions of intentionality (Dennett, 1971), and then summarise foundational works on intelligent agents and multi-agent systems by Wooldridge, Jennings and others. We examine classic agent architectures, from simple reactive agents to Belief-Desire-Intention (BDI) models, and highlight key properties (autonomy, reactivity, proactivity, social capability) that define agency in AI. We then discuss recent developments in large language models (LLMs) and agent platforms based on LLMs, including the emergence of LLM-powered AI agents and open-source multi-agent orchestration frameworks. We argue that the term AI Agentic is often used as a buzzword for what are essentially AI agents, and AI Multiagentic for what are multi-agent systems. This confusion overlooks decades of research in the field of autonomous agents and multi-agent systems. The article advocates for scientific and technological rigour and the use of established terminology from the state of the art in AI, incorporating the wealth of existing knowledge, including standards for multi-agent system platforms, communication languages and coordination and cooperation algorithms, agreement technologies (automated negotiation, argumentation, virtual organisations, trust, reputation, etc.), into the new and promising wave of LLM-based AI agents, so as not to end up reinventing the wheel. 

**Abstract (ZH)**: Agentic AI与Multiagentic AI概念的批判性分析：从自主智能体和多智能体系统视角探究 

---
# ReFoCUS: Reinforcement-guided Frame Optimization for Contextual Understanding 

**Title (ZH)**: ReFoCUS: 基于强化学习的框架优化以实现上下文理解 

**Authors**: Hosu Lee, Junho Kim, Hyunjun Kim, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2506.01274)  

**Abstract**: Recent progress in Large Multi-modal Models (LMMs) has enabled effective vision-language reasoning, yet the ability to understand video content remains constrained by suboptimal frame selection strategies. Existing approaches often rely on static heuristics or external retrieval modules to feed frame information into video-LLMs, which may fail to provide the query-relevant information. In this work, we introduce ReFoCUS (Reinforcement-guided Frame Optimization for Contextual UnderStanding), a novel frame-level policy optimization framework that shifts the optimization target from textual responses to visual input selection. ReFoCUS learns a frame selection policy via reinforcement learning, using reward signals derived from a reference LMM to reflect the model's intrinsic preferences for frames that best support temporally grounded responses. To efficiently explore the large combinatorial frame space, we employ an autoregressive, conditional selection architecture that ensures temporal coherence while reducing complexity. Our approach does not require explicit supervision at the frame-level and consistently improves reasoning performance across multiple video QA benchmarks, highlighting the benefits of aligning frame selection with model-internal utility. 

**Abstract (ZH)**: Recent Progress in Large Multi-modal Models: Reinforcement-guided Frame Optimization for Contextual Understanding 

---
# Neuro-Symbolic Generative Diffusion Models for Physically Grounded, Robust, and Safe Generation 

**Title (ZH)**: 基于物理约束的神经符号生成扩散模型：稳健且安全的内容生成 

**Authors**: Jacob K. Christopher, Michael Cardei, Jinhao Liang, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2506.01121)  

**Abstract**: Despite the remarkable generative capabilities of diffusion models, their integration into safety-critical or scientifically rigorous applications remains hindered by the need to ensure compliance with stringent physical, structural, and operational constraints. To address this challenge, this paper introduces Neuro-Symbolic Diffusion (NSD), a novel framework that interleaves diffusion steps with symbolic optimization, enabling the generation of certifiably consistent samples under user-defined functional and logic constraints. This key feature is provided for both standard and discrete diffusion models, enabling, for the first time, the generation of both continuous (e.g., images and trajectories) and discrete (e.g., molecular structures and natural language) outputs that comply with constraints. This ability is demonstrated on tasks spanning three key challenges: (1) Safety, in the context of non-toxic molecular generation and collision-free trajectory optimization; (2) Data scarcity, in domains such as drug discovery and materials engineering; and (3) Out-of-domain generalization, where enforcing symbolic constraints allows adaptation beyond the training distribution. 

**Abstract (ZH)**: 尽管扩散模型具有显著的生成能力，但将其整合到安全关键或科学严谨的应用中依然受限于确保满足严格的物理、结构和操作约束的需求。为应对这一挑战，本文引入了神经符号扩散（NSD）框架，该框架将扩散步骤与符号优化交错进行，使用户能够在用户定义的函数和逻辑约束下生成认证一致的样本。这一关键特性适用于标准和离散的扩散模型，首次实现了在满足约束条件下的连续输出（例如，图像和轨迹）和离散输出（例如，分子结构和自然语言）的生成。该能力在三个关键挑战任务中得以展示：（1）安全性，特别是在非毒性分子生成和无障碍轨迹优化的背景下；（2）数据稀缺性，特别是在药物发现和材料工程等领域；（3）域外泛化，通过对符号约束的强制执行实现训练分布之外的适应。 

---
# anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding 

**Title (ZH)**: anyECG-chat：一种灵活心电图输入与多任务理解的通用ECG-MLLM 

**Authors**: Haitao Li, Ziyu Li, Yiheng Mao, Ziyi Liu, Zhoujian Sun, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00942)  

**Abstract**: The advent of multimodal large language models (MLLMs) has sparked interest in their application to electrocardiogram (ECG) analysis. However, existing ECG-focused MLLMs primarily focus on report generation tasks, often limited to single 12-lead, short-duration (10s) ECG inputs, thereby underutilizing the potential of MLLMs. To this end, we aim to develop a MLLM for ECG analysis that supports a broader range of tasks and more flexible ECG inputs. However, existing ECG-QA datasets are often monotonous. To address this gap, we first constructed the anyECG dataset, which encompasses a wide variety of tasks, including report generation, abnormal waveform localization, and open-ended question answering. In addition to standard hospital ECGs, we introduced long-duration reduced-lead ECGs for home environments and multiple ECG comparison scenarios commonly encountered in clinical practice. Furthermore, we propose the anyECG-chat model, which supports dynamic-length ECG inputs and multiple ECG inputs. We trained the model using a three-stage curriculum training recipe with the anyECG dataset. A comprehensive evaluation was conducted, demonstrating that anyECG-chat is capable of supporting various practical application scenarios, including not only common report generation tasks but also abnormal waveform localization for long-duration reduced-lead ECGs in home environments and comprehensive comparative analysis of multiple ECGs. 

**Abstract (ZH)**: 多模态大型语言模型在心电图分析中的应用：anyECG-chat模型的设计与实现 

---
# Action Dependency Graphs for Globally Optimal Coordinated Reinforcement Learning 

**Title (ZH)**: 全局最优协调强化学习的动作依赖图 

**Authors**: Jianglin Ding, Jingcheng Tang, Gangshan Jing  

**Link**: [PDF](https://arxiv.org/pdf/2506.00797)  

**Abstract**: Action-dependent individual policies, which incorporate both environmental states and the actions of other agents in decision-making, have emerged as a promising paradigm for achieving global optimality in multi-agent reinforcement learning (MARL). However, the existing literature often adopts auto-regressive action-dependent policies, where each agent's policy depends on the actions of all preceding agents. This formulation incurs substantial computational complexity as the number of agents increases, thereby limiting scalability. In this work, we consider a more generalized class of action-dependent policies, which do not necessarily follow the auto-regressive form. We propose to use the `action dependency graph (ADG)' to model the inter-agent action dependencies. Within the context of MARL problems structured by coordination graphs, we prove that an action-dependent policy with a sparse ADG can achieve global optimality, provided the ADG satisfies specific conditions specified by the coordination graph. Building on this theoretical foundation, we develop a tabular policy iteration algorithm with guaranteed global optimality. Furthermore, we integrate our framework into several SOTA algorithms and conduct experiments in complex environments. The empirical results affirm the robustness and applicability of our approach in more general scenarios, underscoring its potential for broader MARL challenges. 

**Abstract (ZH)**: 基于动作依赖的个体策略：一种在多智能体强化学习中的全局最优实现新范式 

---
# Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn 

**Title (ZH)**: 通过减少 churn 遏制持续强化学习中塑性损失 

**Authors**: Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2506.00592)  

**Abstract**: Plasticity, or the ability of an agent to adapt to new tasks, environments, or distributions, is crucial for continual learning. In this paper, we study the loss of plasticity in deep continual RL from the lens of churn: network output variability for out-of-batch data induced by mini-batch training. We demonstrate that (1) the loss of plasticity is accompanied by the exacerbation of churn due to the gradual rank decrease of the Neural Tangent Kernel (NTK) matrix; (2) reducing churn helps prevent rank collapse and adjusts the step size of regular RL gradients adaptively. Moreover, we introduce Continual Churn Approximated Reduction (C-CHAIN) and demonstrate it improves learning performance and outperforms baselines in a diverse range of continual learning environments on OpenAI Gym Control, ProcGen, DeepMind Control Suite, and MinAtar benchmarks. 

**Abstract (ZH)**: 连续学习中由于 minibatch 训练引起的 churn 加剧导致的可塑性丧失及其缓解方法 

---
# Understanding Behavioral Metric Learning: A Large-Scale Study on Distracting Reinforcement Learning Environments 

**Title (ZH)**: 理解行为度量学习：对具有分散注意力的强化学习环境的大规模研究 

**Authors**: Ziyan Luo, Tianwei Ni, Pierre-Luc Bacon, Doina Precup, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2506.00563)  

**Abstract**: A key approach to state abstraction is approximating behavioral metrics (notably, bisimulation metrics) in the observation space and embedding these learned distances in the representation space. While promising for robustness to task-irrelevant noise, as shown in prior work, accurately estimating these metrics remains challenging, requiring various design choices that create gaps between theory and practice. Prior evaluations focus mainly on final returns, leaving the quality of learned metrics and the source of performance gains unclear. To systematically assess how metric learning works in deep reinforcement learning (RL), we evaluate five recent approaches, unified conceptually as isometric embeddings with varying design choices. We benchmark them with baselines across 20 state-based and 14 pixel-based tasks, spanning 370 task configurations with diverse noise settings. Beyond final returns, we introduce the evaluation of a denoising factor to quantify the encoder's ability to filter distractions. To further isolate the effect of metric learning, we propose and evaluate an isolated metric estimation setting, in which the encoder is influenced solely by the metric loss. Finally, we release an open-source, modular codebase to improve reproducibility and support future research on metric learning in deep RL. 

**Abstract (ZH)**: 一种关键的态抽象方法是通过观测空间近似行为度量（尤其是拟态度量），并在表示空间中嵌入这些学习到的距离。尽管这种方法对无关任务噪声具有鲁棒性，前人研究已经证明，准确估计这些度量仍然具有挑战性，需要各种设计选择以弥合理论与实践之间的差距。先前的评估主要集中在最终回报上，使得所学习度量的质量和性能提升的原因不清晰。为了系统地评估度量学习在深度强化学习（RL）中的作用，我们评估了五种最近的方法，这些方法从概念上统一为具有不同设计选择的等距嵌入。我们在20个基于态和14个基于像素的任务上与基线进行基准测试，涵盖了370种具有不同噪声设置的任务配置。除了最终回报，我们引入了去噪因子的评估来定量衡量编码器过滤干扰的能力。为了进一步隔离度量学习的影响，我们提出并评估了一种孤立的度量估计设置，在这种设置中，编码器仅受到度量损失的影响。最后，我们发布了一个开源模块化代码库，以提高可重复性，并支持未来在深度RL中进行度量学习的研究。 

---
# MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning 

**Title (ZH)**: MMedAgent-RL: 优化多模态医疗推理中的多Agent协作 

**Authors**: Peng Xia, Jinglu Wang, Yibo Peng, Kaide Zeng, Xian Wu, Xiangru Tang, Hongtu Zhu, Yun Li, Shujie Liu, Yan Lu, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00555)  

**Abstract**: Medical Large Vision-Language Models (Med-LVLMs) have shown strong potential in multimodal diagnostic tasks. However, existing single-agent models struggle to generalize across diverse medical specialties, limiting their performance. Recent efforts introduce multi-agent collaboration frameworks inspired by clinical workflows, where general practitioners (GPs) and specialists interact in a fixed sequence. Despite improvements, these static pipelines lack flexibility and adaptability in reasoning. To address this, we propose MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that enables dynamic, optimized collaboration among medical agents. Specifically, we train two GP agents based on Qwen2.5-VL via RL: the triage doctor learns to assign patients to appropriate specialties, while the attending physician integrates the judgments from multi-specialists and its own knowledge to make final decisions. To address the inconsistency in specialist outputs, we introduce a curriculum learning (CL)-guided RL strategy that progressively teaches the attending physician to balance between imitating specialists and correcting their mistakes. Experiments on five medical VQA benchmarks demonstrate that MMedAgent-RL not only outperforms both open-source and proprietary Med-LVLMs, but also exhibits human-like reasoning patterns. Notably, it achieves an average performance gain of 18.4% over supervised fine-tuning baselines. 

**Abstract (ZH)**: 基于强化学习的多Agent动态协作医疗模型（MMedAgent-RL）：在多模态诊断任务中的应用 

---
# Reinforcement Learning for Hanabi 

**Title (ZH)**: 汉诺伊纸牌游戏的强化学习方法 

**Authors**: Nina Cohen, Kordel K. France  

**Link**: [PDF](https://arxiv.org/pdf/2506.00458)  

**Abstract**: Hanabi has become a popular game for research when it comes to reinforcement learning (RL) as it is one of the few cooperative card games where you have incomplete knowledge of the entire environment, thus presenting a challenge for a RL agent. We explored different tabular and deep reinforcement learning algorithms to see which had the best performance both against an agent of the same type and also against other types of agents. We establish that certain agents played their highest scoring games against specific agents while others exhibited higher scores on average by adapting to the opposing agent's behavior. We attempted to quantify the conditions under which each algorithm provides the best advantage and identified the most interesting interactions between agents of different types. In the end, we found that temporal difference (TD) algorithms had better overall performance and balancing of play types compared to tabular agents. Specifically, tabular Expected SARSA and deep Q-Learning agents showed the best performance. 

**Abstract (ZH)**: 汉诺伊纸牌游戏已成为强化学习研究中的一个流行研究对象，因为它是一个少数具有整个环境部分信息的协作纸牌游戏，从而为强化学习代理带来挑战。我们探索了不同的表式和深度强化学习算法，以确定哪种算法在同类型代理和不同类型代理之间都能表现出最佳性能。我们发现某些代理在与特定类型的代理对弈时表现最佳，而其他代理则通过适应对手代理的行为实现了更高的平均评分。我们试图量化每种算法提供最佳优势的条件，并确定不同类型代理之间最有趣的交互。最终，我们发现时差（Temporal Difference，TD）算法在整体性能和不同类型代理间的均衡方面优于表式代理。具体来说，表式Expected SARSA代理和深度Q学习代理表现出最佳性能。 

---
# A Reinforcement Learning-Based Telematic Routing Protocol for the Internet of Underwater Things 

**Title (ZH)**: 基于强化学习的物联网水下节点路由协议 

**Authors**: Mohammadhossein Homaei, Mehran Tarif, Agustin Di Bartolo, Oscar Mogollon Gutierrez, Mar Avila  

**Link**: [PDF](https://arxiv.org/pdf/2506.00133)  

**Abstract**: The Internet of Underwater Things (IoUT) faces major challenges such as low bandwidth, high latency, mobility, and limited energy resources. Traditional routing protocols like RPL, which were designed for land-based networks, do not perform well in these underwater conditions. This paper introduces RL-RPL-UA, a new routing protocol that uses reinforcement learning to improve performance in underwater environments. Each node includes a lightweight RL agent that selects the best parent node based on local information such as packet delivery ratio, buffer level, link quality, and remaining energy. RL-RPL-UA keeps full compatibility with standard RPL messages and adds a dynamic objective function to support real-time decision-making. Simulations using Aqua-Sim show that RL-RPL-UA increases packet delivery by up to 9.2%, reduces energy use per packet by 14.8%, and extends network lifetime by 80 seconds compared to traditional methods. These results suggest that RL-RPL-UA is a promising and energy-efficient routing solution for underwater networks. 

**Abstract (ZH)**: 水下事物流动的互联网（IoUT）面临低带宽、高延迟、移动性和有限的能量资源等重大挑战。传统的路由协议如RPL，适用于陆地网络，在水下环境中表现不佳。本文介绍了一种新的基于强化学习的路由协议RL-RPL-UA，以提高水下环境中的性能。每个节点包含一个轻量级的RL代理，基于诸如数据包投递率、缓冲区水平、链路质量和剩余能量等本地信息来选择最优父节点。RL-RPL-UA与标准RPL消息保持完全兼容，并增加了一个动态目标函数以支持实时决策。使用Aqua-Sim进行的仿真表明，与传统方法相比，RL-RPL-UA可将数据包投递率提高9.2%，每数据包能量使用减少14.8%，网络寿命延长80秒。这些结果表明，RL-RPL-UA是一种有前景的能量高效的水下网络路由解决方案。 

---
# Adapting Offline Reinforcement Learning with Online Delays 

**Title (ZH)**: 具在线延迟的离线强化学习适应 

**Authors**: Simon Sinong Zhan, Qingyuan Wu, Frank Yang, Xiangyu Shi, Chao Huang, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00131)  

**Abstract**: Offline-to-online deployment of reinforcement-learning (RL) agents must bridge two gaps: (1) the sim-to-real gap, where real systems add latency and other imperfections not present in simulation, and (2) the interaction gap, where policies trained purely offline face out-of-distribution states during online execution because gathering new interaction data is costly or risky. Agents therefore have to generalize from static, delay-free datasets to dynamic, delay-prone environments. Standard offline RL learns from delay-free logs yet must act under delays that break the Markov assumption and hurt performance. We introduce DT-CORL (Delay-Transformer belief policy Constrained Offline RL), an offline-RL framework built to cope with delayed dynamics at deployment. DT-CORL (i) produces delay-robust actions with a transformer-based belief predictor even though it never sees delayed observations during training, and (ii) is markedly more sample-efficient than naïve history-augmentation baselines. Experiments on D4RL benchmarks with several delay settings show that DT-CORL consistently outperforms both history-augmentation and vanilla belief-based methods, narrowing the sim-to-real latency gap while preserving data efficiency. 

**Abstract (ZH)**: Offline-to-online 部署中的延迟鲁棒 Offline RL 框架: DT-CORL 

---
# Reducing Latency in LLM-Based Natural Language Commands Processing for Robot Navigation 

**Title (ZH)**: 基于LLM的自然语言命令处理在机器人导航中减少延迟 

**Authors**: Diego Pollini, Bruna V. Guterres, Rodrigo S. Guerra, Ricardo B. Grando  

**Link**: [PDF](https://arxiv.org/pdf/2506.00075)  

**Abstract**: The integration of Large Language Models (LLMs), such as GPT, in industrial robotics enhances operational efficiency and human-robot collaboration. However, the computational complexity and size of these models often provide latency problems in request and response times. This study explores the integration of the ChatGPT natural language model with the Robot Operating System 2 (ROS 2) to mitigate interaction latency and improve robotic system control within a simulated Gazebo environment. We present an architecture that integrates these technologies without requiring a middleware transport platform, detailing how a simulated mobile robot responds to text and voice commands. Experimental results demonstrate that this integration improves execution speed, usability, and accessibility of the human-robot interaction by decreasing the communication latency by 7.01\% on average. Such improvements facilitate smoother, real-time robot operations, which are crucial for industrial automation and precision tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT在工业机器人中的集成增强了操作效率和人机协作，但这些模型的计算复杂度和大小 often 提供了请求和响应时间的延迟问题。本研究探索将ChatGPT自然语言模型与Robot Operating System 2（ROS 2）集成，以减轻互动延迟并改善在模拟Gazebo环境中的机器人系统控制。我们提出了一种architecture，无需中间件传输平台即可集成这些技术，并详细介绍了模拟移动机器人对文本和语音命令的响应方式。实验结果表明，这种集成通过平均减少7.01%的通信延迟，改善了人机交互的执行速度、可用性和便捷性。这些改进促进了更顺畅、实时的机器人操作，这对于工业自动化和精密任务至关重要。 

---
