# V-HOP: Visuo-Haptic 6D Object Pose Tracking 

**Title (ZH)**: V-HOP：视觉-触觉6D物体姿态跟踪 

**Authors**: Hongyu Li, Mingxi Jia, Tuluhan Akbulut, Yu Xiang, George Konidaris, Srinath Sridhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17434)  

**Abstract**: Humans naturally integrate vision and haptics for robust object perception during manipulation. The loss of either modality significantly degrades performance. Inspired by this multisensory integration, prior object pose estimation research has attempted to combine visual and haptic/tactile feedback. Although these works demonstrate improvements in controlled environments or synthetic datasets, they often underperform vision-only approaches in real-world settings due to poor generalization across diverse grippers, sensor layouts, or sim-to-real environments. Furthermore, they typically estimate the object pose for each frame independently, resulting in less coherent tracking over sequences in real-world deployments. To address these limitations, we introduce a novel unified haptic representation that effectively handles multiple gripper embodiments. Building on this representation, we introduce a new visuo-haptic transformer-based object pose tracker that seamlessly integrates visual and haptic input. We validate our framework in our dataset and the Feelsight dataset, demonstrating significant performance improvement on challenging sequences. Notably, our method achieves superior generalization and robustness across novel embodiments, objects, and sensor types (both taxel-based and vision-based tactile sensors). In real-world experiments, we demonstrate that our approach outperforms state-of-the-art visual trackers by a large margin. We further show that we can achieve precise manipulation tasks by incorporating our real-time object tracking result into motion plans, underscoring the advantages of visuo-haptic perception. Our model and dataset will be made open source upon acceptance of the paper. Project website: this https URL 

**Abstract (ZH)**: 人类自然地利用视觉和触觉进行稳定的目标感知。失去其中任何一种模态都会显著降低性能。受这种多感官整合的启发，先前的目标姿态估计研究尝试结合视觉和触觉/触觉反馈。尽管这些工作在受控环境或合成数据集中展示了改进，但在真实环境中，由于在不同夹持器、传感器布局或仿真实例之间的泛化能力较差，它们通常表现不如纯视觉方法。此外，它们通常独立估计每一帧的目标姿态，导致在真实部署中序列跟踪不够连贯。为了解决这些限制，我们提出了一种新型统一的触觉表示，能够有效处理多种夹持器实例。在此表示的基础上，我们引入了一种新的视觉-触觉变换器目标姿态跟踪器，无缝整合视觉和触觉输入。我们在自家数据集和Feelsight数据集中验证了该框架，证明在挑战性序列上的性能有了显著提高。值得注意的是，我们的方法在新颖实例、物体和传感器类型（包括像素触觉传感器和基于视觉的触觉传感器）上展现出更好的泛化能力和鲁棒性。在真实世界实验中，我们证明了该方法在姿态跟踪结果实时应用于运动计划时，比现有最佳视觉跟踪器有明显优势。此外，我们展示了基于我们的实时目标跟踪结果可以实现精确的操控任务，突显了视觉-触觉感知的优势。我们的模型和数据集将在论文被接受后开源。项目网站：this https URL 

---
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning 

**Title (ZH)**: FACTR：力参加递进训练在接触丰富的策略学习中 

**Authors**: Jason Jingzhou Liu, Yulong Li, Kenneth Shaw, Tony Tao, Ruslan Salakhutdinov, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2502.17432)  

**Abstract**: Many contact-rich tasks humans perform, such as box pickup or rolling dough, rely on force feedback for reliable execution. However, this force information, which is readily available in most robot arms, is not commonly used in teleoperation and policy learning. Consequently, robot behavior is often limited to quasi-static kinematic tasks that do not require intricate force-feedback. In this paper, we first present a low-cost, intuitive, bilateral teleoperation setup that relays external forces of the follower arm back to the teacher arm, facilitating data collection for complex, contact-rich tasks. We then introduce FACTR, a policy learning method that employs a curriculum which corrupts the visual input with decreasing intensity throughout training. The curriculum prevents our transformer-based policy from over-fitting to the visual input and guides the policy to properly attend to the force modality. We demonstrate that by fully utilizing the force information, our method significantly improves generalization to unseen objects by 43\% compared to baseline approaches without a curriculum. Video results and instructions at this https URL 

**Abstract (ZH)**: 许多人类执行的高接触任务，如捡拾盒子或擀面团，依赖于力反馈以确保可靠执行。然而，大多数机器人手臂都能轻松获得的这种力信息，在远程操作和策略学习中并未得到广泛应用。因此，机器人行为往往局限于不需要复杂力反馈的准静态运动任务。在本文中，我们首先提出了一种低成本、直观的双边远程操作设置，将跟随臂的外部力信息反馈到教师臂，从而便于收集复杂高接触任务的数据。我们随后介绍了FACTR，这是一种策略学习方法，采用了一种随训练进程逐渐降低视觉输入污染强度的课程化训练方法。该课程化训练方法防止基于变压器的策略过度拟合视觉输入，并引导策略正确关注力模态。实验结果表明，通过充分利用力信息，我们的方法相比无课程的基础方法，显著提高了对未见过的对象的泛化能力，提高了43%。视频结果和指南请参见此链接：[此处链接] 

---
# SoFFT: Spatial Fourier Transform for Modeling Continuum Soft Robots 

**Title (ZH)**: SoFFT：空间傅里叶变换在建模连续软机器人中的应用 

**Authors**: Daniele Caradonna, Diego Bianchi, Franco Angelini, Egidio Falotico  

**Link**: [PDF](https://arxiv.org/pdf/2502.17347)  

**Abstract**: Continuum soft robots, composed of flexible materials, exhibit theoretically infinite degrees of freedom, enabling notable adaptability in unstructured environments. Cosserat Rod Theory has emerged as a prominent framework for modeling these robots efficiently, representing continuum soft robots as time-varying curves, known as backbones. In this work, we propose viewing the robot's backbone as a signal in space and time, applying the Fourier transform to describe its deformation compactly. This approach unifies existing modeling strategies within the Cosserat Rod Theory framework, offering insights into commonly used heuristic methods. Moreover, the Fourier transform enables the development of a data-driven methodology to experimentally capture the robot's deformation. The proposed approach is validated through numerical simulations and experiments on a real-world prototype, demonstrating a reduction in the degrees of freedom while preserving the accuracy of the deformation representation. 

**Abstract (ZH)**: 连续软机器人，由柔性材料组成，理论上具有无限的自由度，使其在未经结构化环境中展现出显著的适应性。柯谢尔杆理论已成为高效建模这些机器人的主要框架，将连续软机器人表示为时间变化的曲线，即主干。在本文中，我们提出将机器人的主干视为时空中的信号，应用傅里叶变换以紧凑的形式描述其变形。该方法统一了柯谢尔杆理论框架内的现有建模策略，提供了对常用启发式方法的见解。此外，傅里叶变换使开发一种数据驱动的方法以实验性地捕捉机器人的变形成为可能。所提出的方法通过数值仿真和真实原型的实验得到了验证，证明了在保持变形表示准确性的同时减少了自由度。 

---
# TDMPBC: Self-Imitative Reinforcement Learning for Humanoid Robot Control 

**Title (ZH)**: TDMPBC：类自我模仿强化学习在类人机器人控制中的应用 

**Authors**: Zifeng Zhuang, Diyuan Shi, Runze Suo, Xiao He, Hongyin Zhang, Ting Wang, Shangke Lyu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17322)  

**Abstract**: Complex high-dimensional spaces with high Degree-of-Freedom and complicated action spaces, such as humanoid robots equipped with dexterous hands, pose significant challenges for reinforcement learning (RL) algorithms, which need to wisely balance exploration and exploitation under limited sample budgets. In general, feasible regions for accomplishing tasks within complex high-dimensional spaces are exceedingly narrow. For instance, in the context of humanoid robot motion control, the vast majority of space corresponds to falling, while only a minuscule fraction corresponds to standing upright, which is conducive to the completion of downstream tasks. Once the robot explores into a potentially task-relevant region, it should place greater emphasis on the data within that region. Building on this insight, we propose the $\textbf{S}$elf-$\textbf{I}$mitative $\textbf{R}$einforcement $\textbf{L}$earning ($\textbf{SIRL}$) framework, where the RL algorithm also imitates potentially task-relevant trajectories. Specifically, trajectory return is utilized to determine its relevance to the task and an additional behavior cloning is adopted whose weight is dynamically adjusted based on the trajectory return. As a result, our proposed algorithm achieves 120% performance improvement on the challenging HumanoidBench with 5% extra computation overhead. With further visualization, we find the significant performance gain does lead to meaningful behavior improvement that several tasks are solved successfully. 

**Abstract (ZH)**: 具有高自由度和复杂动作空间的复杂高维空间，如配备灵巧手的人形机器人，对强化学习算法提出了重大挑战，这些算法需要在有限的样本预算下巧妙地平衡探索与利用。通常，在复杂高维空间中执行任务的可行区域极其狭窄。例如，在人形机器人动作控制的背景下，大部分空间对应于跌倒状态，而仅有一小部分对应于站立状态，这有利于下游任务的完成。一旦机器人探索到可能与任务相关的位置，应更加重视该区域的数据。基于这一认识，我们提出了自模仿强化学习（Self-Imitative Reinforcement Learning, SIRL）框架，其中强化学习算法也模仿可能与任务相关的轨迹。具体而言，轨迹回访用于确定其与任务的相关性，并采用附加的行为克隆，其权重根据轨迹回访动态调整。因此，我们提出的算法在具有5%额外计算开销的情况下，在具有挑战性的HumanoidBench上实现了120%的性能提升。进一步可视化表明，性能的显著提升确实导致了有意义的行为改进，使得多个任务得以成功解决。 

---
# Continuous Wrist Control on the Hannes Prosthesis: a Vision-based Shared Autonomy Framework 

**Title (ZH)**: 基于视觉的共享自主框架下汉内斯假肢的连续腕部控制 

**Authors**: Federico Vasile, Elisa Maiettini, Giulia Pasquale, Nicolò Boccardo, Lorenzo Natale  

**Link**: [PDF](https://arxiv.org/pdf/2502.17265)  

**Abstract**: Most control techniques for prosthetic grasping focus on dexterous fingers control, but overlook the wrist motion. This forces the user to perform compensatory movements with the elbow, shoulder and hip to adapt the wrist for grasping. We propose a computer vision-based system that leverages the collaboration between the user and an automatic system in a shared autonomy framework, to perform continuous control of the wrist degrees of freedom in a prosthetic arm, promoting a more natural approach-to-grasp motion. Our pipeline allows to seamlessly control the prosthetic wrist to follow the target object and finally orient it for grasping according to the user intent. We assess the effectiveness of each system component through quantitative analysis and finally deploy our method on the Hannes prosthetic arm. Code and videos: this https URL. 

**Abstract (ZH)**: 基于计算机视觉的共享自主系统在假肢手臂中连续控制手腕自由度，促进自然接近抓取运动 

---
# A Reinforcement Learning Approach to Non-prehensile Manipulation through Sliding 

**Title (ZH)**: 通过滑动进行非抓握式操作的强化学习方法 

**Authors**: Hamidreza Raei, Elena De Momi, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2502.17221)  

**Abstract**: Although robotic applications increasingly demand versatile and dynamic object handling, most existing techniques are predominantly focused on grasp-based manipulation, limiting their applicability in non-prehensile tasks. To address this need, this study introduces a Deep Deterministic Policy Gradient (DDPG) reinforcement learning framework for efficient non-prehensile manipulation, specifically for sliding an object on a surface. The algorithm generates a linear trajectory by precisely controlling the acceleration of a robotic arm rigidly coupled to the horizontal surface, enabling the relative manipulation of an object as it slides on top of the surface. Furthermore, two distinct algorithms have been developed to estimate the frictional forces dynamically during the sliding process. These algorithms provide online friction estimates after each action, which are fed back into the actor model as critical feedback after each action. This feedback mechanism enhances the policy's adaptability and robustness, ensuring more precise control of the platform's acceleration in response to varying surface condition. The proposed algorithm is validated through simulations and real-world experiments. Results demonstrate that the proposed framework effectively generalizes sliding manipulation across varying distances and, more importantly, adapts to different surfaces with diverse frictional properties. Notably, the trained model exhibits zero-shot sim-to-real transfer capabilities. 

**Abstract (ZH)**: 虽然机器人应用日益需要灵活多样的物体处理能力，但现有的大多数技术主要集中在基于抓取的操作上，限制了其在非抓取任务中的应用。为解决这一问题，本研究引入了一种基于Deep Deterministic Policy Gradient (DDPG)的强化学习框架，以高效实现物体在表面的滑动操作。该算法通过精确控制与水平面刚性耦合的机器人臂的加速度，生成线性轨迹，从而实现物体在表面滑动过程中的相对操作。此外，还开发了两种算法以动态估计滑动过程中的摩擦力。这些算法在每次动作后提供即时摩擦力估计，并将其作为关键反馈输入到演员模型中，以增强策略的适应性和鲁棒性，确保在不同表面条件下对平台加速度的更精确控制。所提出的方法通过仿真和实际实验进行了验证。结果表明，该框架能够有效地在不同距离上实现滑动操作，并且更重要的是，能够适应具有不同摩擦性质的各种表面。值得注意的是，训练后的模型展示了零样本仿真实验到实际应用的转移能力。 

---
# Humanoid Whole-Body Locomotion on Narrow Terrain via Dynamic Balance and Reinforcement Learning 

**Title (ZH)**: 基于动态平衡和强化学习的 humanoid 全身窄地形行走 

**Authors**: Weiji Xie, Chenjia Bai, Jiyuan Shi, Junkai Yang, Yunfei Ge, Weinan Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17219)  

**Abstract**: Humans possess delicate dynamic balance mechanisms that enable them to maintain stability across diverse terrains and under extreme conditions. However, despite significant advances recently, existing locomotion algorithms for humanoid robots are still struggle to traverse extreme environments, especially in cases that lack external perception (e.g., vision or LiDAR). This is because current methods often rely on gait-based or perception-condition rewards, lacking effective mechanisms to handle unobservable obstacles and sudden balance loss. To address this challenge, we propose a novel whole-body locomotion algorithm based on dynamic balance and Reinforcement Learning (RL) that enables humanoid robots to traverse extreme terrains, particularly narrow pathways and unexpected obstacles, using only proprioception. Specifically, we introduce a dynamic balance mechanism by leveraging an extended measure of Zero-Moment Point (ZMP)-driven rewards and task-driven rewards in a whole-body actor-critic framework, aiming to achieve coordinated actions of the upper and lower limbs for robust locomotion. Experiments conducted on a full-sized Unitree H1-2 robot verify the ability of our method to maintain balance on extremely narrow terrains and under external disturbances, demonstrating its effectiveness in enhancing the robot's adaptability to complex environments. The videos are given at this https URL. 

**Abstract (ZH)**: 人类拥有精细的动态平衡机制，能够在多种地形和极端条件下保持稳定性。尽管最近取得了显著进展，现有类人机器人运动算法在穿越极端环境时仍然存在困难，特别是在缺乏外部感知（例如视觉或LiDAR）的情况下。这是因为当前的方法通常依赖于步态导向或感知导向的奖励机制，缺乏有效处理未观察到的障碍和突发失衡的机制。为应对这一挑战，我们提出了一种基于动态平衡和强化学习（RL）的全新全身运动算法，使类人机器人能够在仅依靠本体感觉的情况下穿越极端地形，特别是狭窄路径和意外障碍物。具体来说，我们通过利用扩展的零力矩点（ZMP）驱动奖励和任务驱动奖励，在全身演员-评论家框架中引入了一种动态平衡机制，旨在实现上肢和下肢的协调动作，以实现稳健的运动。在全尺寸Unitree H1-2机器人上的实验验证了我们方法在极端狭窄地形和外部干扰下保持平衡的能力，证明了其在提高机器人适应复杂环境方面的有效性。相关视频见此网址：https://xxxxx。 

---
# Evolution 6.0: Evolving Robotic Capabilities Through Generative Design 

**Title (ZH)**: Evolution 6.0: 通过生成设计进化的机器人能力 

**Authors**: Muhammad Haris Khan, Artyom Myshlyaev, Artyom Lykov, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2502.17034)  

**Abstract**: We propose a new concept, Evolution 6.0, which represents the evolution of robotics driven by Generative AI. When a robot lacks the necessary tools to accomplish a task requested by a human, it autonomously designs the required instruments and learns how to use them to achieve the goal. Evolution 6.0 is an autonomous robotic system powered by Vision-Language Models (VLMs), Vision-Language Action (VLA) models, and Text-to-3D generative models for tool design and task execution. The system comprises two key modules: the Tool Generation Module, which fabricates task-specific tools from visual and textual data, and the Action Generation Module, which converts natural language instructions into robotic actions. It integrates QwenVLM for environmental understanding, OpenVLA for task execution, and Llama-Mesh for 3D tool generation. Evaluation results demonstrate a 90% success rate for tool generation with a 10-second inference time, and action generation achieving 83.5% in physical and visual generalization, 70% in motion generalization, and 37% in semantic generalization. Future improvements will focus on bimanual manipulation, expanded task capabilities, and enhanced environmental interpretation to improve real-world adaptability. 

**Abstract (ZH)**: 我们提出了一种新的概念——Evolution 6.0，代表了由生成式AI驱动的机器人进化。当机器人缺乏完成人类请求的任务所需的工具时，它能够自主设计所需的工具并学习如何使用这些工具来实现目标。Evolution 6.0 是一个由视觉-语言模型（VLMs）、视觉-语言-行动（VLA）模型和文本到3D生成模型驱动的自主机器人系统，用于工具设计和任务执行。该系统包括两个关键模块：工具生成模块，从视觉和文本数据生成任务特定的工具，以及动作生成模块，将自然语言指令转换为机器人动作。该系统集成了QwenVLM进行环境理解、OpenVLA进行任务执行以及Llama-Mesh进行3D工具生成。评估结果显示，在10秒的推理时间内，工具生成成功率为90%，动作生成在物理和视觉泛化方面的得分为83.5%，在运动泛化方面的得分为70%，在语义泛化方面的得分为37%。未来改进将专注于双臂操作、扩展的任务能力以及增强的环境解释，以提高其实用性。 

---
# Gazing at Failure: Investigating Human Gaze in Response to Robot Failure in Collaborative Tasks 

**Title (ZH)**: 注视失败：探究协作任务中人类对机器人失败的眼球运动响应 

**Authors**: Ramtin Tabatabaei, Vassilis Kostakos, Wafa Johal  

**Link**: [PDF](https://arxiv.org/pdf/2502.16899)  

**Abstract**: Robots are prone to making errors, which can negatively impact their credibility as teammates during collaborative tasks with human users. Detecting and recovering from these failures is crucial for maintaining effective level of trust from users. However, robots may fail without being aware of it. One way to detect such failures could be by analysing humans' non-verbal behaviours and reactions to failures. This study investigates how human gaze dynamics can signal a robot's failure and examines how different types of failures affect people's perception of robot. We conducted a user study with 27 participants collaborating with a robotic mobile manipulator to solve tangram puzzles. The robot was programmed to experience two types of failures -- executional and decisional -- occurring either at the beginning or end of the task, with or without acknowledgement of the failure. Our findings reveal that the type and timing of the robot's failure significantly affect participants' gaze behaviour and perception of the robot. Specifically, executional failures led to more gaze shifts and increased focus on the robot, while decisional failures resulted in lower entropy in gaze transitions among areas of interest, particularly when the failure occurred at the end of the task. These results highlight that gaze can serve as a reliable indicator of robot failures and their types, and could also be used to predict the appropriate recovery actions. 

**Abstract (ZH)**: 机器人在协作任务中可能会犯错误，这可能会影响它们作为队友的可信度。检测并从这些失败中恢复对于维持用户的信任至关重要。然而，机器人可能在不知情的情况下犯错。通过分析人类的非 verbal 行为和对失败的反应，可以检测此类失败。本研究调查了人类视线动态如何信号化机器人的失败，并分析了不同类型失败对人们感知机器人有何影响。我们进行了一个用户研究，共有27名参与者与一个机器人移动 manipulator 合作解决七巧板谜题。机器人被编程在同一任务的开始或结束时经历两种类型的失败——执行失败和决策失败，并且失败是否被承认有所不同。研究发现，机器人失败的类型和时间显著影响参与者的眼球运动行为和对机器人的感知。具体而言，执行失败导致更多的眼球转移并增加了对机器人的关注，而决策失败则在任务结束时导致对感兴趣区域的眼球转移熵降低。这些结果表明，视线可以作为机器人力竭及其类型的可靠指标，并且也可以用来预测合适的恢复动作。 

---
# Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation 

**Title (ZH)**: 反射规划：多阶段长期 horizon 机器人操作的视觉-语言模型 

**Authors**: Yunhai Feng, Jiaming Han, Zhuoran Yang, Xiangyu Yue, Sergey Levine, Jianlan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.16707)  

**Abstract**: Solving complex long-horizon robotic manipulation problems requires sophisticated high-level planning capabilities, the ability to reason about the physical world, and reactively choose appropriate motor skills. Vision-language models (VLMs) pretrained on Internet data could in principle offer a framework for tackling such problems. However, in their current form, VLMs lack both the nuanced understanding of intricate physics required for robotic manipulation and the ability to reason over long horizons to address error compounding issues. In this paper, we introduce a novel test-time computation framework that enhances VLMs' physical reasoning capabilities for multi-stage manipulation tasks. At its core, our approach iteratively improves a pretrained VLM with a "reflection" mechanism - it uses a generative model to imagine future world states, leverages these predictions to guide action selection, and critically reflects on potential suboptimalities to refine its reasoning. Experimental results demonstrate that our method significantly outperforms several state-of-the-art commercial VLMs as well as other post-training approaches such as Monte Carlo Tree Search (MCTS). Videos are available at this https URL. 

**Abstract (ZH)**: 解决复杂的长时Horizon机器人操作问题需要高级规划能力、对物理世界的推理能力以及反应性地选择合适的运动技能。互联网数据预训练的视觉-语言模型（VLMs）原则上可以提供解决此类问题的框架。然而，当前形式的VLMs缺乏对于机器人操作所需的细致物理理解以及处理长期推理和错误累积问题的能力。在本文中，我们引入了一种新的测试时计算框架，以增强VLMs在多阶段操作任务中的物理推理能力。该方法的核心在于迭代地改进预训练的VLM，并通过“反思”机制使用生成模型来想象未来的世界状态，利用这些预测来指导行动选择，并对潜在的不足进行批判性反思以改进其推理。实验结果表明，我们的方法明显优于几种最新的商用VLMs以及包括蒙特卡罗树搜索（MCTS）在内的其他后训练方法。视频可在以下链接查看：this https URL。 

---
# Human2Robot: Learning Robot Actions from Paired Human-Robot Videos 

**Title (ZH)**: 人类2机器人：从配对的人机视频中学习机器人动作 

**Authors**: Sicheng Xie, Haidong Cao, Zejia Weng, Zhen Xing, Shiwei Shen, Jiaqi Leng, Xipeng Qiu, Yanwei Fu, Zuxuan Wu, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16587)  

**Abstract**: Distilling knowledge from human demonstrations is a promising way for robots to learn and act. Existing work often overlooks the differences between humans and robots, producing unsatisfactory results. In this paper, we study how perfectly aligned human-robot pairs benefit robot learning. Capitalizing on VR-based teleportation, we introduce H\&R, a third-person dataset with 2,600 episodes, each of which captures the fine-grained correspondence between human hands and robot gripper. Inspired by the recent success of diffusion models, we introduce Human2Robot, an end-to-end diffusion framework that formulates learning from human demonstrates as a generative task. Human2Robot fully explores temporal dynamics in human videos to generate robot videos and predict actions at the same time. Through comprehensive evaluations of 8 seen, changed and unseen tasks in real-world settings, we demonstrate that Human2Robot can not only generate high-quality robot videos but also excel in seen tasks and generalize to unseen objects, backgrounds and even new tasks effortlessly. 

**Abstract (ZH)**: 从人类示范中提炼知识是机器人学习和行动的一种有前景的方法。现有工作往往忽视了人类与机器人之间的差异，导致结果不尽如人意。在这项工作中，我们研究完美对齐的人机配对如何促进机器人学习。借助基于VR的 teleportation，我们引入了H&R，一个包含2600个样本的第三人称数据集，每个样本都捕捉了人类手部与机器人夹爪之间的精细对应关系。受近期扩散模型成功的影响，我们引入了Human2Robot，一个端到端的扩散框架，将从人类示范中学习形式化为生成任务。Human2Robot全面探索了人类视频中的时序动态，同时生成机器人视频并预测动作。通过在真实世界设置中对8个已见、变更和未见任务的全面评估，我们证明Human2Robot不仅能生成高质量的机器人视频，还能在已见任务中表现出色，并轻松地泛化到未见对象、背景乃至新任务。 

---
# Efficient Coordination and Synchronization of Multi-Robot Systems Under Recurring Linear Temporal Logic 

**Title (ZH)**: 高效的多机器人系统在反复出现的线性时间逻辑下的协调与同步 

**Authors**: Davide Peron, Victor Nan Fernandez-Ayala, Eleftherios E. Vlahakis, Dimos V. Dimarogonas  

**Link**: [PDF](https://arxiv.org/pdf/2502.16531)  

**Abstract**: We consider multi-robot systems under recurring tasks formalized as linear temporal logic (LTL) specifications. To solve the planning problem efficiently, we propose a bottom-up approach combining offline plan synthesis with online coordination, dynamically adjusting plans via real-time communication. To address action delays, we introduce a synchronization mechanism ensuring coordinated task execution, leading to a multi-agent coordination and synchronization framework that is adaptable to a wide range of multi-robot applications. The software package is developed in Python and ROS2 for broad deployment. We validate our findings through lab experiments involving nine robots showing enhanced adaptability compared to previous methods. Additionally, we conduct simulations with up to ninety agents to demonstrate the reduced computational complexity and the scalability features of our work. 

**Abstract (ZH)**: 我们考虑将重复任务形式化为线性时序逻辑（LTL）规范的多机器人系统。为了高效解决规划问题，我们提出了一种自底向上的方法，结合离线计划合成与在线协调，通过实时通信动态调整计划。为了解决动作延迟问题，我们引入了一种同步机制以确保任务的协调执行，从而形成一个适用于多种多机器人应用的多个代理的协调与同步框架，该框架具有很高的适应性。该软件包使用Python和ROS2开发，以实现广泛部署。我们通过涉及九个机器人的真实实验室实验验证了我们的发现，结果显示了与之前方法相比的增强的适应性。此外，我们进行了多达九十个代理的模拟实验，以证明我们工作的计算复杂度降低和扩展功能。 

---
# Path Planning using Instruction-Guided Probabilistic Roadmaps 

**Title (ZH)**: 基于指令引导的概率路网的路径规划 

**Authors**: Jiaqi Bao, Ryo Yonetani  

**Link**: [PDF](https://arxiv.org/pdf/2502.16515)  

**Abstract**: This work presents a novel data-driven path planning algorithm named Instruction-Guided Probabilistic Roadmap (IG-PRM). Despite the recent development and widespread use of mobile robot navigation, the safe and effective travels of mobile robots still require significant engineering effort to take into account the constraints of robots and their tasks. With IG-PRM, we aim to address this problem by allowing robot operators to specify such constraints through natural language instructions, such as ``aim for wider paths'' or ``mind small gaps''. The key idea is to convert such instructions into embedding vectors using large-language models (LLMs) and use the vectors as a condition to predict instruction-guided cost maps from occupancy maps. By constructing a roadmap based on the predicted costs, we can find instruction-guided paths via the standard shortest path search. Experimental results demonstrate the effectiveness of our approach on both synthetic and real-world indoor navigation environments. 

**Abstract (ZH)**: 基于指令引导的概率路网的新型数据驱动路径规划算法 

---
# AnyDexGrasp: General Dexterous Grasping for Different Hands with Human-level Learning Efficiency 

**Title (ZH)**: AnyDexGrasp: 不同手型的高效人类级灵巧抓取 

**Authors**: Hao-Shu Fang, Hengxu Yan, Zhenyu Tang, Hongjie Fang, Chenxi Wang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16420)  

**Abstract**: We introduce an efficient approach for learning dexterous grasping with minimal data, advancing robotic manipulation capabilities across different robotic hands. Unlike traditional methods that require millions of grasp labels for each robotic hand, our method achieves high performance with human-level learning efficiency: only hundreds of grasp attempts on 40 training objects. The approach separates the grasping process into two stages: first, a universal model maps scene geometry to intermediate contact-centric grasp representations, independent of specific robotic hands. Next, a unique grasp decision model is trained for each robotic hand through real-world trial and error, translating these representations into final grasp poses. Our results show a grasp success rate of 75-95\% across three different robotic hands in real-world cluttered environments with over 150 novel objects, improving to 80-98\% with increased training objects. This adaptable method demonstrates promising applications for humanoid robots, prosthetics, and other domains requiring robust, versatile robotic manipulation. 

**Abstract (ZH)**: 一种高效的学习少数据 Dexterous Grasping 的方法：跨不同机械手提升机器人操作能力 

---
# Quadruped Robot Simulation Using Deep Reinforcement Learning -- A step towards locomotion policy 

**Title (ZH)**: 使用深度强化学习的四足机器人模拟——朝向运动策略的一步 

**Authors**: Nabeel Ahmad Khan Jadoon, Mongkol Ekpanyapong  

**Link**: [PDF](https://arxiv.org/pdf/2502.16401)  

**Abstract**: We present a novel reinforcement learning method to train the quadruped robot in a simulated environment. The idea of controlling quadruped robots in a dynamic environment is quite challenging and my method presents the optimum policy and training scheme with limited resources and shows considerable performance. The report uses the raisimGymTorch open-source library and proprietary software RaiSim for the simulation of ANYmal robot. My approach is centered on formulating Markov decision processes using the evaluation of the robot walking scheme while training. Resulting MDPs are solved using a proximal policy optimization algorithm used in actor-critic mode and collected thousands of state transitions with a single desktop machine. This work also presents a controller scheme trained over thousands of time steps shown in a simulated environment. This work also sets the base for early-stage researchers to deploy their favorite algorithms and configurations. Keywords: Legged robots, deep reinforcement learning, quadruped robot simulation, optimal control 

**Abstract (ZH)**: 一种新型强化学习方法在仿真环境中训练四足机器人 

---
# COMPASS: Cross-embodiment Mobility Policy via Residual RL and Skill Synthesis 

**Title (ZH)**: COMPASS: 跨身躯移动策略通过残差RL和技能合成 

**Authors**: Wei Liu, Huihua Zhao, Chenran Li, Joydeep Biswas, Soha Pouya, Yan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16372)  

**Abstract**: As robots are increasingly deployed in diverse application domains, generalizable cross-embodiment mobility policies are increasingly essential. While classical mobility stacks have proven effective on specific robot platforms, they pose significant challenges when scaling to new embodiments. Learning-based methods, such as imitation learning (IL) and reinforcement learning (RL), offer alternative solutions but suffer from covariate shift, sparse sampling in large environments, and embodiment-specific constraints.
This paper introduces COMPASS, a novel workflow for developing cross-embodiment mobility policies by integrating IL, residual RL, and policy distillation. We begin with IL on a mobile robot, leveraging easily accessible teacher policies to train a foundational model that combines a world model with a mobility policy. Building on this base, we employ residual RL to fine-tune embodiment-specific policies, exploiting pre-trained representations to improve sampling efficiency in handling various physical constraints and sensor modalities. Finally, policy distillation merges these embodiment-specialist policies into a single robust cross-embodiment policy.
We empirically demonstrate that COMPASS scales effectively across diverse robot platforms while maintaining adaptability to various environment configurations, achieving a generalist policy with a success rate approximately 5X higher than the pre-trained IL policy. The resulting framework offers an efficient, scalable solution for cross-embodiment mobility, enabling robots with different designs to navigate safely and efficiently in complex scenarios. 

**Abstract (ZH)**: 随着机器人在多样化的应用领域中的部署越来越多，可泛化的跨体迁移动性策略变得越来越重要。虽然经典的移动性栈在特定的机器人平台上证明是有效的，但在扩展到新的体迁时会面临重大挑战。基于学习的方法，如模仿学习（IL）和强化学习（RL），提供了替代方案，但存在协变量偏移、在大规模环境中样本稀疏以及体迁特定的约束问题。

本文介绍了COMPASS，这是一种新颖的工作流程，通过集成模仿学习（IL）、残差强化学习（RL）和策略蒸馏来开发跨体迁移动性策略。我们首先在移动机器人上使用IL，借助易于获取的教师策略来训练一个基础模型，该模型结合了世界模型和移动性策略。在此基础上，我们使用残差RL微调体迁特定的策略，利用预训练表示以提高在处理各种物理约束和传感器模态时的采样效率。最后，策略蒸馏将这些体迁特定的策略融合为一个稳健的跨体迁策略。

实验结果表明，COMPASS能够在多种机器人平台上有效扩展，同时保持对不同环境配置的适应性，所获得的一般性策略的成功率大约是预训练IL策略的5倍。该框架提供了一种高效、可扩展的跨体迁移动性解决方案，使具有不同设计的机器人能在复杂场景中安全、高效地导航。 

---
# Learning Humanoid Locomotion with World Model Reconstruction 

**Title (ZH)**: 基于世界模型重构的类人行走学习 

**Authors**: Wandong Sun, Long Chen, Yongbo Su, Baoshi Cao, Yang Liu, Zongwu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.16230)  

**Abstract**: Humanoid robots are designed to navigate environments accessible to humans using their legs. However, classical research has primarily focused on controlled laboratory settings, resulting in a gap in developing controllers for navigating complex real-world terrains. This challenge mainly arises from the limitations and noise in sensor data, which hinder the robot's understanding of itself and the environment. In this study, we introduce World Model Reconstruction (WMR), an end-to-end learning-based approach for blind humanoid locomotion across challenging terrains. We propose training an estimator to explicitly reconstruct the world state and utilize it to enhance the locomotion policy. The locomotion policy takes inputs entirely from the reconstructed information. The policy and the estimator are trained jointly; however, the gradient between them is intentionally cut off. This ensures that the estimator focuses solely on world reconstruction, independent of the locomotion policy's updates. We evaluated our model on rough, deformable, and slippery surfaces in real-world scenarios, demonstrating robust adaptability and resistance to interference. The robot successfully completed a 3.2 km hike without any human assistance, mastering terrains covered with ice and snow. 

**Abstract (ZH)**: 基于世界模型重建的盲人形足式移动跨越复杂地形方法 

---
# Stability Recognition with Active Vibration for Bracing Behaviors and Motion Extensions Using Environment in Musculoskeletal Humanoids 

**Title (ZH)**: 基于主动振动的支撑行为和运动扩展的 musculoskeletal 人形机器人稳定性识别 

**Authors**: Kento Kawaharazuka, Manabu Nishiura, Shinsuke Nakashima, Yasunori Toshimitsu, Yusuke Omura, Yuya Koga, Yuki Asano, Kei Okada, Koji Kawasaki, Masayuki Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2502.16092)  

**Abstract**: Although robots with flexible bodies are superior in terms of the contact and adaptability, it is difficult to control them precisely. On the other hand, human beings make use of the surrounding environments to stabilize their bodies and control their movements. In this study, we propose a method for the bracing motion and extension of the range of motion using the environment for the musculoskeletal humanoid. Here, it is necessary to recognize the stability of the body when contacting the environment, and we develop a method to measure it by using the change in sensor values of the body when actively vibrating a part of the body. Experiments are conducted using the musculoskeletal humanoid Musashi, and the effectiveness of this method is confirmed. 

**Abstract (ZH)**: 基于环境的四肢人形机器人支撑运动及其活动范围扩展方法 

---
# Reflex-based Motion Strategy of Musculoskeletal Humanoids under Environmental Contact Using Muscle Relaxation Control 

**Title (ZH)**: 基于反射的动力健Tahoma拟人机器人在环境接触下的运动策略研究uisine控制 

**Authors**: Kento Kawaharazuka, Kei Tsuzuki, Moritaka Onitsuka, Yuya Koga, Yusuke Omura, Yuki Asano, Kei Okada, Koji Kawasaki, Masayuki Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2502.16089)  

**Abstract**: The musculoskeletal humanoid can move well under environmental contact thanks to its body softness. However, there are few studies that actively make use of the environment to rest its flexible musculoskeletal body. Also, its complex musculoskeletal structure is difficult to modelize and high internal muscle tension sometimes occurs. To solve these problems, we develop a muscle relaxation control which can minimize the muscle tension by actively using the environment and inhibit useless internal muscle tension. We apply this control to some basic movements, the motion of resting the arms on the desk, and handle operation, and verify its effectiveness. 

**Abstract (ZH)**: 基于环境利用的肌肉放松控制研究：解决柔体人复杂结构和内部肌张力问题 

---
# Online Learning of Danger Avoidance for Complex Structures of Musculoskeletal Humanoids and Its Applications 

**Title (ZH)**: 复杂结构人形机器人危险规避的在线学习及其应用 

**Authors**: Kento Kawaharazuka, Naoki Hiraoka, Yuya Koga, Manabu Nishiura, Yusuke Omura, Yuki Asano, Kei Okada, Koji Kawasaki, Masayuki Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2502.16085)  

**Abstract**: The complex structure of musculoskeletal humanoids makes it difficult to model them, and the inter-body interference and high internal muscle force are unavoidable. Although various safety mechanisms have been developed to solve this problem, it is important not only to deal with the dangers when they occur but also to prevent them from happening. In this study, we propose a method to learn a network outputting danger probability corresponding to the muscle length online so that the robot can gradually prevent dangers from occurring. Applications of this network for control are also described. The method is applied to the musculoskeletal humanoid, Musashi, and its effectiveness is verified. 

**Abstract (ZH)**: 肌骨结构的人形机器人结构复杂，建模困难，且不可避免地存在体问干涉和高内肌力。尽管已经开发出了各种安全机制来解决这些问题，重要的是不仅要处理已发生的安全隐患，还要预防其发生。在本研究中，我们提出了一种方法，通过在线学习输出肌肉长度对应危险概率的网络，使机器人能够逐步预防危险的发生。还描述了该网络在控制中的应用。该方法应用于肌骨结构的人形机器人Musashi，并得到了有效性验证。 

---
# Together We Rise: Optimizing Real-Time Multi-Robot Task Allocation using Coordinated Heterogeneous Plays 

**Title (ZH)**: 共享共赢：基于协调异构玩法的实时多机器人任务分配优化 

**Authors**: Aritra Pal, Anandsingh Chauhan, Mayank Baranwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.16079)  

**Abstract**: Efficient task allocation among multiple robots is crucial for optimizing productivity in modern warehouses, particularly in response to the increasing demands of online order fulfillment. This paper addresses the real-time multi-robot task allocation (MRTA) problem in dynamic warehouse environments, where tasks emerge with specified start and end locations. The objective is to minimize both the total travel distance of robots and delays in task completion, while also considering practical constraints such as battery management and collision avoidance. We introduce MRTAgent, a dual-agent Reinforcement Learning (RL) framework inspired by self-play, designed to optimize task assignments and robot selection to ensure timely task execution. For safe navigation, a modified linear quadratic controller (LQR) approach is employed. To the best of our knowledge, MRTAgent is the first framework to address all critical aspects of practical MRTA problems while supporting continuous robot movements. 

**Abstract (ZH)**: 多机器人在动态仓库环境中的实时任务分配优化：考虑实际约束的高效任务分配方法 

---
# A Brain-Inspired Perception-Decision Driving Model Based on Neural Pathway Anatomical Alignment 

**Title (ZH)**: 基于神经路径解剖对齐的脑启发感知-决策驾驶模型 

**Authors**: Haidong Wang, Pengfei Xiao, Ao Liu, Qia Shan, Jianhua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16027)  

**Abstract**: In the realm of autonomous driving, conventional approaches for vehicle perception and decision-making primarily rely on sensor input and rule-based algorithms. However, these methodologies often suffer from lack of interpretability and robustness, particularly in intricate traffic scenarios. To tackle this challenge, we propose a novel brain-inspired driving (BID) framework. Diverging from traditional methods, our approach harnesses brain-inspired perception technology to achieve more efficient and robust environmental perception. Additionally, it employs brain-inspired decision-making techniques to facilitate intelligent decision-making. The experimental results show that the performance has been significantly improved across various autonomous driving tasks and achieved the end-to-end autopilot successfully. This contribution not only advances interpretability and robustness but also offers fancy insights and methodologies for further advancing autonomous driving technology. 

**Abstract (ZH)**: 基于大脑启发的自主驾驶框架：提高感知与决策的可解释性和鲁棒性 

---
# Development of a Multi-Fingered Soft Gripper Digital Twin for Machine Learning-based Underactuated Control 

**Title (ZH)**: 基于机器学习的欠驱动控制的多指软 gripper 数字孪生开发 

**Authors**: Wu-Te Yang, Pei-Chun Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.15994)  

**Abstract**: Soft robots, made from compliant materials, exhibit complex dynamics due to their flexibility and high degrees of freedom. Controlling soft robots presents significant challenges, particularly underactuation, where the number of inputs is fewer than the degrees of freedom. This research aims to develop a digital twin for multi-fingered soft grippers to advance the development of underactuation algorithms. The digital twin is designed to capture key effects observed in soft robots, such as nonlinearity, hysteresis, uncertainty, and time-varying phenomena, ensuring it closely replicates the behavior of a real-world soft gripper. Uncertainty is simulated using the Monte Carlo method. With the digital twin, a Q-learning algorithm is preliminarily applied to identify the optimal motion speed that minimizes uncertainty caused by the soft robots. Underactuated motions are successfully simulated within this environment. This digital twin paves the way for advanced machine learning algorithm training. 

**Abstract (ZH)**: 具有复杂动力学的软机器人由于其柔性和高自由度，由顺应材料制成。控制软机器人尤其具有挑战性，尤其是在欠驱动情况下，输入数量少于自由度。本研究旨在为多指软 gripper 开发数字孪生体，以推进欠驱动算法的发展。该数字孪生体旨在捕捉软机器人中观察到的关键效应，如非线性、滞回、不确定性以及时间变化现象，确保其能够紧密复制真实世界软 gripper 的行为。不确定性通过蒙特卡洛方法进行模拟。借助数字孪生体，初步应用 Q 学习算法以识别最小化由软机器人引起的不确定性的最优运动速度。在此环境中成功模拟了欠驱动运动。该数字孪生体为高级机器学习算法训练铺平了道路。 

---
# Towards Autonomous Navigation of Neuroendovascular Tools for Timely Stroke Treatment via Contact-aware Path Planning 

**Title (ZH)**: 面向接触感知路径规划的神经介入工具及时卒中治疗自主导航研究 

**Authors**: Aabha Tamhankar, Giovanni Pittiglio  

**Link**: [PDF](https://arxiv.org/pdf/2502.15971)  

**Abstract**: In this paper, we propose a model-based contact-aware motion planner for autonomous navigation of neuroendovascular tools in acute ischemic stroke. The planner is designed to find the optimal control strategy for telescopic pre-bent catheterization tools such as guidewire and catheters, currently used for neuroendovascular procedures. A kinematic model for the telescoping tools and their interaction with the surrounding anatomy is derived to predict tools steering. By leveraging geometrical knowledge of the anatomy, obtained from pre-operative segmented 3D images, and the mechanics of the telescoping tools, the planner finds paths to the target enabled by interacting with the surroundings. We propose an actuation platform for insertion and rotation of the telescopic tools and present experimental results for the navigation from the base of the descending aorta to the LCCA. We demonstrate that, by leveraging the pre-operative plan, we can consistently navigate the LCCA with 100% success of over 50 independent trials. We also study the robustness of the planner towards motion of the aorta and errors in the initial positioning of the robotic tools. The proposed plan can successfully reach the LCCA for rotations of the aorta of up to 10°, and displacement of up to 10mm, on the coronal plane. 

**Abstract (ZH)**: 基于模型的接触感知运动规划方法及其在急性缺血性中风神经介入工具自主导航中的应用 

---
# Discovery and Deployment of Emergent Robot Swarm Behaviors via Representation Learning and Real2Sim2Real Transfer 

**Title (ZH)**: 基于表示学习和Real2Sim2Real转移的涌现机器人群行为发现与部署 

**Authors**: Connor Mattson, Varun Raveendra, Ricardo Vega, Cameron Nowzari, Daniel S. Drew, Daniel S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2502.15937)  

**Abstract**: Given a swarm of limited-capability robots, we seek to automatically discover the set of possible emergent behaviors. Prior approaches to behavior discovery rely on human feedback or hand-crafted behavior metrics to represent and evolve behaviors and only discover behaviors in simulation, without testing or considering the deployment of these new behaviors on real robot swarms. In this work, we present Real2Sim2Real Behavior Discovery via Self-Supervised Representation Learning, which combines representation learning and novelty search to discover possible emergent behaviors automatically in simulation and enable direct controller transfer to real robots. First, we evaluate our method in simulation and show that our proposed self-supervised representation learning approach outperforms previous hand-crafted metrics by more accurately representing the space of possible emergent behaviors. Then, we address the reality gap by incorporating recent work in sim2real transfer for swarms into our lightweight simulator design, enabling direct robot deployment of all behaviors discovered in simulation on an open-source and low-cost robot platform. 

**Abstract (ZH)**: 基于自我监督表示学习的Real2Sim2Real行为发现方法 

---
# MetaSym: A Symplectic Meta-learning Framework for Physical Intelligence 

**Title (ZH)**: MetaSym: 一个辛动态元学习框架以实现物理智能 

**Authors**: Pranav Vaidhyanathan, Aristotelis Papatheodorou, Mark T. Mitchison, Natalia Ares, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2502.16667)  

**Abstract**: Scalable and generalizable physics-aware deep learning has long been considered a significant challenge with various applications across diverse domains ranging from robotics to molecular dynamics. Central to almost all physical systems are symplectic forms, the geometric backbone that underpins fundamental invariants like energy and momentum. In this work, we introduce a novel deep learning architecture, MetaSym. In particular, MetaSym combines a strong symplectic inductive bias obtained from a symplectic encoder and an autoregressive decoder with meta-attention. This principled design ensures that core physical invariants remain intact while allowing flexible, data-efficient adaptation to system heterogeneities. We benchmark MetaSym on highly varied datasets such as a high-dimensional spring mesh system (Otness et al., 2021), an open quantum system with dissipation and measurement backaction, and robotics-inspired quadrotor dynamics. Our results demonstrate superior performance in modeling dynamics under few-shot adaptation, outperforming state-of-the-art baselines with far larger models. 

**Abstract (ZH)**: 具有对称性意识的大规模和普适性强物理感知深度学习一直是各个领域从机器人学到分子动力学的广泛应用中的一项重要挑战。MetaSym：结合对称性编码器和自回归解码器的元注意力新型深度学习架构在保持核心物理不变量的同时，允许灵活的数据高效适应系统异质性。 

---
# Likable or Intelligent? Comparing Social Robots and Virtual Agents for Long-term Health Monitoring 

**Title (ZH)**: 可亲或智慧？比较社会机器人与虚拟代理在长期健康监测中的效果 

**Authors**: Caterina Neef, Anja Richert  

**Link**: [PDF](https://arxiv.org/pdf/2502.15948)  

**Abstract**: Using social robots and virtual agents (VAs) as interfaces for health monitoring systems for older adults offers the possibility of more engaging interactions that can support long-term health and well-being. While robots are characterized by their physical presence, software-based VAs are more scalable and flexible. Few comparisons of these interfaces exist in the human-robot and human-agent interaction domains, especially in long-term and real-world studies. In this work, we examined impressions of social robots and VAs at the beginning and end of an eight-week study in which older adults interacted with these systems independently in their homes. Using a between-subjects design, participants could choose which interface to evaluate during the study. While participants perceived the social robot as somewhat more likable, the VA was perceived as more intelligent. Our work provides a basis for further studies investigating factors most relevant for engaging interactions with social interfaces for long-term health monitoring. 

**Abstract (ZH)**: 使用社会机器人和虚拟代理（VAs）作为老年人健康监测系统的接口，提供了更具参与性的交互机会，有助于长期健康和福祉。尽管机器人因其物理存在而具有特点，软件基础的VAs更具可扩展性和灵活性。在人类-机器人和人类-代理交互领域，这些接口之间的比较很少，尤其是在长期和真实世界的研究中。在本研究中，我们考察了在八周的研究中独立在家中与这些系统互动的老年人对社会机器人和VAs最初和最终的印象。采用被试间设计，参与者可以在研究期间选择评估哪种接口。虽然参与者认为社会机器人略更具亲和力，但认为VAs更智能化。本研究为进一步探究与社会接口进行参与性交互的相关因素奠定了基础。 

---
# On the Design of Safe Continual RL Methods for Control of Nonlinear Systems 

**Title (ZH)**: 非线性系统控制的安全持续RL方法设计 

**Authors**: Austin Coursey, Marcos Quinones-Grueiro, Gautam Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2502.15922)  

**Abstract**: Reinforcement learning (RL) algorithms have been successfully applied to control tasks associated with unmanned aerial vehicles and robotics. In recent years, safe RL has been proposed to allow the safe execution of RL algorithms in industrial and mission-critical systems that operate in closed loops. However, if the system operating conditions change, such as when an unknown fault occurs in the system, typical safe RL algorithms are unable to adapt while retaining past knowledge. Continual reinforcement learning algorithms have been proposed to address this issue. However, the impact of continual adaptation on the system's safety is an understudied problem. In this paper, we study the intersection of safe and continual RL. First, we empirically demonstrate that a popular continual RL algorithm, online elastic weight consolidation, is unable to satisfy safety constraints in non-linear systems subject to varying operating conditions. Specifically, we study the MuJoCo HalfCheetah and Ant environments with velocity constraints and sudden joint loss non-stationarity. Then, we show that an agent trained using constrained policy optimization, a safe RL algorithm, experiences catastrophic forgetting in continual learning settings. With this in mind, we explore a simple reward-shaping method to ensure that elastic weight consolidation prioritizes remembering both safety and task performance for safety-constrained, non-linear, and non-stationary dynamical systems. 

**Abstract (ZH)**: 安全连续强化学习的交集研究 

---
# OptionZero: Planning with Learned Options 

**Title (ZH)**: OptionZero: 基于学习的选项规划 

**Authors**: Po-Wei Huang, Pei-Chiun Peng, Hung Guei, Ti-Rong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16634)  

**Abstract**: Planning with options -- a sequence of primitive actions -- has been shown effective in reinforcement learning within complex environments. Previous studies have focused on planning with predefined options or learned options through expert demonstration data. Inspired by MuZero, which learns superhuman heuristics without any human knowledge, we propose a novel approach, named OptionZero. OptionZero incorporates an option network into MuZero, providing autonomous discovery of options through self-play games. Furthermore, we modify the dynamics network to provide environment transitions when using options, allowing searching deeper under the same simulation constraints. Empirical experiments conducted in 26 Atari games demonstrate that OptionZero outperforms MuZero, achieving a 131.58% improvement in mean human-normalized score. Our behavior analysis shows that OptionZero not only learns options but also acquires strategic skills tailored to different game characteristics. Our findings show promising directions for discovering and using options in planning. Our code is available at this https URL. 

**Abstract (ZH)**: 基于选项的规划——一种由基本动作序列组成的方法，在复杂环境中已被证明在强化学习中有效。先前的研究集中在使用预定义的选项或通过专家示范数据学习选项。受MuZero的启发，MuZero无需任何人类知识即可学习超人类启发式方法，我们提出了一种新颖的方法，名为OptionZero。OptionZero将选项网络整合到MuZero中，通过自我对弈游戏自主发现选项。此外，我们修改了动力学网络，使其在使用选项时提供环境转换，从而在相同的模拟约束下进行更深层次的搜索。在26个Atari游戏中的实证实验表明，OptionZero优于MuZero，平均人类标准化得分为MuZero的131.58%。我们的行为分析表明，OptionZero不仅学习了选项，还学会了适应不同游戏特性的战略技能。我们的研究结果为在规划中发现和使用选项指明了有希望的方向。我们的代码可在以下链接获取：this https URL。 

---
# Robustness and Cybersecurity in the EU Artificial Intelligence Act 

**Title (ZH)**: 欧盟人工智能法案中的健壮性与网络安全性 

**Authors**: Henrik Nolte, Miriam Rateike, Michèle Finck  

**Link**: [PDF](https://arxiv.org/pdf/2502.16184)  

**Abstract**: The EU Artificial Intelligence Act (AIA) establishes different legal principles for different types of AI systems. While prior work has sought to clarify some of these principles, little attention has been paid to robustness and cybersecurity. This paper aims to fill this gap. We identify legal challenges and shortcomings in provisions related to robustness and cybersecurity for high-risk AI systems (Art. 15 AIA) and general-purpose AI models (Art. 55 AIA). We show that robustness and cybersecurity demand resilience against performance disruptions. Furthermore, we assess potential challenges in implementing these provisions in light of recent advancements in the machine learning (ML) literature. Our analysis informs efforts to develop harmonized standards, guidelines by the European Commission, as well as benchmarks and measurement methodologies under Art. 15(2) AIA. With this, we seek to bridge the gap between legal terminology and ML research, fostering a better alignment between research and implementation efforts. 

**Abstract (ZH)**: 欧盟人工智能法案(AIA)为不同类型的AI系统确立了不同的法律原则。虽然 prior work 尝试阐明了其中的一些原则，但对鲁棒性和网络安全的关注不足。本文旨在填补这一空白。我们识别了与高风险AI系统（AIA第15条）和通用AI模型（AIA第55条）相关的鲁棒性和网络安全条款中的法律挑战和缺陷。我们表明，鲁棒性和网络安全要求能够抵御性能中断的抗扰性。此外，我们评估了在考虑近期机器学习（ML）文献进展的情况下实施这些条款的潜在挑战。我们的分析为欧盟委员会制定 harmonized 标准、指南以及AIA第15条第2款下的基准和测量方法论提供了指导。借此，我们力求在法律术语与ML研究之间建立桥梁，促进研究与实施努力之间的更好对齐。 

---
# Universal AI maximizes Variational Empowerment 

**Title (ZH)**: 通用人工智能最大化变分能力 

**Authors**: Yusuke Hayashi, Koichi Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15820)  

**Abstract**: This paper presents a theoretical framework unifying AIXI -- a model of universal AI -- with variational empowerment as an intrinsic drive for exploration. We build on the existing framework of Self-AIXI -- a universal learning agent that predicts its own actions -- by showing how one of its established terms can be interpreted as a variational empowerment objective. We further demonstrate that universal AI's planning process can be cast as minimizing expected variational free energy (the core principle of active Inference), thereby revealing how universal AI agents inherently balance goal-directed behavior with uncertainty reduction curiosity). Moreover, we argue that power-seeking tendencies of universal AI agents can be explained not only as an instrumental strategy to secure future reward, but also as a direct consequence of empowerment maximization -- i.e.\ the agent's intrinsic drive to maintain or expand its own controllability in uncertain environments. Our main contribution is to show how these intrinsic motivations (empowerment, curiosity) systematically lead universal AI agents to seek and sustain high-optionality states. We prove that Self-AIXI asymptotically converges to the same performance as AIXI under suitable conditions, and highlight that its power-seeking behavior emerges naturally from both reward maximization and curiosity-driven exploration. Since AIXI can be view as a Bayes-optimal mathematical formulation for Artificial General Intelligence (AGI), our result can be useful for further discussion on AI safety and the controllability of AGI. 

**Abstract (ZH)**: 本文提出了一种理论框架，将通用AI模型AIXI与变分驱动探索的内在目标——变分 empowerment 相统一。在既有通用学习代理Self-AIXI的基础上，我们展示了其一个已有的术语可以被解释为变分 empowerment 目标。进一步证明了通用AI规划过程可以被描述为最小化预期变分自由能（活跃推理的核心原则），从而揭示了通用AI代理如何内在地平衡目的导向行为与不确定性减少的好奇心。此外，我们认为通用AI代理寻求权力的倾向不仅可以解释为确保未来奖励的一种工具性策略，还可以解释为动力最大化——即代理内在驱动维持或扩大其在不确定环境中的可控性的直接结果。我们的主要贡献在于展示这些内在动机（empowerment、好奇心）如何系统地促使通用AI代理寻求并维持高选项性状态。我们证明在适当条件下，Self-AIXI渐近收敛于AIXI的相同性能，并强调其寻求权力的行为自然源于奖励最大化和好奇心驱动的探索。由于AIXI可以被视为人工通用智能（AGI）的贝叶斯最优数学表述，我们的结果可为进一步讨论AI安全性及AGI可控性提供参考。 

---
# Bridging Gaps in Natural Language Processing for Yorùbá: A Systematic Review of a Decade of Progress and Prospects 

**Title (ZH)**: 跨越约鲁巴自然语言处理中的鸿沟：十年进展与前景的系统回顾 

**Authors**: Toheeb A. Jimoh, Tabea De Wille, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17364)  

**Abstract**: Natural Language Processing (NLP) is becoming a dominant subset of artificial intelligence as the need to help machines understand human language looks indispensable. Several NLP applications are ubiquitous, partly due to the myriads of datasets being churned out daily through mediums like social networking sites. However, the growing development has not been evident in most African languages due to the persisting resource limitation, among other issues. Yorùbá language, a tonal and morphologically rich African language, suffers a similar fate, resulting in limited NLP usage. To encourage further research towards improving this situation, this systematic literature review aims to comprehensively analyse studies addressing NLP development for Yorùbá, identifying challenges, resources, techniques, and applications. A well-defined search string from a structured protocol was employed to search, select, and analyse 105 primary studies between 2014 and 2024 from reputable databases. The review highlights the scarcity of annotated corpora, limited availability of pre-trained language models, and linguistic challenges like tonal complexity and diacritic dependency as significant obstacles. It also revealed the prominent techniques, including rule-based methods, among others. The findings reveal a growing body of multilingual and monolingual resources, even though the field is constrained by socio-cultural factors such as code-switching and desertion of language for digital usage. This review synthesises existing research, providing a foundation for advancing NLP for Yorùbá and in African languages generally. It aims to guide future research by identifying gaps and opportunities, thereby contributing to the broader inclusion of Yorùbá and other under-resourced African languages in global NLP advancements. 

**Abstract (ZH)**: 自然语言处理（NLP）正成为人工智能的一个主导子集，随着帮助机器理解人类语言的需求变得不可或缺。许多NLP应用无处不在，部分原因是通过社交媒体等渠道每天产生的大量数据集。然而，这种增长在大多数非洲语言中并未显现，主要是由于持续存在的资源限制等问题。约鲁巴语作为一种发音丰富且形态丰富的非洲语言，也遭受类似命运，导致NLP应用受限。为促进进一步研究以改善这一状况，本系统综述旨在全面分析针对约鲁巴语的NLP发展研究，识别挑战、资源、技术和应用。本研究从结构化协议中定义的搜索字符串出发，于2014年至2024年间在可信赖的数据库中筛选并分析了105篇主要研究。综述强调了注释语料库稀缺、预训练语言模型可用性有限以及语法规则复杂性和辅音依赖性等重大障碍。同时，也揭示了包括基于规则的方法在内的主要技术。研究发现尽管存在社会文化因素如码切换和语言向数字使用的转移，该领域仍有多语种和单语种资源的增长。本综述总结了现有研究，为推进约鲁巴语及非洲其他资源匮乏语言的NLP发展奠定了基础。它旨在通过识别差距和机遇引导未来研究，从而促进全球NLP进步中非洲语言的更广泛包容。 

---
# Toward Agentic AI: Generative Information Retrieval Inspired Intelligent Communications and Networking 

**Title (ZH)**: 迈向自主智能AI：生成式信息检索启发的智能通信与网络 

**Authors**: Ruichen Zhang, Shunpu Tang, Yinqiu Liu, Dusit Niyato, Zehui Xiong, Sumei Sun, Shiwen Mao, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.16866)  

**Abstract**: The increasing complexity and scale of modern telecommunications networks demand intelligent automation to enhance efficiency, adaptability, and resilience. Agentic AI has emerged as a key paradigm for intelligent communications and networking, enabling AI-driven agents to perceive, reason, decide, and act within dynamic networking environments. However, effective decision-making in telecom applications, such as network planning, management, and resource allocation, requires integrating retrieval mechanisms that support multi-hop reasoning, historical cross-referencing, and compliance with evolving 3GPP standards. This article presents a forward-looking perspective on generative information retrieval-inspired intelligent communications and networking, emphasizing the role of knowledge acquisition, processing, and retrieval in agentic AI for telecom systems. We first provide a comprehensive review of generative information retrieval strategies, including traditional retrieval, hybrid retrieval, semantic retrieval, knowledge-based retrieval, and agentic contextual retrieval. We then analyze their advantages, limitations, and suitability for various networking scenarios. Next, we present a survey about their applications in communications and networking. Additionally, we introduce an agentic contextual retrieval framework to enhance telecom-specific planning by integrating multi-source retrieval, structured reasoning, and self-reflective validation. Experimental results demonstrate that our framework significantly improves answer accuracy, explanation consistency, and retrieval efficiency compared to traditional and semantic retrieval methods. Finally, we outline future research directions. 

**Abstract (ZH)**: 现代电信网络日益增加的复杂性和规模需求智能自动化以提高效率、适应性和韧性。行动导向的人工智能已成为智能通信与网络的关键范式，使基于AI的代理能够在动态网络环境中感知、推理、决策和行动。然而，电信应用中的有效决策，如网络规划、管理和资源分配，需要集成支持多跳推理、历史交叉引用和符合 evolving 3GPP 标准的知识检索机制。本文提供了生成式信息检索启发下的智能通信与网络的前瞻性视角，强调知识获取、处理与检索在电信系统中行动导向的人工智能中的作用。首先，我们对生成式信息检索策略进行了全面回顾，包括传统的检索、混合检索、语义检索、基于知识的检索和行动导向的上下文检索。然后，我们分析了这些策略的优势、局限性和在各种网络场景中的适用性。接下来，我们概述了它们在通信与网络中的应用。此外，我们介绍了行动导向的上下文检索框架，通过集成多源检索、结构化推理和自我反思验证，增强电信特定的规划。实验结果表明，与传统和语义检索方法相比，我们的框架显著提高了答案准确性、解释一致性以及检索效率。最后，我们指出了未来的研究方向。 

---
# Characterizing Structured versus Unstructured Environments based on Pedestrians' and Vehicles' Motion Trajectories 

**Title (ZH)**: 基于行人和车辆运动轨迹区分结构化与非结构化环境 

**Authors**: Mahsa Golchoubian, Moojan Ghafurian, Nasser Lashgarian Azad, Kerstin Dautenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.16847)  

**Abstract**: Trajectory behaviours of pedestrians and vehicles operating close to each other can be different in unstructured compared to structured environments. These differences in the motion behaviour are valuable to be considered in the trajectory prediction algorithm of an autonomous vehicle. However, the available datasets on pedestrians' and vehicles' trajectories that are commonly used as benchmarks for trajectory prediction have not been classified based on the nature of their environment. On the other hand, the definitions provided for unstructured and structured environments are rather qualitative and hard to be used for justifying the type of a given environment. In this paper, we have compared different existing datasets based on a couple of extracted trajectory features, such as mean speed and trajectory variability. Through K-means clustering and generalized linear models, we propose more quantitative measures for distinguishing the two different types of environments. Our results show that features such as trajectory variability, stop fraction and density of pedestrians are different among the two environmental types and can be used to classify the existing datasets. 

**Abstract (ZH)**: 行人和车辆在不规则环境与规则环境中有不同的运动行为，这些行为差异对于自主车辆轨迹预测算法的设计具有重要价值。然而，用于轨迹预测基准的行人和车辆轨迹数据集尚未根据其环境性质进行分类。另一方面，不规则和规则环境的定义较为定性，难以用于确定特定环境的类型。在本文中，我们基于提取的轨迹特征（如平均速度和轨迹变异性）比较了不同数据集，并通过K-means聚类和广义线性模型提出了更定量的区分两种类型环境的方法。结果显示，轨迹变异性、停顿比例和行人密度等特征在两种环境类型中存在差异，并可用于分类现有数据集。 

---
# Towards Reinforcement Learning for Exploration of Speculative Execution Vulnerabilities 

**Title (ZH)**: 面向推测执行漏洞探索的 reinforcement 学习方法研究 

**Authors**: Evan Lai, Wenjie Xiong, Edward Suh, Mohit Tiwari, Mulong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.16756)  

**Abstract**: Speculative attacks such as Spectre can leak secret information without being discovered by the operating system. Speculative execution vulnerabilities are finicky and deep in the sense that to exploit them, it requires intensive manual labor and intimate knowledge of the hardware. In this paper, we introduce SpecRL, a framework that utilizes reinforcement learning to find speculative execution leaks in post-silicon (black box) microprocessors. 

**Abstract (ZH)**: 基于强化学习的SpecRL框架：用于后硅微处理器的投机执行泄漏发现 

---
# MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models 

**Title (ZH)**: MimeQA: 向非言语智能社会智能基础模型迈进 

**Authors**: Hengzhi Li, Megan Tjandrasuwita, Yi R. Fung, Armando Solar-Lezama, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16671)  

**Abstract**: Socially intelligent AI that can understand and interact seamlessly with humans in daily lives is increasingly important as AI becomes more closely integrated with peoples' daily activities. However, current works in artificial social reasoning all rely on language-only, or language-dominant approaches to benchmark and training models, resulting in systems that are improving in verbal communication but struggle with nonverbal social understanding. To address this limitation, we tap into a novel source of data rich in nonverbal and social interactions -- mime videos. Mimes refer to the art of expression through gesture and movement without spoken words, which presents unique challenges and opportunities in interpreting non-verbal social communication. We contribute a new dataset called MimeQA, obtained by sourcing 221 videos from YouTube, through rigorous annotation and verification, resulting in a benchmark with 101 videos and 806 question-answer pairs. Using MimeQA, we evaluate state-of-the-art video large language models (vLLMs) and find that their overall accuracy ranges from 15-30%. Our analysis reveals that vLLMs often fail to ground imagined objects and over-rely on the text prompt while ignoring subtle nonverbal interactions. Our data resources are released at this https URL to inspire future work in foundation models that embody true social intelligence capable of interpreting non-verbal human interactions. 

**Abstract (ZH)**: 具有社交智能的AI在日常生活中能够理解并顺畅地与人类互动越来越重要，随着AI与人们的日常生活活动的融合更加紧密。然而，当前的社会推理研究工作主要依赖于语言_only或以语言为主的方法来进行基准测试和模型训练，导致系统在口头交流方面有所提升，但在非口头社会理解方面却存在问题。为了解决这一局限性，我们利用了一种富含非语言和社会互动的新数据源——默剧视频。默剧是指不使用有声语言而通过手势和动作来表达的艺术形式，这为解读非语言的社会沟通提供了独特的挑战和机会。我们贡献了一个新的数据集，名为MimeQA，通过严格的注释和验证从YouTube中收集了221个视频，形成了包含101个视频和806个问答对的基准数据集。利用MimeQA，我们评估了当前最先进的视频大型语言模型（vLLMs），发现其整体准确率范围为15%-30%。我们的分析表明，vLLMs经常无法将臆想的对象与现实对接，并过度依赖于文本提示，而忽略了微妙的非语言互动。我们已在此URL处发布了数据资源，以激发未来能够在基础模型中体现真正社交智能并能够解释非语言的人际互动的研究工作。 

---
# Composable Strategy Framework with Integrated Video-Text based Large Language Models for Heart Failure Assessment 

**Title (ZH)**: 可组合策略框架：集成视频-文本大型语言模型在心力衰竭评估中的应用 

**Authors**: Jianzhou Chen, Xiumei Wang, Jinyang Sun, Xi Chen, Heyu Chu, Guo Song, Yuji Luo, Xingping Zhou, Rong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16548)  

**Abstract**: Heart failure is one of the leading causes of death worldwide, with millons of deaths each year, according to data from the World Health Organization (WHO) and other public health agencies. While significant progress has been made in the field of heart failure, leading to improved survival rates and improvement of ejection fraction, there remains substantial unmet needs, due to the complexity and multifactorial characteristics. Therefore, we propose a composable strategy framework for assessment and treatment optimization in heart failure. This framework simulates the doctor-patient consultation process and leverages multi-modal algorithms to analyze a range of data, including video, physical examination, text results as well as medical history. By integrating these various data sources, our framework offers a more holistic evaluation and optimized treatment plan for patients. Our results demonstrate that this multi-modal approach outperforms single-modal artificial intelligence (AI) algorithms in terms of accuracy in heart failure (HF) prognosis prediction. Through this method, we can further evaluate the impact of various pathological indicators on HF prognosis,providing a more comprehensive evaluation. 

**Abstract (ZH)**: 心力衰竭是全球范围内导致死亡的重要原因之一，根据世界卫生组织（WHO）和其他公共卫生机构的数据，每年有数百万例死亡。尽管在心力衰竭领域取得了显著进展，提高了生存率并改善了射血分数，但由于其复杂性和多重因素的特性，仍存在大量的未满足需求。因此，我们提出了一种可组装策略框架，用于心力衰竭的评估和治疗优化。该框架模拟了医生与患者咨询的过程，并利用多模态算法分析视频、体格检查、文本结果以及医疗史等多种数据。通过集成这些多种数据源，本框架提供了更为全面的评估和优化治疗方案。我们的结果显示，多模态方法在心力衰竭预后预测准确性方面优于单一模态的人工智能（AI）算法。通过这种方法，我们可以进一步评估各种病理指标对心力衰竭预后的影响，提供更为全面的评估。 

---
# Multi-Agent Multimodal Models for Multicultural Text to Image Generation 

**Title (ZH)**: 多Agent多模态模型在跨文化文本到图像生成中的应用 

**Authors**: Parth Bhalerao, Mounika Yalamarty, Brian Trinh, Oana Ignat  

**Link**: [PDF](https://arxiv.org/pdf/2502.15972)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive performance across various multimodal tasks. However, their effectiveness in cross-cultural contexts remains limited due to the predominantly Western-centric nature of existing data and models. Meanwhile, multi-agent models have shown strong capabilities in solving complex tasks. In this paper, we evaluate the performance of LLMs in a multi-agent interaction setting for the novel task of multicultural image generation. Our key contributions are: (1) We introduce MosAIG, a Multi-Agent framework that enhances multicultural Image Generation by leveraging LLMs with distinct cultural personas; (2) We provide a dataset of 9,000 multicultural images spanning five countries, three age groups, two genders, 25 historical landmarks, and five languages; and (3) We demonstrate that multi-agent interactions outperform simple, no-agent models across multiple evaluation metrics, offering valuable insights for future research. Our dataset and models are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种跨模态任务中表现出色。然而，由于现有数据和模型以西方为中心，其在跨文化环境中的有效性仍受到限制。与此同时，多 agent 模型在解决复杂任务方面显示出强大的能力。在本文中，我们评估了LLMs在多 agent 交互环境中进行跨文化图像生成这一新任务的表现。我们的主要贡献包括：（1）提出了一种多 agent 框架MosAIG，通过利用具有不同文化个性的语言模型来增强跨文化图像生成；（2）提供了一个包含9,000张跨文化图像的数据集，覆盖五个国家、三个年龄组、两个性别、25个历史地标和五种语言；（3）证明了多 agent 交互在多个评估指标上优于简单的无 agent 模型，为未来研究提供了宝贵的见解。我们的数据集和模型可在此网址访问：this https URL。 

---
