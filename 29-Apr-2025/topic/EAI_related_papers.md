# Real-Time Imitation of Human Head Motions, Blinks and Emotions by Nao Robot: A Closed-Loop Approach 

**Title (ZH)**: Nao机器人基于闭环方法的实时模仿人类头部运动、眨眼及情绪：一种闭环方法 

**Authors**: Keyhan Rayati, Amirhossein Feizi, Alireza Beigy, Pourya Shahverdi, Mehdi Tale Masouleh, Ahmad Kalhor  

**Link**: [PDF](https://arxiv.org/pdf/2504.19985)  

**Abstract**: This paper introduces a novel approach for enabling real-time imitation of human head motion by a Nao robot, with a primary focus on elevating human-robot interactions. By using the robust capabilities of the MediaPipe as a computer vision library and the DeepFace as an emotion recognition library, this research endeavors to capture the subtleties of human head motion, including blink actions and emotional expressions, and seamlessly incorporate these indicators into the robot's responses. The result is a comprehensive framework which facilitates precise head imitation within human-robot interactions, utilizing a closed-loop approach that involves gathering real-time feedback from the robot's imitation performance. This feedback loop ensures a high degree of accuracy in modeling head motion, as evidenced by an impressive R2 score of 96.3 for pitch and 98.9 for yaw. Notably, the proposed approach holds promise in improving communication for children with autism, offering them a valuable tool for more effective interaction. In essence, proposed work explores the integration of real-time head imitation and real-time emotion recognition to enhance human-robot interactions, with potential benefits for individuals with unique communication needs. 

**Abstract (ZH)**: 本文介绍了一种新的方法，通过NAO机器人实现人类头部运动的实时模仿，重点提升人机交互。该研究利用MediaPipe的强大计算机视觉能力和DeepFace的情感识别能力，力求捕捉人类头部运动的细微之处，包括眨眼动作和情感表达，并将这些指示无缝集成到机器人的响应中。结果形成了一套全面的框架，通过闭环方法利用机器人模仿表现的实时反馈，实现了精准的头部模仿。该反馈回路确保了头部运动建模的高度精确性，例如，俯仰角度的R2得分为96.3，偏航角度的R2得分为98.9。值得注意的是，所提出的方法在提高自闭症儿童的沟通能力方面展现出潜力，为他们提供了更为有效的交流工具。简而言之，本研究探讨了实时头部模仿和实时情感识别的集成，以增强人机交互，尤其对于有独特沟通需求的个体具有潜在益处。 

---
# NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks 

**Title (ZH)**: NORA：一个小规模开源通用视觉语言行动模型用于具身任务 

**Authors**: Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U-Xuan Tan, Navonil Majumder, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2504.19854)  

**Abstract**: Existing Visual-Language-Action (VLA) models have shown promising performance in zero-shot scenarios, demonstrating impressive task execution and reasoning capabilities. However, a significant challenge arises from the limitations of visual encoding, which can result in failures during tasks such as object grasping. Moreover, these models typically suffer from high computational overhead due to their large sizes, often exceeding 7B parameters. While these models excel in reasoning and task planning, the substantial computational overhead they incur makes them impractical for real-time robotic environments, where speed and efficiency are paramount. To address the limitations of existing VLA models, we propose NORA, a 3B-parameter model designed to reduce computational overhead while maintaining strong task performance. NORA adopts the Qwen-2.5-VL-3B multimodal model as its backbone, leveraging its superior visual-semantic understanding to enhance visual reasoning and action grounding. Additionally, our \model{} is trained on 970k real-world robot demonstrations and equipped with the FAST+ tokenizer for efficient action sequence generation. Experimental results demonstrate that NORA outperforms existing large-scale VLA models, achieving better task performance with significantly reduced computational overhead, making it a more practical solution for real-time robotic autonomy. 

**Abstract (ZH)**: 现有的视觉-语言-动作（VLA）模型在零样本场景中展示了有前景的性能，展示了令人印象深刻的任务执行和推理能力。然而，视觉编码的限制导致了在抓取等任务中出现困难。此外，这些模型通常由于其巨大的规模而遭受高额的计算开销，参数量往往超过7B。虽然这些模型在推理和任务规划方面表现出色，但其带来的巨大计算开销使其在需要高速度和高效率的实时机器人环境中不切实际。为了解决现有VLA模型的限制，我们提出了NORA，一个参数量为3B的模型，旨在减少计算开销同时保持强大的任务性能。NORA 采用Qwen-2.5-VL-3B 多模态模型作为骨干，利用其优越的视觉语义理解来增强视觉推理和动作定位。此外，我们的模型在970k真实的机器人演示数据上进行训练，并配备了FAST+分词器，以实现高效的动作序列生成。实验结果表明，NORA 在计算开销大幅减少的前提下，优于现有的大规模VLA模型，实现了更好的任务性能，使其成为实时机器人自主性更具实用性的解决方案。 

---
# Do You Know the Way? Human-in-the-Loop Understanding for Fast Traversability Estimation in Mobile Robotics 

**Title (ZH)**: 你知道路在何方？移动机器人快速可通过性评估的人机循环理解方法 

**Authors**: Andre Schreiber, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2504.19851)  

**Abstract**: The increasing use of robots in unstructured environments necessitates the development of effective perception and navigation strategies to enable field robots to successfully perform their tasks. In particular, it is key for such robots to understand where in their environment they can and cannot travel -- a task known as traversability estimation. However, existing geometric approaches to traversability estimation may fail to capture nuanced representations of traversability, whereas vision-based approaches typically either involve manually annotating a large number of images or require robot experience. In addition, existing methods can struggle to address domain shifts as they typically do not learn during deployment. To this end, we propose a human-in-the-loop (HiL) method for traversability estimation that prompts a human for annotations as-needed. Our method uses a foundation model to enable rapid learning on new annotations and to provide accurate predictions even when trained on a small number of quickly-provided HiL annotations. We extensively validate our method in simulation and on real-world data, and demonstrate that it can provide state-of-the-art traversability prediction performance. 

**Abstract (ZH)**: 在无结构环境中超密集机器人应用促使开发有效的感知与导航策略，以使田地机器人成功完成任务。特别是，此类机器人需要理解其环境中的可通行与不可通行区域——这一任务称为通行性估计。然而，现有的几何方法可能难以捕捉通行性的细微表现，而基于视觉的方法通常需要手动标注大量图像或依靠机器人的经验。此外，现有方法在部署过程中通常无法学习，难以应对领域转移。为此，我们提出了一种基于人类在环（HiL）的通行性估计方法，该方法需要时即请人类进行标注。我们的方法利用基础模型快速学习新的标注，并在仅使用少量快速提供的HiL标注训练时仍能够提供准确的预测。我们通过仿真和实际数据进行了广泛验证，并证明该方法可以提供最先进的通行性预测性能。 

---
# GPA-RAM: Grasp-Pretraining Augmented Robotic Attention Mamba for Spatial Task Learning 

**Title (ZH)**: GPA-RAM: 抓取预训练增强的机器人注意力Mamba空间任务学习 

**Authors**: Juyi Sheng, Yangjun Liu, Sheng Xu, Zhixin Yang, Mengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.19683)  

**Abstract**: Most existing robot manipulation methods prioritize task learning by enhancing perception through complex deep network architectures. However, they face challenges in real-time collision-free planning. Hence, Robotic Attention Mamba (RAM) is designed for refined planning. Specifically, by integrating Mamba and parallel single-view attention, RAM aligns multi-view vision and task-related language features, ensuring efficient fine-grained task planning with linear complexity and robust real-time performance. Nevertheless, it has the potential for further improvement in high-precision grasping and manipulation. Thus, Grasp-Pretraining Augmentation (GPA) is devised, with a grasp pose feature extractor pretrained utilizing object grasp poses directly inherited from whole-task demonstrations. Subsequently, the extracted grasp features are fused with the spatially aligned planning features from RAM through attention-based Pre-trained Location Fusion, preserving high-resolution grasping cues overshadowed by an overemphasis on global planning. To summarize, we propose Grasp-Pretraining Augmented Robotic Attention Mamba (GPA-RAM), dividing spatial task learning into RAM for planning skill learning and GPA for grasping skill learning. GPA-RAM demonstrates superior performance across three robot systems with distinct camera configurations in simulation and the real world. Compared with previous state-of-the-art methods, it improves the absolute success rate by 8.2% (from 79.3% to 87.5%) on the RLBench multi-task benchmark and 40\% (from 16% to 56%), 12% (from 86% to 98%) on the ALOHA bimanual manipulation tasks, while delivering notably faster inference. Furthermore, experimental results demonstrate that both RAM and GPA enhance task learning, with GPA proving robust to different architectures of pretrained grasp pose feature extractors. The website is: this https URL\_RAM\_website/. 

**Abstract (ZH)**: 基于抓取先-training增强的精细化机器人注意力规划方法（GPA-RAM） 

---
# An End-to-End Framework for Optimizing Foot Trajectory and Force in Dry Adhesion Legged Wall-Climbing Robots 

**Title (ZH)**: 一种优化足轨迹和干黏附壁爬机器人接触力的端到端框架 

**Authors**: Jichun Xiao, Jiawei Nie, Lina Hao, Zhi Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.19448)  

**Abstract**: Foot trajectory planning for dry adhesion legged climbing robots presents challenges, as the phases of foot detachment, swing, and adhesion significantly influence the adhesion and detachment forces essential for stable climbing. To tackle this, an end-to-end foot trajectory and force optimization framework (FTFOF) is proposed, which optimizes foot adhesion and detachment forces through trajectory adjustments. This framework accepts general foot trajectory constraints and user-defined parameters as input, ultimately producing an optimal single foot trajectory. It integrates three-segment $C^2$ continuous Bezier curves, tailored to various foot structures, enabling the generation of effective climbing trajectories. A dilate-based GRU predictive model establishes the relationship between foot trajectories and the corresponding foot forces. Multi-objective optimization algorithms, combined with a redundancy hierarchical strategy, identify the most suitable foot trajectory for specific tasks, thereby ensuring optimal performance across detachment force, adhesion force and vibration amplitude. Experimental validation on the quadruped climbing robot MST-M3F showed that, compared to commonly used trajectories in existing legged climbing robots, the proposed framework achieved reductions in maximum detachment force by 28 \%, vibration amplitude by 82 \%, which ensures the stable climbing of dry adhesion legged climbing robots. 

**Abstract (ZH)**: 基于干粘附的腿足攀爬机器人足轨迹规划与力优化框架 

---
# Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation 

**Title (ZH)**: 学习感知前向动力学模型以实现安全且平台感知的机器人导航 

**Authors**: Pascal Roth, Jonas Frey, Cesar Cadena, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2504.19322)  

**Abstract**: Ensuring safe navigation in complex environments requires accurate real-time traversability assessment and understanding of environmental interactions relative to the robot`s capabilities. Traditional methods, which assume simplified dynamics, often require designing and tuning cost functions to safely guide paths or actions toward the goal. This process is tedious, environment-dependent, and not this http URL overcome these issues, we propose a novel learned perceptive Forward Dynamics Model (FDM) that predicts the robot`s future state conditioned on the surrounding geometry and history of proprioceptive measurements, proposing a more scalable, safer, and heuristic-free solution. The FDM is trained on multiple years of simulated navigation experience, including high-risk maneuvers, and real-world interactions to incorporate the full system dynamics beyond rigid body simulation. We integrate our perceptive FDM into a zero-shot Model Predictive Path Integral (MPPI) planning framework, leveraging the learned mapping between actions, future states, and failure probability. This allows for optimizing a simplified cost function, eliminating the need for extensive cost-tuning to ensure safety. On the legged robot ANYmal, the proposed perceptive FDM improves the position estimation by on average 41% over competitive baselines, which translates into a 27% higher navigation success rate in rough simulation environments. Moreover, we demonstrate effective sim-to-real transfer and showcase the benefit of training on synthetic and real data. Code and models are made publicly available under this https URL. 

**Abstract (ZH)**: 确保在复杂环境中的安全导航需要进行准确的实时通行性评估及理解环境交互相对于机器人能力的关联。传统的假设简化动力学的方法通常需要设计和调整成本函数以安全地指导路径或动作朝向目标。这一过程繁琐，且依赖于环境，我们提出了一种新的基于学习的感知前向动力学模型（FDM），该模型可根据周围的几何结构和本体感知测量的历史预测机器人的未来状态，提供一种更为可扩展、更安全且无需启发式的解决方案。该FDM基于多年模拟导航经验进行训练，包括高风险机动和真实世界交互，以涵盖超出刚体模拟的整个系统动力学。我们将感知FDM整合到零样本模型预测路径积分（MPPI）规划框架中，利用动作、未来状态和故障概率之间的学习映射进行优化，从而消除广泛成本调优以确保安全的需要。在腿式机器人ANYmal上，提出的感知FDM在位置估计上平均改善了41%，在粗糙的模拟环境中将导航成功率提高了27%。此外，我们展示了有效的仿真实验转移，并展示了在合成和真实数据上进行训练的好处。代码和模型已在此处公开。 

---
# Unscented Particle Filter for Visual-inertial Navigation using IMU and Landmark Measurements 

**Title (ZH)**: 基于IMU和地标测量的无迹粒子滤波视觉-惯性导航 

**Authors**: Khashayar Ghanizadegan, Hashim A. Hashim  

**Link**: [PDF](https://arxiv.org/pdf/2504.19318)  

**Abstract**: This paper introduces a geometric Quaternion-based Unscented Particle Filter for Visual-Inertial Navigation (QUPF-VIN) specifically designed for a vehicle operating with six degrees of freedom (6 DoF). The proposed QUPF-VIN technique is quaternion-based capturing the inherently nonlinear nature of true navigation kinematics. The filter fuses data from a low-cost inertial measurement unit (IMU) and landmark observations obtained via a vision sensor. The QUPF-VIN is implemented in discrete form to ensure seamless integration with onboard inertial sensing systems. Designed for robustness in GPS-denied environments, the proposed method has been validated through experiments with real-world dataset involving an unmanned aerial vehicle (UAV) equipped with a 6-axis IMU and a stereo camera, operating with 6 DoF. The numerical results demonstrate that the QUPF-VIN provides superior tracking accuracy compared to ground truth data. Additionally, a comparative analysis with a standard Kalman filter-based navigation technique further highlights the enhanced performance of the QUPF-VIN. 

**Abstract (ZH)**: 基于几何四元数的无迹粒子滤波视觉惯性导航（QUPF-VIN）技术及其在六自由度车辆中的应用 

---
# Quantitative evaluation of brain-inspired vision sensors in high-speed robotic perception 

**Title (ZH)**: 基于大脑启发的视觉传感器在高速机器人感知中的定量评价 

**Authors**: Taoyi Wang, Lijian Wang, Yihan Lin, Mingtao Ou, Yuguo Chen, Xinglong Ji, Rong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.19253)  

**Abstract**: Perception systems in robotics encounter significant challenges in high-speed and dynamic conditions when relying on traditional cameras, where motion blur can compromise spatial feature integrity and task performance. Brain-inspired vision sensors (BVS) have recently gained attention as an alternative, offering high temporal resolution with reduced bandwidth and power requirements. Here, we present the first quantitative evaluation framework for two representative classes of BVSs in variable-speed robotic sensing, including event-based vision sensors (EVS) that detect asynchronous temporal contrasts, and the primitive-based sensor Tianmouc that employs a complementary mechanism to encode both spatiotemporal changes and intensity. A unified testing protocol is established, including crosssensor calibrations, standardized testing platforms, and quality metrics to address differences in data modality. From an imaging standpoint, we evaluate the effects of sensor non-idealities, such as motion-induced distortion, on the capture of structural information. For functional benchmarking, we examine task performance in corner detection and motion estimation under different rotational speeds. Results indicate that EVS performs well in highspeed, sparse scenarios and in modestly fast, complex scenes, but exhibits performance limitations in high-speed, cluttered settings due to pixel-level bandwidth variations and event rate saturation. In comparison, Tianmouc demonstrates consistent performance across sparse and complex scenarios at various speeds, supported by its global, precise, high-speed spatiotemporal gradient samplings. These findings offer valuable insights into the applicationdependent suitability of BVS technologies and support further advancement in this area. 

**Abstract (ZH)**: 基于脑启发视觉传感器在可变速度机器人感知中的量化评估框架 

---
# Generative AI in Embodied Systems: System-Level Analysis of Performance, Efficiency and Scalability 

**Title (ZH)**: 生成式AI在具身系统中的应用：性能、效率和扩展性系统的分析 

**Authors**: Zishen Wan, Jiayi Qian, Yuhang Du, Jason Jabbour, Yilun Du, Yang Katie Zhao, Arijit Raychowdhury, Tushar Krishna, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18945)  

**Abstract**: Embodied systems, where generative autonomous agents engage with the physical world through integrated perception, cognition, action, and advanced reasoning powered by large language models (LLMs), hold immense potential for addressing complex, long-horizon, multi-objective tasks in real-world environments. However, deploying these systems remains challenging due to prolonged runtime latency, limited scalability, and heightened sensitivity, leading to significant system inefficiencies.
In this paper, we aim to understand the workload characteristics of embodied agent systems and explore optimization solutions. We systematically categorize these systems into four paradigms and conduct benchmarking studies to evaluate their task performance and system efficiency across various modules, agent scales, and embodied tasks. Our benchmarking studies uncover critical challenges, such as prolonged planning and communication latency, redundant agent interactions, complex low-level control mechanisms, memory inconsistencies, exploding prompt lengths, sensitivity to self-correction and execution, sharp declines in success rates, and reduced collaboration efficiency as agent numbers increase. Leveraging these profiling insights, we suggest system optimization strategies to improve the performance, efficiency, and scalability of embodied agents across different paradigms. This paper presents the first system-level analysis of embodied AI agents, and explores opportunities for advancing future embodied system design. 

**Abstract (ZH)**: 具身系统中的生成自主代理通过集成感知、认知、行动和由大规模语言模型支持的高级推理与物理世界互动，具有应对真实环境中的复杂、长周期、多目标任务的巨大潜力。然而，由于持续的运行时延、有限的可扩展性和增强的敏感性，部署这些系统仍然具有挑战性，导致系统效率低下。
本文旨在理解具身代理系统的工作负载特征并探索优化方案。我们系统地将这些系统划分为四种范式，并通过基准测试研究评估其在不同模块、代理规模和具身任务中的任务性能和系统效率。我们的基准测试研究揭示了关键挑战，如持续的规划和通信延迟、冗余的代理交互、复杂的低级控制机制、内存不一致性、提示长度爆炸性增长、对自我校正和执行的敏感性、成功率的急剧下降以及随着代理数量增加的合作效率降低。基于这些分析结果，我们提出了系统的优化策略，以提高不同范式下具身代理的性能、效率和可扩展性。本文首次对具身AI代理进行了系统级分析，并探讨了推进未来具身系统设计的机会。 

---
# Demonstrating DVS: Dynamic Virtual-Real Simulation Platform for Mobile Robotic Tasks 

**Title (ZH)**: 基于移动机器人任务的动态虚拟-现实仿真平台Demonstrating DVS: Dynamic Virtual-Real Simulation Platform for Mobile Robotic Tasks 

**Authors**: Zijie Zheng, Zeshun Li, Yunpeng Wang, Qinghongbing Xie, Long Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2504.18944)  

**Abstract**: With the development of embodied artificial intelligence, robotic research has increasingly focused on complex tasks. Existing simulation platforms, however, are often limited to idealized environments, simple task scenarios and lack data interoperability. This restricts task decomposition and multi-task learning. Additionally, current simulation platforms face challenges in dynamic pedestrian modeling, scene editability, and synchronization between virtual and real assets. These limitations hinder real world robot deployment and feedback. To address these challenges, we propose DVS (Dynamic Virtual-Real Simulation Platform), a platform for dynamic virtual-real synchronization in mobile robotic tasks. DVS integrates a random pedestrian behavior modeling plugin and large-scale, customizable indoor scenes for generating annotated training datasets. It features an optical motion capture system, synchronizing object poses and coordinates between virtual and real world to support dynamic task benchmarking. Experimental validation shows that DVS supports tasks such as pedestrian trajectory prediction, robot path planning, and robotic arm grasping, with potential for both simulation and real world deployment. In this way, DVS represents more than just a versatile robotic platform; it paves the way for research in human intervention in robot execution tasks and real-time feedback algorithms in virtual-real fusion environments. More information about the simulation platform is available on this https URL. 

**Abstract (ZH)**: 随着体态人工智能的发展，机器人研究越来越多地关注复杂任务。然而，现有的仿真平台往往局限于理想化的环境、简单的任务场景以及数据互操作性不足的问题，这限制了任务分解和多任务学习。此外，当前的仿真平台在动态行人建模、场景可编辑性和虚拟与现实资产之间的同步方面也面临挑战。这些限制妨碍了现实世界中机器人部署和反馈。为了解决这些挑战，我们提出DVS（动态虚拟-现实仿真平台），一个用于移动机器人任务动态虚拟-现实同步的平台。DVS集成了随机行人行为建模插件和可定制的大规模室内场景，用于生成标注的训练数据集。它配备了光学动作捕捉系统，实现虚拟和现实世界中物体姿态和坐标的同步，以支持动态任务基准测试。实验验证显示，DVS支持行人轨迹预测、机器人路径规划和机械臂抓取等任务，具有在仿真和现实世界中部署的潜力。通过这种方式，DVS不仅代表了一个多功能的机器人平台，还为在虚拟-现实融合环境中进行人类干预机器人执行任务的研究以及实时反馈算法的研究铺平了道路。了解更多关于仿真平台的信息，请访问此 [链接]。 

---
# RoboVerse: Towards a Unified Platform, Dataset and Benchmark for Scalable and Generalizable Robot Learning 

**Title (ZH)**: RoboVerse: 向统一平台、数据集和基准测试方向实现可扩展和泛化的机器人学习 

**Authors**: Haoran Geng, Feishi Wang, Songlin Wei, Yuyang Li, Bangjun Wang, Boshi An, Charlie Tianyue Cheng, Haozhe Lou, Peihao Li, Yen-Jen Wang, Yutong Liang, Dylan Goetting, Chaoyi Xu, Haozhe Chen, Yuxi Qian, Yiran Geng, Jiageng Mao, Weikang Wan, Mingtong Zhang, Jiangran Lyu, Siheng Zhao, Jiazhao Zhang, Jialiang Zhang, Chengyang Zhao, Haoran Lu, Yufei Ding, Ran Gong, Yuran Wang, Yuxuan Kuang, Ruihai Wu, Baoxiong Jia, Carlo Sferrazza, Hao Dong, Siyuan Huang, Yue Wang, Jitendra Malik, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2504.18904)  

**Abstract**: Data scaling and standardized evaluation benchmarks have driven significant advances in natural language processing and computer vision. However, robotics faces unique challenges in scaling data and establishing evaluation protocols. Collecting real-world data is resource-intensive and inefficient, while benchmarking in real-world scenarios remains highly complex. Synthetic data and simulation offer promising alternatives, yet existing efforts often fall short in data quality, diversity, and benchmark standardization. To address these challenges, we introduce RoboVerse, a comprehensive framework comprising a simulation platform, a synthetic dataset, and unified benchmarks. Our simulation platform supports multiple simulators and robotic embodiments, enabling seamless transitions between different environments. The synthetic dataset, featuring high-fidelity physics and photorealistic rendering, is constructed through multiple approaches. Additionally, we propose unified benchmarks for imitation learning and reinforcement learning, enabling evaluation across different levels of generalization. At the core of the simulation platform is MetaSim, an infrastructure that abstracts diverse simulation environments into a universal interface. It restructures existing simulation environments into a simulator-agnostic configuration system, as well as an API aligning different simulator functionalities, such as launching simulation environments, loading assets with initial states, stepping the physics engine, etc. This abstraction ensures interoperability and extensibility. Comprehensive experiments demonstrate that RoboVerse enhances the performance of imitation learning, reinforcement learning, world model learning, and sim-to-real transfer. These results validate the reliability of our dataset and benchmarks, establishing RoboVerse as a robust solution for advancing robot learning. 

**Abstract (ZH)**: 基于数据缩放和标准化评估基准，自然语言处理和计算机视觉取得了显著进展。然而，机器人技术在数据缩放和建立评估标准方面面临独特挑战。收集真实世界数据资源密集且效率低下，而在真实世界场景中的基准测试依然非常复杂。合成数据和模拟提供了有希望的替代方案，但现有努力往往在数据质量和多样性以及基准测试标准化方面不尽如人意。为应对这些挑战，我们引入了RoboVerse，这是一个包含模拟平台、合成数据集和统一基准的全面框架。我们的模拟平台支持多种模拟器和机器人实体，使得在不同环境之间实现无缝过渡成为可能。合成数据集包含高保真物理和光realistic渲染，通过多种方法构建而成。此外，我们还提出了统一的模仿学习和强化学习基准，使得在不同泛化水平上进行评估成为可能。模拟平台的核心是MetaSim基础设施，它将多种多样的模拟环境抽象为一个通用界面。该基础设施重构现有模拟环境为一种与模拟器无关的配置系统，并提供一种API来对齐不同模拟器的功能，如启动模拟环境、加载初始状态的资源、推进物理引擎等。这种抽象确保了互操作性和可扩展性。综合实验表明，RoboVerse提升了模仿学习、强化学习、世界模型学习和模拟到现实应用的性能。这些结果验证了我们数据集和基准的可靠性，将RoboVerse确立为推动机器人学习发展的稳健解决方案。 

---
# Imitation Learning for Autonomous Driving: Insights from Real-World Testing 

**Title (ZH)**: 自动驾驶领域的模仿学习：来自实际测试的见解 

**Authors**: Hidayet Ersin Dursun, Yusuf Güven, Tufan Kumbasar  

**Link**: [PDF](https://arxiv.org/pdf/2504.18847)  

**Abstract**: This work focuses on the design of a deep learning-based autonomous driving system deployed and tested on the real-world MIT Racecar to assess its effectiveness in driving scenarios. The Deep Neural Network (DNN) translates raw image inputs into real-time steering commands in an end-to-end learning fashion, following the imitation learning framework. The key design challenge is to ensure that DNN predictions are accurate and fast enough, at a high sampling frequency, and result in smooth vehicle operation under different operating conditions. In this study, we design and compare various DNNs, to identify the most effective approach for real-time autonomous driving. In designing the DNNs, we adopted an incremental design approach that involved enhancing the model capacity and dataset to address the challenges of real-world driving scenarios. We designed a PD system, CNN, CNN-LSTM, and CNN-NODE, and evaluated their performance on the real-world MIT Racecar. While the PD system handled basic lane following, it struggled with sharp turns and lighting variations. The CNN improved steering but lacked temporal awareness, which the CNN-LSTM addressed as it resulted in smooth driving performance. The CNN-NODE performed similarly to the CNN-LSTM in handling driving dynamics, yet with slightly better driving performance. The findings of this research highlight the importance of iterative design processes in developing robust DNNs for autonomous driving applications. The experimental video is available at this https URL. 

**Abstract (ZH)**: 基于深度学习的自主驾驶系统设计与在MIT Racecar上的实际测试及其在驾驶场景中的有效性评估 

---
# STDArm: Transferring Visuomotor Policies From Static Data Training to Dynamic Robot Manipulation 

**Title (ZH)**: STDArm: 从静态数据训练向动态机器人操作转移视觉运动策略 

**Authors**: Yifan Duan, Heng Li, Yilong Wu, Wenhao Yu, Xinran Zhang, Yedong Shen, Jianmin Ji, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18792)  

**Abstract**: Recent advances in mobile robotic platforms like quadruped robots and drones have spurred a demand for deploying visuomotor policies in increasingly dynamic environments. However, the collection of high-quality training data, the impact of platform motion and processing delays, and limited onboard computing resources pose significant barriers to existing solutions. In this work, we present STDArm, a system that directly transfers policies trained under static conditions to dynamic platforms without extensive modifications.
The core of STDArm is a real-time action correction framework consisting of: (1) an action manager to boost control frequency and maintain temporal consistency, (2) a stabilizer with a lightweight prediction network to compensate for motion disturbances, and (3) an online latency estimation module for calibrating system parameters. In this way, STDArm achieves centimeter-level precision in mobile manipulation tasks.
We conduct comprehensive evaluations of the proposed STDArm on two types of robotic arms, four types of mobile platforms, and three tasks. Experimental results indicate that the STDArm enables real-time compensation for platform motion disturbances while preserving the original policy's manipulation capabilities, achieving centimeter-level operational precision during robot motion. 

**Abstract (ZH)**: 近期，四足机器人和无人机等移动机器人平台的进步推动了在动态环境中部署视听运动策略的需求。然而，高质量训练数据的采集、平台运动和处理延迟的影响以及有限的车载计算资源构成了现有解决方案的重要障碍。本文介绍了一种名为STDArm的系统，该系统能够在无需进行大量修改的情况下，将静态条件下训练的策略直接转移到动态平台上。STDArm的核心是一个实时动作修正框架，包括：（1）动作管理器以提高控制频率并保持时间一致性，（2）用于补偿运动干扰的轻量级预测稳定器，以及（3）在线延时估计模块以校准系统参数。通过这种方式，STDArm在移动操作任务中实现了厘米级的精度。我们在两种类型的机器人手臂、四种类型的移动平台和三种任务上进行了全面评估。实验结果表明，STDArm能够在保持原始策略操作能力的同时，实时补偿平台运动干扰，实现机器人运动中的厘米级操作精度。 

---
# Robust Push Recovery on Bipedal Robots: Leveraging Multi-Domain Hybrid Systems with Reduced-Order Model Predictive Control 

**Title (ZH)**: 双足机器人稳健的推送恢复：基于降阶模型预测控制的多领域混合系统方法 

**Authors**: Min Dai, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2504.18698)  

**Abstract**: In this paper, we present a novel control framework to achieve robust push recovery on bipedal robots while locomoting. The key contribution is the unification of hybrid system models of locomotion with a reduced-order model predictive controller determining: foot placement, step timing, and ankle control. The proposed reduced-order model is an augmented Linear Inverted Pendulum model with zero moment point coordinates; this is integrated within a model predictive control framework for robust stabilization under external disturbances. By explicitly leveraging the hybrid dynamics of locomotion, our approach significantly improves stability and robustness across varying walking heights, speeds, step durations, and is effective for both flat-footed and more complex multi-domain heel-to-toe walking patterns. The framework is validated with high-fidelity simulation on Cassie, a 3D underactuated robot, showcasing real-time feasibility and substantially improved stability. The results demonstrate the robustness of the proposed method in dynamic environments. 

**Abstract (ZH)**: 本文提出了一种新颖的控制框架，以实现具有运动能力的双足机器人在行进过程中稳健的推倒恢复。关键贡献是将行进的混合系统模型与降低阶数的模型预测控制器结合，该控制器决定：脚部放置、步态时间以及踝关节控制。提出的降低阶数模型为带零力点坐标的增广线性倒 pendulum 模型；该模型整合到一种模型预测控制框架中，以在外部干扰下实现鲁棒的稳定化。通过明确利用行进的混合动力学特性，我们的方法显著提高了在不同行走高度、速度和步长时间下的稳定性，并且对于平足行走模式和更复杂的多领域后跟到脚尖行走模式都有效。该框架通过在高保真模拟中使用 Cassie（一种 3D 欠驱动机器人）进行验证，展示了实时可行性和显著改善的稳定性。结果表明，所提出方法在动态环境中的鲁棒性。 

---
# Learning to Drive from a World Model 

**Title (ZH)**: 从世界模型学习驾驶 

**Authors**: Mitchell Goff, Greg Hogan, George Hotz, Armand du Parc Locmaria, Kacper Raczy, Harald Schäfer, Adeeb Shihadeh, Weixing Zhang, Yassine Yousfi  

**Link**: [PDF](https://arxiv.org/pdf/2504.19077)  

**Abstract**: Most self-driving systems rely on hand-coded perception outputs and engineered driving rules. Learning directly from human driving data with an end-to-end method can allow for a training architecture that is simpler and scales well with compute and data.
In this work, we propose an end-to-end training architecture that uses real driving data to train a driving policy in an on-policy simulator. We show two different methods of simulation, one with reprojective simulation and one with a learned world model. We show that both methods can be used to train a policy that learns driving behavior without any hand-coded driving rules. We evaluate the performance of these policies in a closed-loop simulation and when deployed in a real-world advanced driver-assistance system. 

**Abstract (ZH)**: 一种使用实际驾驶数据训练驾驶策略的端到端训练架构：基于策略模拟的方法和 Learned 世界模型的实现 

---
# Deep Learning-Based Multi-Modal Fusion for Robust Robot Perception and Navigation 

**Title (ZH)**: 基于深度学习的多模态融合方法用于鲁棒机器人感知与导航 

**Authors**: Delun Lai, Yeyubei Zhang, Yunchong Liu, Chaojie Li, Huadong Mo  

**Link**: [PDF](https://arxiv.org/pdf/2504.19002)  

**Abstract**: This paper introduces a novel deep learning-based multimodal fusion architecture aimed at enhancing the perception capabilities of autonomous navigation robots in complex environments. By utilizing innovative feature extraction modules, adaptive fusion strategies, and time-series modeling mechanisms, the system effectively integrates RGB images and LiDAR data. The key contributions of this work are as follows: a. the design of a lightweight feature extraction network to enhance feature representation; b. the development of an adaptive weighted cross-modal fusion strategy to improve system robustness; and c. the incorporation of time-series information modeling to boost dynamic scene perception accuracy. Experimental results on the KITTI dataset demonstrate that the proposed approach increases navigation and positioning accuracy by 3.5% and 2.2%, respectively, while maintaining real-time performance. This work provides a novel solution for autonomous robot navigation in complex environments. 

**Abstract (ZH)**: 基于深度学习的多模态融合架构在复杂环境自主导航机器人感知能力增强的研究 

---
# Hierarchical Reinforcement Learning in Multi-Goal Spatial Navigation with Autonomous Mobile Robots 

**Title (ZH)**: 多目标空间导航中基于层次的强化学习自主移动机器人研究 

**Authors**: Brendon Johnson, Alfredo Weitzenfeld  

**Link**: [PDF](https://arxiv.org/pdf/2504.18794)  

**Abstract**: Hierarchical reinforcement learning (HRL) is hypothesized to be able to take advantage of the inherent hierarchy in robot learning tasks with sparse reward schemes, in contrast to more traditional reinforcement learning algorithms. In this research, hierarchical reinforcement learning is evaluated and contrasted with standard reinforcement learning in complex navigation tasks. We evaluate unique characteristics of HRL, including their ability to create sub-goals and the termination function. We constructed experiments to test the differences between PPO and HRL, different ways of creating sub-goals, manual vs automatic sub-goal creation, and the effects of the frequency of termination on performance. These experiments highlight the advantages of HRL and how it achieves these advantages. 

**Abstract (ZH)**: 层次强化学习（HRL）被假定能够在稀疏奖励方案下利用机器人学习任务中的固有层次结构，与传统的强化学习算法相比具有优势。在本研究中，层次强化学习与标准强化学习在复杂导航任务中的性能进行了评估和对比。我们评估了层次强化学习的独特特性，包括其创建子目标和终止函数的能力。我们构建了实验来测试PPO与HRL之间的差异、不同子目标创建方式、手动 vs 自动子目标创建以及终止频率对性能的影响。这些实验突显了层次强化学习的优势及其实现这些优势的方式。 

---
# Sparks: Multi-Agent Artificial Intelligence Model Discovers Protein Design Principles 

**Title (ZH)**: 火花：多智能体人工智能模型发现蛋白质设计原理 

**Authors**: Alireza Ghafarollahi, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2504.19017)  

**Abstract**: Advances in artificial intelligence (AI) promise autonomous discovery, yet most systems still resurface knowledge latent in their training data. We present Sparks, a multi-modal multi-agent AI model that executes the entire discovery cycle that includes hypothesis generation, experiment design and iterative refinement to develop generalizable principles and a report without human intervention. Applied to protein science, Sparks uncovered two previously unknown phenomena: (i) a length-dependent mechanical crossover whereby beta-sheet-biased peptides surpass alpha-helical ones in unfolding force beyond ~80 residues, establishing a new design principle for peptide mechanics; and (ii) a chain-length/secondary-structure stability map revealing unexpectedly robust beta-sheet-rich architectures and a "frustration zone" of high variance in mixed alpha/beta folds. These findings emerged from fully self-directed reasoning cycles that combined generative sequence design, high-accuracy structure prediction and physics-aware property models, with paired generation-and-reflection agents enforcing self-correction and reproducibility. The key result is that Sparks can independently conduct rigorous scientific inquiry and identify previously unknown scientific principles. 

**Abstract (ZH)**: 人工智能的进步 promise 自主发现，然而大多数系统仍然依赖于其训练数据中隐含的知识。我们提出了 Sparks，一个跨模态多代理 AI 模型，能够执行包括假设生成、实验设计和迭代精炼在内的整个发现周期，以开发可泛化的原理并生成报告，无需人类干预。应用于蛋白质科学，Sparks 揭示了两个未知的现象：（i）长度相关的机械交叉现象，其中偏向beta片层的肽在超过约80残基时的解折叠力超过α螺旋肽，确立了肽力学设计的新原则；以及（ii）链长/二级结构稳定性图，揭示了出人意料的丰富β片层结构，并且在α/β混合折叠中存在高变异性“挫折区”。这些发现源自结合生成序列设计、高精度结构预测和物理感知属性模型的完全自主推理循环，并通过配对的生成和反思代理确保自我修正和可重复性。关键成果是 Sparks 能够独立进行严格的科学研究，并识别出未知的科学原理。 

---
# Securing Agentic AI: A Comprehensive Threat Model and Mitigation Framework for Generative AI Agents 

**Title (ZH)**: 保障自主AI：生成型AI代理的全面威胁模型与缓解框架 

**Authors**: Vineeth Sai Narajala, Om Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2504.19956)  

**Abstract**: As generative AI (GenAI) agents become more common in enterprise settings, they introduce security challenges that differ significantly from those posed by traditional systems. These agents are not just LLMs; they reason, remember, and act, often with minimal human oversight. This paper introduces a comprehensive threat model tailored specifically for GenAI agents, focusing on how their autonomy, persistent memory access, complex reasoning, and tool integration create novel risks. This research work identifies 9 primary threats and organizes them across five key domains: cognitive architecture vulnerabilities, temporal persistence threats, operational execution vulnerabilities, trust boundary violations, and governance circumvention. These threats are not just theoretical they bring practical challenges such as delayed exploitability, cross-system propagation, cross system lateral movement, and subtle goal misalignments that are hard to detect with existing frameworks and standard approaches. To help address this, the research work present two complementary frameworks: ATFAA - Advanced Threat Framework for Autonomous AI Agents, which organizes agent-specific risks, and SHIELD, a framework proposing practical mitigation strategies designed to reduce enterprise exposure. While this work builds on existing work in LLM and AI security, the focus is squarely on what makes agents different and why those differences matter. Ultimately, this research argues that GenAI agents require a new lens for security. If we fail to adapt our threat models and defenses to account for their unique architecture and behavior, we risk turning a powerful new tool into a serious enterprise liability. 

**Abstract (ZH)**: 随着生成性人工智能（GenAI）代理在企业环境中的应用日益普遍，它们引入了与传统系统截然不同的安全挑战。这些代理不仅包括LLM，还能推理、记忆和行动，通常在最少的人类监督下工作。本文提出了一个针对GenAI代理的全面威胁模型，重点关注它们的自主权、持久内存访问、复杂推理和工具整合如何创造新的风险。这项研究工作识别出了9个主要威胁，并在五个关键领域进行了组织：认知架构漏洞、时间持久威胁、操作执行漏洞、信任边界违规和治理规避。这些威胁不仅是理论上的，还带来了实际挑战，如延迟的利用性、跨系统传播、跨系统横向移动和难以用现有框架和标准方法检测的微妙目标不一致。为了解决这些问题，研究工作提出了两个互补框架：ATFAA — 高级自主AI代理威胁框架，以及SHIELD，一个提出实际缓解策略的框架，旨在减少企业的暴露风险。虽然这项工作基于对LLM和AI安全的现有研究，但重点在于代理的不同之处及其重要性。最终，这项研究认为，生成性人工智能代理需要一种新的安全视角。如果不适应其独特的架构和行为，我们可能会将一个强大的新工具转变为企业的严重负担。 

---
# Human-Centered AI and Autonomy in Robotics: Insights from a Bibliometric Study 

**Title (ZH)**: 以人为本的AI与机器人自主性：基于文献计量学的研究洞察 

**Authors**: Simona Casini, Pietro Ducange, Francesco Marcelloni, Lorenzo Pollini  

**Link**: [PDF](https://arxiv.org/pdf/2504.19848)  

**Abstract**: The development of autonomous robotic systems offers significant potential for performing complex tasks with precision and consistency. Recent advances in Artificial Intelligence (AI) have enabled more capable intelligent automation systems, addressing increasingly complex challenges. However, this progress raises questions about human roles in such systems. Human-Centered AI (HCAI) aims to balance human control and automation, ensuring performance enhancement while maintaining creativity, mastery, and responsibility. For real-world applications, autonomous robots must balance task performance with reliability, safety, and trustworthiness. Integrating HCAI principles enhances human-robot collaboration and ensures responsible operation.
This paper presents a bibliometric analysis of intelligent autonomous robotic systems, utilizing SciMAT and VOSViewer to examine data from the Scopus database. The findings highlight academic trends, emerging topics, and AI's role in self-adaptive robotic behaviour, with an emphasis on HCAI architecture. These insights are then projected onto the IBM MAPE-K architecture, with the goal of identifying how these research results map into actual robotic autonomous systems development efforts for real-world scenarios. 

**Abstract (ZH)**: 自主机器人系统的发展为精确和一致地执行复杂任务提供了重要潜力。近年来，人工智能（AI）的进步使更强大的智能自动化系统成为可能，以应对日益复杂的挑战。然而，这一进展引发了关于此类系统中人类角色的问题。以人为中心的人工智能（HCAI）旨在平衡人类控制与自动化，确保性能提升的同时保持创造力、专业知识和责任感。在实际应用中，自主机器人必须在任务性能、可靠性和信任度之间达到平衡。将HCAI原则集成可以增强人机合作并确保负责任的操作。

本文通过SciMAT和VOSViewer对Scopus数据库中的数据进行文献计量分析，研究智能自主机器人系统的学术趋势、新兴主题以及AI在自我适应机器人行为中的作用，重点关注HCAI架构。这些见解随后被投射到IBM MAPE-K架构上，旨在确定这些研究结果如何映射到实际的自主机器人系统开发努力中，以应对现实世界的情景。 

---
# Model-based controller assisted domain randomization in deep reinforcement learning: application to nonlinear powertrain control 

**Title (ZH)**: 基于模型的控制器辅助领域随机化在深度强化学习中的应用：以非线性动力总成控制为例 

**Authors**: Heisei Yonezawa, Ansei Yonezawa, Itsuro Kajiwara  

**Link**: [PDF](https://arxiv.org/pdf/2504.19715)  

**Abstract**: Complex mechanical systems such as vehicle powertrains are inherently subject to multiple nonlinearities and uncertainties arising from parametric variations. Modeling and calibration errors are therefore unavoidable, making the transfer of control systems from simulation to real-world systems a critical challenge. Traditional robust controls have limitations in handling certain types of nonlinearities and uncertainties, requiring a more practical approach capable of comprehensively compensating for these various constraints. This study proposes a new robust control approach using the framework of deep reinforcement learning (DRL). The key strategy lies in the synergy among domain randomization-based DRL, long short-term memory (LSTM)-based actor and critic networks, and model-based control (MBC). The problem setup is modeled via the latent Markov decision process (LMDP), a set of vanilla MDPs, for a controlled system subject to uncertainties and nonlinearities. In LMDP, the dynamics of an environment simulator is randomized during training to improve the robustness of the control system to real testing environments. The randomization increases training difficulties as well as conservativeness of the resultant control system; therefore, progress is assisted by concurrent use of a model-based controller based on a nominal system model. Compared to traditional DRL-based controls, the proposed controller design is smarter in that we can achieve a high level of generalization ability with a more compact neural network architecture and a smaller amount of training data. The proposed approach is verified via practical application to active damping for a complex powertrain system with nonlinearities and parametric variations. Comparative tests demonstrate the high robustness of the proposed approach. 

**Abstract (ZH)**: 基于深度强化学习的新型鲁棒控制方法：处理复杂动力系统非线性和不确定性 

---
# Transformation & Translation Occupancy Grid Mapping: 2-Dimensional Deep Learning Refined SLAM 

**Title (ZH)**: transformations & translation occupancy grid mapping: 二维深度学习优化的SLAM 

**Authors**: Leon Davies, Baihua Li, Mohamad Saada, Simon Sølvsten, Qinggang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.19654)  

**Abstract**: SLAM (Simultaneous Localisation and Mapping) is a crucial component for robotic systems, providing a map of an environment, the current location and previous trajectory of a robot. While 3D LiDAR SLAM has received notable improvements in recent years, 2D SLAM lags behind. Gradual drifts in odometry and pose estimation inaccuracies hinder modern 2D LiDAR-odometry algorithms in large complex environments. Dynamic robotic motion coupled with inherent estimation based SLAM processes introduce noise and errors, degrading map quality. Occupancy Grid Mapping (OGM) produces results that are often noisy and unclear. This is due to the fact that evidence based mapping represents maps according to uncertain observations. This is why OGMs are so popular in exploration or navigation tasks. However, this also limits OGMs' effectiveness for specific mapping based tasks such as floor plan creation in complex scenes. To address this, we propose our novel Transformation and Translation Occupancy Grid Mapping (TT-OGM). We adapt and enable accurate and robust pose estimation techniques from 3D SLAM to the world of 2D and mitigate errors to improve map quality using Generative Adversarial Networks (GANs). We introduce a novel data generation method via deep reinforcement learning (DRL) to build datasets large enough for training a GAN for SLAM error correction. We demonstrate our SLAM in real-time on data collected at Loughborough University. We also prove its generalisability on a variety of large complex environments on a collection of large scale well-known 2D occupancy maps. Our novel approach enables the creation of high quality OGMs in complex scenes, far surpassing the capabilities of current SLAM algorithms in terms of quality, accuracy and reliability. 

**Abstract (ZH)**: 2D LiDAR SLAM中的变换和平移占位格网 mapping (Transformation and Translation Occupancy Grid Mapping, TT-OGM): 基于生成对抗网络的SLAM误差校正方法 

---
# VCM: Vision Concept Modeling Based on Implicit Contrastive Learning with Vision-Language Instruction Fine-Tuning 

**Title (ZH)**: 基于隐式对比学习和视觉语言指令微调的视觉概念模型 

**Authors**: Run Luo, Renke Shan, Longze Chen, Ziqiang Liu, Lu Wang, Min Yang, Xiaobo Xia  

**Link**: [PDF](https://arxiv.org/pdf/2504.19627)  

**Abstract**: Large Vision-Language Models (LVLMs) are pivotal for real-world AI tasks like embodied intelligence due to their strong vision-language reasoning abilities. However, current LVLMs process entire images at the token level, which is inefficient compared to humans who analyze information and generate content at the conceptual level, extracting relevant visual concepts with minimal effort. This inefficiency, stemming from the lack of a visual concept model, limits LVLMs' usability in real-world applications. To address this, we propose VCM, an end-to-end self-supervised visual concept modeling framework. VCM leverages implicit contrastive learning across multiple sampled instances and vision-language fine-tuning to construct a visual concept model without requiring costly concept-level annotations. Our results show that VCM significantly reduces computational costs (e.g., 85\% fewer FLOPs for LLaVA-1.5-7B) while maintaining strong performance across diverse image understanding tasks. Moreover, VCM enhances visual encoders' capabilities in classic visual concept perception tasks. Extensive quantitative and qualitative experiments validate the effectiveness and efficiency of VCM. 

**Abstract (ZH)**: 大规模视觉-语言模型（LVLMs）在体坛智能等实际AI任务中起到了关键作用，得益于它们强大的视觉-语言推理能力。然而，当前的LVLMs在处理整个图像时是在tokens级别进行的，这与人类在概念级别分析信息和生成内容的方式相比缺乏效率，人类能够以最小的努力提取相关的视觉概念。这种低效率源于缺乏视觉概念模型，限制了LVLMs在实际应用中的可用性。为了解决这个问题，我们提出了VCM，一个端到端的自监督视觉概念建模框架。VCM利用多个采样实例间的隐式对比学习和视觉-语言微调来构建视觉概念模型，而无需昂贵的概念级别注释。实验结果表明，VCM在显著降低计算成本（例如，LLaVA-1.5-7B减少85%的FLOPs）的同时，仍能保持在各种图像理解任务中的强大性能。此外，VCM增强了视觉编码器在经典视觉概念感知任务中的能力。大量的定量和定性实验验证了VCM的有效性和高效性。 

---
# Dynamic Action Interpolation: A Universal Approach for Accelerating Reinforcement Learning with Expert Guidance 

**Title (ZH)**: 动态动作插值：一种借助专家指导加速强化学习的通用方法 

**Authors**: Wenjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.18766)  

**Abstract**: Reinforcement learning (RL) suffers from severe sample inefficiency, especially during early training, requiring extensive environmental interactions to perform competently. Existing methods tend to solve this by incorporating prior knowledge, but introduce significant architectural and implementation complexity. We propose Dynamic Action Interpolation (DAI), a universal yet straightforward framework that interpolates expert and RL actions via a time-varying weight $\alpha(t)$, integrating into any Actor-Critic algorithm with just a few lines of code and without auxiliary networks or additional losses. Our theoretical analysis shows that DAI reshapes state visitation distributions to accelerate value function learning while preserving convergence guarantees. Empirical evaluations across MuJoCo continuous control tasks demonstrate that DAI improves early-stage performance by over 160\% on average and final performance by more than 50\%, with the Humanoid task showing a 4$\times$ improvement early on and a 2$\times$ gain at convergence. These results challenge the assumption that complex architectural modifications are necessary for sample-efficient reinforcement learning. 

**Abstract (ZH)**: 动态动作插值（DAI）：一种加速值函数学习的通用简便框架 

---
# Deep Learning with Pretrained 'Internal World' Layers: A Gemma 3-Based Modular Architecture for Wildfire Prediction 

**Title (ZH)**: 基于Gemma 3的模块化架构：带有预训练“内部世界”层的深度学习方法用于野火预测 

**Authors**: Ayoub Jadouli, Chaker El Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2504.18562)  

**Abstract**: Deep learning models, especially large Transformers, carry substantial "memory" in their intermediate layers -- an \emph{internal world} that encodes a wealth of relational and contextual knowledge. This work harnesses that internal world for wildfire occurrence prediction by introducing a modular architecture built upon Gemma 3, a state-of-the-art multimodal model. Rather than relying on Gemma 3's original embedding and positional encoding stacks, we develop a custom feed-forward module that transforms tabular wildfire features into the hidden dimension required by Gemma 3's mid-layer Transformer blocks. We freeze these Gemma 3 sub-layers -- thus preserving their pretrained representation power -- while training only the smaller input and output networks. This approach minimizes the number of trainable parameters and reduces the risk of overfitting on limited wildfire data, yet retains the benefits of Gemma 3's broad knowledge. Evaluations on a Moroccan wildfire dataset demonstrate improved predictive accuracy and robustness compared to standard feed-forward and convolutional baselines. Ablation studies confirm that the frozen Transformer layers consistently contribute to better representations, underscoring the feasibility of reusing large-model mid-layers as a learned internal world. Our findings suggest that strategic modular reuse of pretrained Transformers can enable more data-efficient and interpretable solutions for critical environmental applications such as wildfire risk management. 

**Abstract (ZH)**: 深度学习模型，尤其是大型Transformer模型，在其中间层携带了大量的“记忆”——一种内部世界，编码了丰富的关系性和背景知识。本研究通过在Gemma 3这一先进多模态模型基础上构建一个模块化架构，利用这种内部世界进行野火发生预测。我们开发了一个自定义的前馈模块，将表格形式的野火特征转换为Gemma 3中间Transformer层所需隐藏维度。我们冻结了这些Gemma 3子层，从而保留了它们预训练的表示能力，仅训练较小的输入和输出网络。这种方法减少了可训练参数的数量，并降低了在有限的野火数据上过拟合的风险，同时保留了Gemma 3广泛知识的益处。在摩洛哥野火数据集上的评估表明，与标准前馈和卷积基准方法相比，具有改进的预测准确性和鲁棒性。消融研究证实，冻结的Transformer层始终有助于更好的表示，突显了在学习内部世界中重用大模型中间层的可行性。我们的研究结果表明，战略性地重用预训练的Transformer可以为关键的环境应用（如野火风险管理）提供更高效且可解释的解决方案。 

---
